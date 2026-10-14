"""Refresh the calling Kaggle team's complete RSNA submission history.

Requires the official ``kagglesdk`` package (installed with ``kaggle``) and
KAGGLE_API_TOKEN in the environment. This script only calls read endpoints.
Optional --sdk-path selects an existing installation. Raw API evidence is kept
outside the repository; public exports omit descriptions and private scores.

Example, from the repository root:
    python scripts/refresh_submission_history.py --require-ref 57194274
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
KST = timezone(timedelta(hours=9))
HISTORY_COLUMNS = [
    "submission_ref", "submitted_utc", "submitted_kst", "utc_day",
    "submitted_by", "team", "notebook_owner", "notebook_slug",
    "script_version_id", "notebook_url", "public_score", "api_status",
    "outcome", "hidden_scoring_runtime_seconds", "runtime_availability",
]
DAY_COLUMNS = ["utc_day", "submission_count", "scored_count",
               "best_displayed_public_score", "submitters"]
CANDIDATE_COLUMNS = [
    "notebook_ref", "submission_count", "scored_count",
    "best_displayed_public_score", "latest_displayed_public_score",
    "first_submitted_utc", "last_submitted_utc", "latest_script_version_id",
    "hidden_scoring_runtime_seconds",
]


def timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def notebook_reference(value: str) -> tuple[str, str, str, str]:
    """Reconstruct a public code URL, retaining only a numeric version ID."""
    parsed = urlparse(value)
    if parsed.netloc and parsed.netloc.lower() not in {"www.kaggle.com", "kaggle.com"}:
        return "", "", "", ""
    if parsed.scheme and parsed.scheme not in {"http", "https"}:
        return "", "", "", ""
    match = re.fullmatch(r"/code/([A-Za-z0-9_-]+)/([A-Za-z0-9_-]+)/?", parsed.path)
    if not match:
        return "", "", "", ""
    owner, slug = match.groups()
    versions = parse_qs(parsed.query).get("scriptVersionId", [])
    version = versions[0] if versions and versions[0].isdigit() else ""
    url = f"https://www.kaggle.com/code/{owner}/{slug}"
    if version:
        url += "?scriptVersionId=" + version
    return owner, slug, version, url


def clean_submission(record: dict) -> dict:
    date = timestamp(record["date"]).astimezone(timezone.utc)
    owner, slug, version, url = notebook_reference(record.get("url") or "")
    score = record.get("publicScore") or ""
    if score:
        try:
            number = Decimal(score)
        except InvalidOperation as exc:
            raise ValueError("Unexpected public score format") from exc
        if not number.is_finite() or not 0 <= number <= 1:
            raise ValueError("Unexpected public score range")
    status = (record.get("status") or "PENDING").upper()
    error = record.get("errorDescription") or ""
    if error:
        outcome = "rerun_error" if "rerun" in error.lower() else "error"
    elif score:
        outcome = "scored"
    elif status in {"PENDING", "QUEUED", "RUNNING"}:
        outcome = "pending"
    else:
        outcome = "unscored"
    return dict(zip(HISTORY_COLUMNS, [
        str(record["ref"]), date.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        date.astimezone(KST).isoformat(timespec="milliseconds"), date.date().isoformat(),
        record.get("submittedByRef") or record.get("submittedBy") or "",
        record.get("teamName") or "", owner, slug, version, url, score, status,
        outcome, "", "not_exposed_by_submission_api",
    ]))


def best_score(rows: list[dict]) -> str:
    scores = [row["public_score"] for row in rows if row["public_score"]]
    return max(scores, key=Decimal) if scores else ""


def grouped_exports(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    days, candidates = defaultdict(list), defaultdict(list)
    for row in rows:
        days[row["utc_day"]].append(row)
        if row["notebook_owner"] and row["notebook_slug"]:
            candidates[row["notebook_owner"] + "/" + row["notebook_slug"]].append(row)
    day_rows = [{
        "utc_day": day, "submission_count": len(group),
        "scored_count": sum(row["outcome"] == "scored" for row in group),
        "best_displayed_public_score": best_score(group),
        "submitters": ",".join(sorted({row["submitted_by"] for row in group})),
    } for day, group in sorted(days.items())]
    candidate_rows = [{
        "notebook_ref": ref, "submission_count": len(group),
        "scored_count": sum(row["outcome"] == "scored" for row in group),
        "best_displayed_public_score": best_score(group),
        "latest_displayed_public_score": group[0]["public_score"],
        "first_submitted_utc": group[-1]["submitted_utc"],
        "last_submitted_utc": group[0]["submitted_utc"],
        "latest_script_version_id": group[0]["script_version_id"],
        "hidden_scoring_runtime_seconds": "",
    } for ref, group in sorted(candidates.items())]
    return day_rows, candidate_rows


def csv_text(columns: list[str], rows: list[dict]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def save_evidence(directory: Path, observed: datetime, evidence: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = directory / (observed.strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    # Restrict access from creation; raw responses may contain private fields.
    descriptor = os.open(snapshot, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(evidence, stream, indent=2)
        stream.write("\n")


def read_history(args, evidence: dict) -> tuple[list[dict], dict, int]:
    from kagglesdk.kaggle_client import KaggleClient
    from kagglesdk.competitions.types.competition_api_service import (
        ApiGetSubmissionLimitsRequest, ApiListSubmissionsRequest,
    )

    def capture(response):
        evidence["api_responses"].append({
            "observed_at_utc": datetime.now(timezone.utc).isoformat(),
            "endpoint": response.url, "http_status": response.status_code,
            "body": response.json(),
        })

    with KaggleClient(api_token=os.environ["KAGGLE_API_TOKEN"],
                      response_processor=capture) as client:
        api = client.competitions.competition_api_client
        limit_request = ApiGetSubmissionLimitsRequest()
        limit_request.competition_name = args.competition
        # A new submission arriving during traversal can invalidate the count.
        for attempt in range(3):
            before = api.get_submission_limits(limit_request)
            records, seen_tokens, pages = [], set(), 0
            token = ""
            while True:
                request = ApiListSubmissionsRequest()
                request.competition_name = args.competition
                request.page_size = 100
                if token:
                    request.page_token = token
                else:
                    request.page = 1
                response = api.list_submissions(request)
                pages += 1
                records.extend(type(item).to_dict(item, ignore_defaults=False)
                               for item in response.submissions or [] if item is not None)
                token = response.next_page_token
                if not token:
                    break
                if token in seen_tokens or pages >= 1000:
                    raise ValueError("Submission pagination repeated a token or exceeded its bound")
                seen_tokens.add(token)
            after = api.get_submission_limits(limit_request)
            if before.num_total != after.num_total:
                continue
            refs = [str(record["ref"]) for record in records]
            if len(refs) != len(set(refs)):
                raise ValueError("Duplicate submission references returned by the API")
            if len(records) != after.num_total:
                raise ValueError("Submission count does not match the official lifetime total; exports unchanged")
            if not set(args.require_ref).issubset(refs):
                if attempt < 2:
                    continue
                raise ValueError("A required submission is not yet visible; exports unchanged")
            limits = {"numToday": after.num_today, "numTotal": after.num_total,
                      "numAllowedNow": after.num_allowed_now, "limitedByTotal": after.limited_by_total}
            return records, limits, pages
    raise ValueError("Submission count changed during all three snapshots; exports unchanged")


def current_markdown(summary: dict, rows: list[dict]) -> str:
    text = [
        f"Current RSNA Knee submission history for `{summary['account']}`, {summary['team']}, "
        f"queried {summary['observed_at_utc']}.", "",
        f"The authenticated official API returned **{len(rows)} unique team submissions**, "
        f"matching its lifetime total: **{summary['scored_count']} scored**, "
        f"**{summary['pending_count']} pending**, and **{summary['error_count']} errors**. "
        f"Best displayed public score: **{summary['best_displayed_public_score']}**.", "",
        f"Official daily allowance: **{summary['official_submission_limits']['numToday']} used**, "
        f"**{summary['official_submission_limits']['numAllowedNow']} available**. "
        f"Next reset: {summary['next_submission_reset_utc']} / {summary['next_submission_reset_kst']}.", "",
        "Scores retain the precision returned by Kaggle. Hidden scoring runtimes are not exposed. "
        "Blank scores on pending attempts are not measured results.", "",
        "| Submitted UTC | Submission | Notebook | Status | Public score |",
        "|---|---:|---|---|---:|",
    ]
    for row in rows[:10]:
        ref = row["notebook_owner"] + "/" + row["notebook_slug"] if row["notebook_owner"] else ""
        text.append(f"|{row['submitted_utc']}|{row['submission_ref']}|{ref}|"
                    f"{row['api_status']} / {row['outcome']}|{row['public_score']}|")
    text.extend(["", "Refresh with `scripts/refresh_submission_history.py`; "
                 "authentication uses `KAGGLE_API_TOKEN` from the environment. "
                 "Private raw snapshots preserve each API observation; the initial dated audit remains in "
                 "`submission_history.md`.", ""])
    return "\n".join(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition", default="rsna-knee-abnormality-detection")
    parser.add_argument("--account", default="gwey1025")
    parser.add_argument("--team", default="Team Alicia")
    parser.add_argument("--sdk-path", type=Path)
    parser.add_argument("--output-dir", action="append", type=Path,
                        help="Clean export directory; repeat to update additional mirrors")
    parser.add_argument("--raw-dir", type=Path,
                        default=ROOT.parent / ".private/kaggle_probe/submission_history_refresh")
    parser.add_argument("--require-ref", action="append", default=[],
                        help="Require a known submission reference before replacing exports")
    args = parser.parse_args()
    if not os.environ.get("KAGGLE_API_TOKEN"):
        parser.error("Set KAGGLE_API_TOKEN in the environment")
    if args.sdk_path:
        sys.path.insert(0, str(args.sdk_path.resolve()))
    raw_dir = args.raw_dir.resolve()
    outputs = [path.resolve() for path in (args.output_dir or [ROOT / "research"])]
    if raw_dir.is_relative_to(ROOT) or any(raw_dir.is_relative_to(path) for path in outputs):
        parser.error("Raw API evidence must be outside the repository and clean export directories")
    now = datetime.now(timezone.utc)
    evidence = {"competition": args.competition, "observed_at_utc": now.isoformat(), "api_responses": []}
    evidence_saved = False
    try:
        records, limits, pages = read_history(args, evidence)
        rows = sorted((clean_submission(record) for record in records),
                      key=lambda row: (row["submitted_utc"], int(row["submission_ref"])), reverse=True)
        if {row["team"] for row in rows if row["team"]} - {args.team}:
            raise ValueError("Returned team does not match the expected team; exports unchanged")
        current_refs = {row["submission_ref"] for row in rows}
        for output in outputs:
            previous = output / "submission_history.csv"
            if previous.exists():
                with previous.open(encoding="utf-8", newline="") as stream:
                    previous_refs = {row["submission_ref"] for row in csv.DictReader(stream)}
                if not previous_refs.issubset(current_refs):
                    raise ValueError("Previously exported submissions are missing; exports unchanged")
        days, candidates = grouped_exports(rows)
        outcomes = Counter(row["outcome"] for row in rows)
        observed = datetime.now(timezone.utc)
        reset = (observed + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        summary_path = outputs[0] / "submission_history.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        if summary and "initial_audit" not in summary:
            summary["initial_audit"] = {key: summary[key] for key in [
                "observed_at_utc", "submission_count", "scored_count", "rerun_error_count",
                "pagination_validation", "official_submission_limits", "official_accelerator_quota",
            ] if key in summary}
        summary.update({
            "competition": args.competition, "account": args.account, "team": args.team,
            "observed_at_utc": observed.isoformat(),
            "submission_count": len(rows), "unique_submission_refs": len(current_refs),
            "unique_notebook_refs": len(candidates), "scored_count": outcomes["scored"],
            "pending_count": outcomes["pending"], "rerun_error_count": outcomes["rerun_error"],
            "error_count": outcomes["error"] + outcomes["rerun_error"],
            "unscored_count": outcomes["unscored"],
            "best_displayed_public_score": float(best_score(rows)) if best_score(rows) else None,
            "first_submission_utc": rows[-1]["submitted_utc"] if rows else None,
            "latest_submission_utc": rows[0]["submitted_utc"] if rows else None,
            "score_precision_warning": "Scores retain Kaggle's displayed precision; unrounded gains are not exposed.",
            "pagination_validation": {"requested_page_size": 100, "pages_returned": pages,
                "records_returned": len(rows), "matches_official_num_total": True,
                "all_refs_unique": True, "previous_exported_refs_preserved": True,
                "completeness_basis": "Distinct references equal official lifetime total; only official next_page_token is traversed."},
            "official_submission_limits": limits, "current_quota_utc_day": observed.date().isoformat(),
            "next_submission_reset_utc": reset.isoformat().replace("+00:00", "Z"),
            "next_submission_reset_kst": reset.astimezone(KST).isoformat(),
            "accelerator_quota_refreshed": False,
            "runtime_availability": "Submission list does not expose hidden-scoring runtime.",
            "submitter_counts": dict(Counter(row["submitted_by"] for row in rows)),
            "pending_submission_refs": [row["submission_ref"] for row in rows if row["outcome"] == "pending"],
        })
        initial_raw = ROOT.parent / ".private/kaggle_probe/submissions_sdk_page1.json"
        if initial_raw.exists():
            summary["initial_raw_evidence_sha256"] = hashlib.sha256(initial_raw.read_bytes()).hexdigest()
        exports = {
            "submission_history.csv": csv_text(HISTORY_COLUMNS, rows),
            "submission_history_by_utc_day.csv": csv_text(DAY_COLUMNS, days),
            "submission_history_candidates.csv": csv_text(CANDIDATE_COLUMNS, candidates),
            "submission_history.json": json.dumps(summary, indent=2) + "\n",
            "submission_history_current.md": current_markdown(summary, rows),
        }
        save_evidence(raw_dir, now, evidence)
        evidence_saved = True
        for output in outputs:
            for name, text in exports.items():
                atomic_text(output / name, text)
        print(json.dumps({key: summary[key] for key in ["submission_count", "scored_count",
            "pending_count", "error_count", "best_displayed_public_score",
            "official_submission_limits", "pending_submission_refs"]}))
        return 0
    except Exception as exc:
        # SDK exceptions can embed response content or URLs. Never print them.
        code = getattr(getattr(exc, "response", None), "status_code", None)
        if isinstance(exc, (ValueError, ImportError)):
            print(f"History refresh failed: {exc}", file=sys.stderr)
        else:
            print(f"Kaggle read failed ({type(exc).__name__}, HTTP {code}); exports unchanged", file=sys.stderr)
        return 1
    finally:
        if not evidence_saved:
            save_evidence(raw_dir, now, evidence)


if __name__ == "__main__":
    raise SystemExit(main())
