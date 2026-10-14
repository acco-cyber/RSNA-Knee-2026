#!/usr/bin/env python3
"""Build private, reproducible L2 targets without fitting on the 58 gold studies.

The CLI can first prepare report-only annotation batches with --prepare-batches.
Inputs/outputs must remain outside a public repository; this file contains no data.
No score is promised. Gold diagnostics are development diagnostics, not clean CV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

UID = "StudyInstanceUID"
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
# Frozen shares recovered from the actual round-2 training source, not fitted here.
REPORT_SHARE = {"ACL": .6, "MCL": .6, "Medial Meniscus": .5, "Lateral Meniscus": .4,
                "Medial OA": .4, "Lateral OA": .2, "PF OA": .4, "Effusion": .2,
                "Synovitis": .2, "Baker's": .4, "Contusion": .2, "Fracture": .25}
STATES = {"positive", "negative", "uncertain", "unmentioned"}
RUBRIC_URL = "https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343"

GUIDE = """# Private report annotation guide for RSNA Knee L2

Read reports only. Do not consult expert labels, image-model predictions, leaderboard
scores, or previous parser votes. Preserve the study UID exactly and label one knee.
Use the host's rubric: {rubric_url}

This is a report-derived weak label, not an expert interpretation of the MRI.
Translate mentally where needed, retaining the original short evidence span.
Do not invent findings, severity, trauma, laterality, chronicity, or image evidence.

## Competition thresholds

- ACL: substantial partial/full tear; degeneration alone is negative.
- MCL: substantial acute tear; mild sprain/chronic change is negative.
- Menisci: definite tear/morphologic disruption; internal degeneration is negative.
- OA compartments: extensive deep cartilage loss (approximately >=1 cm, >50% depth).
- Effusion and Baker's: moderate/large collections; small/minimal is negative.
- Synovitis: synovial thickening/inflammation; effusion alone does not establish it.
- Contusion: impact-related marrow edema without a fracture line.
- Fracture: acute cortical break/fracture line.
- Host-borderline findings are negative. Unresolved report wording remains uncertain.

## Four report states

`positive`: evidence explicitly meets the criterion. Default probability 0.95.
`negative`: explicit absence OR explicitly below the host criterion. Default 0.05.
`uncertain`: finding/severity is unresolved from the text. Default probability 0.50;
confidence <=0.40. A suspected diagnosis does not become definite because another
finding is present. Prefer this state when the necessary severity is not specified.
`unmentioned`: no relevant evidence. Probability null, confidence 0, evidence empty.

An unmentioned finding must never be labeled negative. A normal-structure sentence
can establish negative evidence only for structures it actually covers. Historical
surgery/graft repair alone does not establish a current tear. Different compartments
and medial/lateral structures need separate evidence. Generic degeneration/OA without
compartment or severity should not be assigned to all compartments as positive.

Confidence reflects how clearly the REPORT resolves the host criterion, not your
medical expertise or how likely the abnormality is in general. Use <=0.95. For a
clear explicit statement, 0.85-0.95 is appropriate; weaker wording needs less.
Every positive, negative, or uncertain finding requires a short literal original
report substring as evidence. Do not paraphrase the evidence. For an unmentioned
finding, evidence is the empty string. Keep evidence short; no whole-report copying.

## Required output JSON

Write one file for the assigned batch to the specified private annotation folder.
Do not modify the input batch, put reports in the output, or publish private data.

```json
{{
  "batch_id": "b000",
  "annotations": [
    {{
      "StudyInstanceUID": "EXACT_UID_FROM_INPUT",
      "findings": {{
        "ACL": {{"state": "negative", "probability": 0.05,
                 "confidence": 0.90, "evidence": "literal short report span"}},
        "MCL": {{"state": "unmentioned", "probability": null,
                 "confidence": 0.0, "evidence": ""}}
      }}
    }}
  ]
}}
```

Each study must include all twelve exact canonical finding keys: {labels}.
The example is abbreviated; actual output cannot omit other finding keys.
Probability/confidence must be finite numbers in [0,1]. Positive probability must
be >0.5; negative <0.5. Unmentioned probability must be null. Each batch output
must cover exactly its assigned UIDs once, including the final smaller batch.

Never use expert gold labels to correct your output. Flag uncertainty honestly.
The builder checks UIDs, complete finding keys, states, probability ranges,
confidence limits, missing-as-abstention, and literal evidence spans.
"""


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def input_record(path: Path, role: str, **extra) -> dict:
    return {"role": role, "file": str(path.resolve()), "sha256": digest(path), **extra}


def read_table(path: Path, *, allow_extra_ids=False, universe=None) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={UID: str})
    if UID not in df:
        raise ValueError(f"{path}: missing {UID}; no positional/fuzzy joining allowed")
    if df[UID].isna().any() or df[UID].duplicated().any():
        raise ValueError(f"{path}: missing or duplicate study UIDs")
    missing = set(LABELS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing canonical findings {sorted(missing)}")
    df = df.set_index(UID)
    for c in LABELS:
        df[c] = pd.to_numeric(df[c], errors="raise")
    a = df[LABELS].to_numpy(float)
    finite_or_missing = np.isfinite(a) | np.isnan(a)
    if not finite_or_missing.all() or ((a < 0) | (a > 1)).any():
        raise ValueError(f"{path}: predictions must be finite probabilities in [0,1] or missing")
    if universe is not None and not allow_extra_ids:
        extra = set(df.index) - set(universe)
        if extra:
            raise ValueError(f"{path}: {len(extra)} UIDs are outside the competition train universe")
    return df


def load_train(path: Path) -> pd.DataFrame:
    tr = read_table(path)
    if "Report" not in tr or tr.Report.isna().any():
        raise ValueError("Training table must include a report for each study")
    partial = tr[LABELS].notna().any(axis=1) & ~tr[LABELS].notna().all(axis=1)
    if partial.any():
        raise ValueError("Partially expert-labeled rows need an explicit new policy")
    known = tr[LABELS].to_numpy(float)
    if not (np.isnan(known) | (known == 0) | (known == 1)).all():
        raise ValueError("Expert labels must be binary or missing")
    return tr


def normalized_span(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def prepare_batches(train_path: Path, batch_dir: Path, guide_path: Path, batch_size: int) -> dict:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    tr = load_train(train_path)
    gold = tr[LABELS].notna().all(axis=1)
    weak = tr.loc[~gold, ["Report"]]
    batch_dir.mkdir(parents=True, exist_ok=True)
    guide_path.parent.mkdir(parents=True, exist_ok=True)
    guide_path.write_text(GUIDE.format(rubric_url=RUBRIC_URL, labels=json.dumps(LABELS)), encoding="utf-8")
    assignments = []
    for start in range(0, len(weak), batch_size):
        part = weak.iloc[start:start + batch_size]
        batch_id = f"b{start // batch_size:03d}"
        body = {"batch_id": batch_id, "source_train_sha256": digest(train_path),
                "guide_sha256": digest(guide_path), "labels": LABELS,
                "studies": [{UID: str(uid), "Report": str(row.Report)} for uid, row in part.iterrows()]}
        dest = batch_dir / f"{batch_id}.json"
        serialized = json.dumps(body, ensure_ascii=False, indent=2) + "\n"
        if dest.exists() and dest.read_text(encoding="utf-8") != serialized:
            raise ValueError(f"Refusing to overwrite a changed existing assignment: {dest}")
        dest.write_text(serialized, encoding="utf-8")
        assignments.append({"batch_id": batch_id, "file": str(dest), "studies": len(part),
                            "sha256": digest(dest)})
    manifest = {"train_sha256": digest(train_path), "gold_excluded": int(gold.sum()),
                "report_only_studies": len(weak), "batch_size": batch_size,
                "batches": assignments, "guide_sha256": digest(guide_path)}
    (batch_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def regenerate_a1(train_path: Path, consensus_path: Path, destination: Path,
                  fold_directories: list[Path] | None = None) -> dict:
    """Reproduce recovered A1 lines 87-109, including report-group folds."""
    from sklearn.model_selection import GroupKFold
    tr = load_train(train_path)
    cons = read_table(consensus_path, universe=tr.index)
    gold = tr[LABELS]
    is_gold = gold.notna().all(axis=1)
    soft = cons.reindex(tr.index)[LABELS].astype(float)
    missing_before_legacy_fill = int(soft.isna().any(axis=1).sum())
    # This reproduces historical A1 only. The new combiner itself never treats
    # missing annotations as negatives and records this legacy imputation.
    soft = soft.fillna(soft.mean()).clip(.01, .99)
    out = soft.copy()
    out.insert(0, "is_gold", is_gold.astype(int).to_numpy())
    out["fold"] = -1
    weak = ~is_gold.to_numpy()
    grp = tr.Report.map(lambda t: re.sub(r"\W+", "", str(t).lower())).to_numpy()[weak]
    order = np.random.RandomState(0).permutation(weak.sum())
    folds = np.zeros(weak.sum(), int)
    for k, (_, va) in enumerate(GroupKFold(5).split(order, groups=grp[order])):
        folds[order[va]] = k
    out.loc[weak, "fold"] = folds
    regenerated_fold_mismatches = 0
    fold_records = []
    if fold_directories:
        authoritative = None
        expected = set(out.index[weak])
        for directory in fold_directories:
            parts = []
            for path in sorted(directory.glob("oof_r2_*_f[0-9].csv")):
                f = int(re.search(r"_f([0-9])\.csv$", path.name).group(1))
                df = read_table(path, universe=tr.index)
                if "set" not in df:
                    raise ValueError(f"{path}: missing held-out set provenance")
                ids = df.index[df["set"] == "val"]
                parts.append(pd.Series(f, index=ids, name="fold"))
                fold_records.append(input_record(path, "authoritative_student_fold_assignment", fold=f))
            if not parts:
                raise ValueError(f"No fold provenance under {directory}")
            actual = pd.concat(parts)
            if actual.index.duplicated().any() or set(actual.index) != expected:
                raise ValueError(f"{directory}: direct held-out sets must partition all 4349 report-only studies")
            actual = actual.reindex(out.index[weak])
            if authoritative is None:
                authoritative = actual
            elif not actual.equals(authoritative):
                raise ValueError("Round-2 prediction families disagree on their study-fold assignments")
        groups = pd.DataFrame({"report_group": grp, "fold": authoritative.to_numpy()})
        if (groups.groupby("report_group").fold.nunique() > 1).any():
            raise ValueError("Recovered held-out folds split duplicate normalized reports")
        regenerated_fold_mismatches = int((out.loc[weak, "fold"] != authoritative).sum())
        out.loc[weak, "fold"] = authoritative
    for c in LABELS:
        out[f"gold_{c}"] = gold[c]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        old = read_table(destination, universe=tr.index).reindex(tr.index)
        if not out.equals(old[out.columns]):
            # CSV round-trips may change floating point representation only;
            # preserve existing recovered bytes when values/folds are equivalent.
            non_fold = [c for c in out if c != "fold"]
            same_targets = np.allclose(out[non_fold].to_numpy(float), old[non_fold].to_numpy(float),
                                       rtol=1e-12, atol=1e-12, equal_nan=True)
            same_folds = np.array_equal(out.fold.to_numpy(), old.fold.to_numpy())
            if not same_targets or (not same_folds and not fold_directories):
                raise ValueError("Refusing to replace an A1 file with different targets/folds")
            if not same_folds:
                out.to_csv(destination)
    else:
        out.to_csv(destination)
    manifest = {"file": str(destination), "studies": len(out), "gold": int(is_gold.sum()),
            "missing_before_legacy_fill": missing_before_legacy_fill,
            "fold_counts": {str(k): int(v) for k, v in out.fold.value_counts().sort_index().items()},
            "train_sha256": digest(train_path), "consensus_sha256": digest(consensus_path),
            "a1_sha256": digest(destination), "regenerated_fold_mismatches": regenerated_fold_mismatches,
            "fold_assignment": "direct recovered round-2 held-out sets" if fold_directories else "regenerated historical GroupKFold recipe",
            "fold_provenance": fold_records}
    destination.with_suffix(".regeneration.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def annotation_file(path: Path, tr: pd.DataFrame, batch_dir: Path | None):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("annotations"), list):
        raise ValueError(f"{path}: expected batch_id and annotations list")
    batch_id = data.get("batch_id")
    if not isinstance(batch_id, str) or not re.fullmatch(r"b\d{3}", batch_id):
        raise ValueError(f"{path}: invalid batch ID")
    rows = data["annotations"]
    uids = [str(row.get(UID, "")) for row in rows]
    if len(uids) != len(set(uids)):
        raise ValueError(f"{path}: duplicate annotation UIDs")
    if batch_dir is not None:
        assigned = json.loads((batch_dir / f"{batch_id}.json").read_text(encoding="utf-8"))
        expected = {str(s[UID]) for s in assigned["studies"]}
        if set(uids) != expected:
            raise ValueError(f"{path}: assignment coverage mismatch: missing {len(expected-set(uids))}, extra {len(set(uids)-expected)}")
    out = []
    for row, uid in zip(rows, uids):
        if uid not in tr.index or tr.loc[uid, LABELS].notna().all():
            raise ValueError(f"{path}: UID not in report-only training pool: {uid}")
        findings = row.get("findings", {})
        if set(findings) != set(LABELS):
            raise ValueError(f"{path}: {uid}: all twelve exact finding keys are required")
        report = normalized_span(str(tr.loc[uid, "Report"]))
        for c, item in findings.items():
            state = item.get("state")
            q = item.get("confidence")
            p = item.get("probability")
            evidence = item.get("evidence", "")
            if state not in STATES or not isinstance(evidence, str):
                raise ValueError(f"{path}: {uid}/{c}: invalid state/evidence")
            if isinstance(q, bool) or not isinstance(q, (int, float)) or not math.isfinite(q) or not 0 <= q <= .95:
                raise ValueError(f"{path}: {uid}/{c}: confidence must be in [0,.95]")
            if state == "unmentioned":
                if p is not None or q != 0 or evidence.strip():
                    raise ValueError(f"{path}: {uid}/{c}: unmentioned must abstain")
                p = np.nan
            else:
                if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
                    raise ValueError(f"{path}: {uid}/{c}: invalid probability")
                if state == "positive" and p <= .5 or state == "negative" and p >= .5:
                    raise ValueError(f"{path}: {uid}/{c}: probability contradicts state")
                if state == "uncertain" and q > .40:
                    raise ValueError(f"{path}: {uid}/{c}: unresolved evidence needs confidence <=.40")
                span = normalized_span(evidence)
                if not span or span not in report:
                    raise ValueError(f"{path}: {uid}/{c}: evidence is not a literal report span")
            out.append({UID: uid, "label": c, "state": state, "probability": p,
                        "confidence": float(q), "batch_id": batch_id})
    return out


def load_annotations(annotation_dir: Path | None, tr: pd.DataFrame, batch_dir: Path | None):
    probs = pd.DataFrame(np.nan, index=tr.index, columns=LABELS)
    conf = pd.DataFrame(0.0, index=tr.index, columns=LABELS)
    states = pd.DataFrame("source_missing", index=tr.index, columns=LABELS)
    records, all_rows, seen = [], [], set()
    if annotation_dir is not None and annotation_dir.exists():
        for path in sorted(annotation_dir.glob("b[0-9][0-9][0-9].json")):
            rows = annotation_file(path, tr, batch_dir)
            file_uids = {r[UID] for r in rows}
            if seen & file_uids:
                raise ValueError(f"Duplicate studies across annotation files: {path}")
            seen |= file_uids
            for row in rows:
                uid, c = row[UID], row["label"]
                probs.loc[uid, c] = row["probability"]
                conf.loc[uid, c] = row["confidence"]
                states.loc[uid, c] = row["state"]
            records.append(input_record(path, "new_report_annotations", studies=len(file_uids)))
            all_rows.extend(rows)
    return probs, conf, states, records, seen, all_rows


def load_round2(directory: Path, tr: pd.DataFrame, folds: pd.Series):
    val_parts, gold_parts, records = [], [], []
    files = sorted(directory.glob("oof_r2_*_f[0-9].csv"))
    if not files:
        raise ValueError(f"No round-2 prediction files under {directory}")
    covered_folds = set()
    gold_ids = set(tr.index[tr[LABELS].notna().all(axis=1)])
    for path in files:
        f = int(re.search(r"_f([0-9])\.csv$", path.name).group(1))
        if f in covered_folds:
            raise ValueError(f"Multiple architecture/fold files in one prediction family: {directory}, fold {f}")
        covered_folds.add(f)
        df = read_table(path, universe=tr.index)
        if "set" not in df or not set(df["set"]) <= {"val", "gold"}:
            raise ValueError(f"{path}: expected set=val/gold prediction provenance")
        val = df.loc[df["set"] == "val", LABELS]
        gd = df.loc[df["set"] == "gold", LABELS]
        if (folds.reindex(val.index) != f).any():
            raise ValueError(f"{path}: validation row does not match the frozen A1 fold")
        if not set(gd.index) <= gold_ids:
            raise ValueError(f"{path}: mislabeled gold predictions")
        val_parts.append(val)
        gold_parts.append(gd)
        records.append(input_record(path, "round2_prediction", model_fold=f,
                                    validation_studies=len(val), gold_studies=len(gd),
                                    upstream_teacher="in_sample_on_report_only_pool"))
    va = pd.concat(val_parts)
    if va.index.duplicated().any():
        raise ValueError(f"{directory}: a row has more than one direct OOF prediction")
    # Gold predictions are all held-out from the recovered student training script;
    # average fold models. This does not erase prior gold-set model selection.
    gd = pd.concat(gold_parts).groupby(level=0).mean()
    family = pd.concat([va, gd])
    if family.index.duplicated().any():
        raise ValueError(f"{directory}: gold/val UID collision")
    return family.reindex(tr.index)[LABELS], records


def available_mean(arrays: list[np.ndarray]):
    arr = np.stack(arrays)
    count = np.isfinite(arr).sum(0)
    return np.divide(np.nansum(arr, axis=0), count,
                     out=np.full(arr.shape[1:], np.nan), where=count > 0), count


def combine(base: pd.DataFrame, teacher: pd.DataFrame, families: list[pd.DataFrame],
            manual_p: pd.DataFrame, manual_q: pd.DataFrame, *, manual_share=1/7,
            teacher_share=.5, public_source_counts=None):
    b = base[LABELS].to_numpy(float)
    t = teacher[LABELS].to_numpy(float)
    # The teacher's all-0.5 row is the recovered failed-study sentinel.
    failed_teacher = np.isclose(t, .5, rtol=0, atol=1e-8).all(axis=1)
    t[failed_teacher] = np.nan
    if families:
        r2, r2_count = available_mean([f[LABELS].to_numpy(float) for f in families])
    else:
        r2, r2_count = np.full_like(b, np.nan), np.zeros_like(b, int)
    has_t, has_r2 = np.isfinite(t), np.isfinite(r2)
    image = np.full_like(b, np.nan)
    image[has_t & has_r2] = teacher_share*t[has_t & has_r2] + (1-teacher_share)*r2[has_t & has_r2]
    image[has_t & ~has_r2] = t[has_t & ~has_r2]
    image[has_r2 & ~has_t] = r2[has_r2 & ~has_t]
    mp, mq = manual_p.to_numpy(float), manual_q.to_numpy(float)
    report = b.copy()
    public_counts = np.full((len(b), 1), 6.0) if public_source_counts is None else np.asarray(public_source_counts, float)[:, None]
    # Default: six-source aggregate weight 6, new independent annotation weight
    # confidence. Missing/unmentioned annotation contributes exactly zero votes.
    if manual_share == 1:
        gate = (mq > 0).astype(float)
    else:
        manual_votes = 6 * manual_share / (1-manual_share) * mq
        gate = np.divide(manual_votes, public_counts + manual_votes,
                         out=np.zeros_like(mq), where=(public_counts + manual_votes) > 0)
    manual_present = np.isfinite(mp) & (mq > 0)
    both_report = np.isfinite(b) & manual_present
    report[both_report] = (1-gate[both_report])*b[both_report] + gate[both_report]*mp[both_report]
    report[~np.isfinite(b) & manual_present] = mp[~np.isfinite(b) & manual_present]
    alpha = np.array([REPORT_SHARE[c] for c in LABELS])[None]
    has_report, has_image = np.isfinite(report), np.isfinite(image)
    target = np.where(has_report & has_image, alpha*report+(1-alpha)*image,
                      np.where(has_report, report, image))
    # Reliability weights use provenance/availability and contradictions, rather
    # than turning an unmentioned finding into a negative. No gold fitting.
    report_q = np.where(np.isfinite(b), .70, 0.0)
    report_q = np.maximum(report_q, mq)
    image_q = np.where(has_r2, .75, np.where(has_t, .50, 0.0))
    weight = np.where(has_report & has_image, alpha*report_q+(1-alpha)*image_q,
                      np.where(has_report, report_q, image_q))
    disagreement = np.abs(report - image)
    contradiction = has_report & has_image & (disagreement > .60)
    weight[contradiction] *= .5
    missing = ~np.isfinite(target)
    target[missing], weight[missing] = .5, 0.0
    target = np.clip(target, .001, .999)
    return target, weight, {"teacher_failed": failed_teacher, "r2_family_count": r2_count,
                            "contradiction": contradiction, "report": report, "image": image,
                            "no_target": missing}


def diagnostics(gold: pd.DataFrame, candidates: dict[str, np.ndarray]) -> dict:
    from sklearn.metrics import roc_auc_score
    out = {}
    for name, p in candidates.items():
        by_label = {}
        for j, c in enumerate(LABELS):
            y = gold[c].to_numpy(float)
            valid = np.isfinite(y) & np.isfinite(p[:, j])
            yy, pp = y[valid], p[valid, j]
            auc = float(roc_auc_score(yy, pp)) if len(np.unique(yy)) == 2 else None
            by_label[c] = {"studies": int(valid.sum()), "positive": int((yy == 1).sum()),
                           "negative": int((yy == 0).sum()), "auc": auc}
        aucs = [v["auc"] for v in by_label.values() if v["auc"] is not None]
        out[name] = {"macro_auc": float(np.mean(aucs)) if aucs else None,
                     "valid_findings": len(aucs), "per_finding": by_label}
    return out


def build(args) -> dict:
    if not 0 <= args.manual_share <= 1 or not 0 <= args.teacher_share <= 1:
        raise ValueError("Manual/teacher shares must be in [0,1]")
    tr = load_train(args.train)
    base = read_table(args.base_labels, universe=tr.index).reindex(tr.index)
    if "is_gold" not in base or "fold" not in base or base[["is_gold", "fold"]].isna().any().any():
        raise ValueError("A1 table must cover all UIDs and retain is_gold/fold metadata")
    is_gold = tr[LABELS].notna().all(axis=1)
    if not np.array_equal(base.is_gold.to_numpy(bool), is_gold.to_numpy()):
        raise ValueError("A1 gold flags disagree with the recovered competition labels")
    if not (base.loc[is_gold, "fold"] == -1).all():
        raise ValueError("Gold studies must remain fold=-1 and be excluded from L2 student training")
    if not set(base.loc[~is_gold, "fold"]) <= {0, 1, 2, 3, 4}:
        raise ValueError("Report-only rows need frozen five-fold assignment")
    teacher = read_table(args.teacher, universe=tr.index).reindex(tr.index)
    inputs = [input_record(args.train, "competition_training_table"),
              input_record(args.base_labels, "legacy_public_consensus_and_folds",
                           unknown_vs_negative="not recoverable from the pooled binary table",
                           legacy_missing_imputation="A1 source mean-filled before saving"),
              input_record(args.teacher, "image_teacher", provenance="in_sample_on_report_only_pool",
                           gold_training_exclusion="claimed in source comments, not independently established")]
    families = []
    for directory in args.r2_dir:
        family, records = load_round2(directory, tr, base.fold)
        families.append(family)
        inputs.extend(records)
    manual_p, manual_q, states, ann_inputs, seen, rows = load_annotations(args.annotations, tr, args.batch_dir)
    inputs.extend(ann_inputs)
    weak_ids = set(tr.index[~is_gold])
    if not args.allow_partial and seen != weak_ids:
        raise ValueError(f"Report annotations incomplete: {len(seen)}/{len(weak_ids)}")
    public_counts = None
    if args.public_consensus:
        public = read_table(args.public_consensus, universe=tr.index).reindex(tr.index)
        if "n_sources" not in public:
            raise ValueError("Public consensus must expose n_sources for weighted seventh-vote combination")
        public_counts = pd.to_numeric(public.n_sources, errors="raise").to_numpy(float)
        if not np.isfinite(public_counts).all() or not ((public_counts >= 1) & (public_counts <= 6)).all():
            raise ValueError("Unexpected six-source consensus source-count values")
        inputs.append(input_record(args.public_consensus, "public_consensus_source_counts", maximum_sources=6))
    target, weight, audit = combine(base, teacher, families, manual_p, manual_q,
                                    manual_share=args.manual_share, teacher_share=args.teacher_share,
                                    public_source_counts=public_counts)
    labels = pd.DataFrame(target, index=tr.index, columns=LABELS)
    labels.insert(0, "fold", base.fold.astype(int))
    labels.insert(0, "is_gold", is_gold.astype(int))
    for c in LABELS:
        labels[f"gold_{c}"] = tr[c]
    weights = pd.DataFrame(weight, index=tr.index, columns=LABELS)
    # Gold never contributes to this training generation, even if a trainer omits
    # its is_gold filter. Targets stay un-overridden for honest label diagnostics.
    weights.loc[is_gold] = 0.0
    labels.index.name = weights.index.name = UID
    gi = is_gold.to_numpy()
    image = audit["image"]
    diag = diagnostics(tr.loc[is_gold, LABELS], {
        "public_consensus": base.loc[is_gold, LABELS].to_numpy(float),
        "image_target": image[gi], "l2_target_before_any_gold_override": target[gi]})
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    labels.to_csv(output / "labels_l2.csv")
    weights.to_csv(output / "weights_l2.csv")
    # This provenance file has no reports or evidence spans; it is still private.
    pd.DataFrame({"annotated": [uid in seen for uid in tr.index], "is_gold": gi,
                  "fold": base.fold.to_numpy(int), "teacher_failed": audit["teacher_failed"],
                  "contradiction_findings": audit["contradiction"].sum(1),
                  "zero_target_findings": audit["no_target"].sum(1)}, index=tr.index).to_csv(output / "row_provenance.csv")
    manifest = {"algorithm": "l2_fixed_round2_report_shares_v1", "studies": len(tr),
                "gold_excluded_from_training": int(is_gold.sum()), "report_only_studies": len(weak_ids),
                "annotated_studies": len(seen), "annotations_complete": seen == weak_ids,
                "is_pilot": seen != weak_ids, "teacher_share_within_image": args.teacher_share,
                "manual_max_share_within_report_at_six_public_sources": args.manual_share,
                "manual_vote_formula": "confidence / (n_public_sources + confidence) at default settings",
                "report_share": REPORT_SHARE,
                "round2_families": len(families), "teacher_failed_studies": int(audit["teacher_failed"].sum()),
                "contradictory_cells": int(audit["contradiction"].sum()),
                "missing_target_cells": int(audit["no_target"].sum()),
                "annotation_state_counts": states.stack().value_counts().to_dict(),
                "inputs": inputs, "gold_development_diagnostics": diag,
                "diagnostic_limitations": [
                    "Teacher is in-sample on the report-only pool; subsequent student OOF targets depend on it.",
                    "Gold studies were repeatedly used for model/label selection; these scores are development diagnostics, not independent CV.",
                    "New report annotations exclude all 58 gold studies; gold target AUC does not evaluate new annotation quality.",
                    "L2 target AUC does not forecast student or leaderboard AUC; 0.962 is a target, not an achieved score.",
                    "Legacy public consensus does not expose missing/uncertain source states or independent family count."],
                "privacy": "Private competition-derived targets; do not commit or publish outputs/reports without checking competition terms.",
                "outputs": {p.name: digest(p) for p in output.glob("*.csv")}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--prepare-batches", action="store_true")
    parser.add_argument("--regenerate-a1", action="store_true")
    parser.add_argument("--public-consensus", type=Path)
    parser.add_argument("--fold-provenance-dir", type=Path, action="append", default=[])
    parser.add_argument("--batch-dir", type=Path)
    parser.add_argument("--guide", type=Path)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--base-labels", type=Path)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--r2-dir", type=Path, action="append", default=[])
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--require-complete-annotations", action="store_true")
    parser.add_argument("--allow-partial", action="store_true", help="Explicit pilot-only build; final builds require all report annotations")
    parser.add_argument("--manual-share", type=float, default=1/7)
    parser.add_argument("--teacher-share", type=float, default=.50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.allow_partial and args.require_complete_annotations:
        parser.error("--allow-partial conflicts with --require-complete-annotations")
    if args.regenerate_a1:
        if not args.public_consensus or not args.base_labels:
            parser.error("--regenerate-a1 requires --public-consensus and --base-labels")
        result = regenerate_a1(args.train, args.public_consensus, args.base_labels, args.fold_provenance_dir)
        print(json.dumps({k: v for k, v in result.items() if k != "fold_provenance"}))
    elif args.prepare_batches:
        if not args.batch_dir or not args.guide:
            parser.error("--prepare-batches requires --batch-dir and --guide")
        result = prepare_batches(args.train, args.batch_dir, args.guide, args.batch_size)
        print(json.dumps({"batches": len(result["batches"]), "studies": result["report_only_studies"],
                          "gold_excluded": result["gold_excluded"]}))
    else:
        if not all([args.base_labels, args.teacher, args.output]):
            parser.error("build requires --base-labels, --teacher and --output")
        result = build(args)
        print(json.dumps({"studies": result["studies"], "annotated": result["annotated_studies"],
                          "annotations_complete": result["annotations_complete"],
                          "gold_diagnostics": {k: v["macro_auc"] for k, v in result["gold_development_diagnostics"].items()}}))


if __name__ == "__main__":
    main()
