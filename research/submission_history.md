Audit of RSNA Knee submissions for account `gwey1025`, Team Alicia, queried 2026-10-14T20:03:26.602028+00:00.

The official Kaggle SDK returned **71 unique team submissions**, exactly matching `numTotal=71`: **70 scored** and **one rerun error without a score**. They span 2026-08-09 to 2026-10-14 UTC and 51 unique notebook references. Current best displayed public score is **0.956**, reached by seven submissions. No submission in this history establishes 0.957, 0.960, 0.962, or 0.964.

All returned API statuses are COMPLETE; the failed September22 submission is identified by its absent score and explicit rerun-error field. The clean exports omit raw private submission descriptions, private-score fields, reports, labels, file contents, and credentials.

Official submission limits at audit time: **2 used today, 3 available now**, with no lifetime-cap restriction. The count uses UTC2026-10-14; the next reset is **2026-10-15 00:00UTC / October15 09:00KST**, consistent with [Kaggle Staff guidance](https://www.kaggle.com/discussions/product-feedback/48286). The two UTC-October14 submissions occurred at 17:41 and18:10KST on October14.

GPU quota: **30h total, 29.6380067h used, 21m43.176s remaining, zero reserved**. Weekly refresh is **October17 00:00UTC /09:00KST**. TPU quota:20h available. These are normalized via timedelta.total_seconds(); SDK to_dict incorrectly discards whole days and must not be used to interpret total GPU quota.

Pagination integrity: page_size100 returned71 rows; every submission ref is unique; timestamps descend; first50 match the earlier legacy response; count equals the official lifetime total. Requesting page_number2 repeats the first page in this endpoint. Completeness therefore rests on the large first-page result matching the official total, not on a false page traversal.

The submissions API has no hidden-scoring runtime field. Runtime columns are left blank. Execution durations embedded in downloaded notebooks describe visible sample runs and do not establish hidden-dataset runtime.

Recent candidate outcomes:

| Submitted UTC | Candidate | Score | Evidence |
|---|---|---:|---|
| October14 | Raptor384+224 +R7 +reader |0.956|Public224 checkpoint diversification retained best displayed score.|
| October14 | Same base +15students, weight0.10 |0.956|No measurable gain at the API score precision.|
| October13 | h957 Raptor+R7 +reader +15students, weight0.10 |0.956|The h957 name does not establish a0.957 score.|
| October13 | h957 +15students, weight0.12 |0.956|Same displayed score as h957+reader control.|
| October13 | h957 Raptor+R7 +reader |0.956|Strong current baseline for controlled L2 comparison.|
| October13 | GP-A frontier957 and superblend |0.956|Neither submission establishes0.957.|
| October12 | D954 +15students, weight0.25 |0.954|Large student blend did not improve displayed baseline.|
| October12 | D954 +15students, weight0.40 |0.953|Displayed deterioration relative to0.954.|
| October12 | D954 +15students, per-finding |0.953|Gold58-guided weights did not transfer in this attempt.|
| October11 | V20 +first-round students |0.951|Improved displayed0.950V20 baseline.|

Use a stable R7-based0.956 control for L2. The team history supports conservative student weights, and warns against transferring weights selected on only58gold studies without an independent evaluation. Three-decimal scores cannot resolve small gains or rank differences. A0.962 notebook name is a target, not a measured result.

Clean files: `submission_history.csv` contains every attempt; `submission_history_candidates.csv` groups notebook references; `submission_history_by_utc_day.csv` groups UTC submission days; `submission_history.json` contains quota and completeness checks.

Recent UTC days:

| UTC day | Attempts | Scored | Best displayed score |
|---|---:|---:|---:|
|2026-10-03|1|1|0.943|
|2026-10-06|1|1|0.944|
|2026-10-10|1|1|0.950|
|2026-10-11|5|5|0.954|
|2026-10-12|3|3|0.954|
|2026-10-13|5|5|0.956|
|2026-10-14|2|2|0.956|
