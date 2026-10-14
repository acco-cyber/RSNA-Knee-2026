# L2 labels, validation, and submission selection

Prepared 2026-10-15 KST. This is a methodology audit, not a claim that a model has achieved 0.962.

## What is established and what remains unverified

The supplied history describes 4,349 report-only studies, seven report-label sources, teacher/round-2 image pseudo-labels, and a pilot label-combiner score of 0.940 on the 58 expert-labeled studies. That score is unverified until its source arrays, UID joins, scoring code, and prediction provenance are recovered. The public repository currently has no `build_l2.py`, labeling guide, report-batch outputs, or L2 training artifacts. Continuing a labeling wave cannot be claimed from the supplied progress text alone.

The root agent authenticated the current Kaggle account and reported a scored 0.956 submission. Its description includes Raptor384+224, R7, the goodpjw reader, and 15 students at weight 0.10. This submission evidence is stronger than a model name or a claimed improvement in label agreement.

Recovered notebook `RSNA-Knee-2026/rsna-knee-v224-r7-reader-students10.ipynb` includes five first-round ConvNeXt-T checkpoints, five round-2 ConvNeXt-T checkpoints, and five round-2 EfficientNetV2-S checkpoints. The student stage averages their sigmoid predictions, then blends student and base ranks at `STUDENT_W=0.1`. It discovers checkpoints with broad `student*.pt` globs. If five L2 models are simply added to those mounts, their share becomes only one quarter of the student component, or approximately 2.5% of the final blend. Explicit checkpoint manifests and explicit generation groups are required.

## Match the host's labels before combining sources

Primary rubric: [Knee Abnormality Detection AI Challenge Overview, discussion 733343](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343).

The reference labels reflect expert image review and severity thresholds. Borderline cases are negative. ACL requires a major tear; degeneration alone is negative. MCL requires substantial acute tearing, excluding mild sprains and chronic changes. Meniscal degeneration without a surface-reaching tear is negative. OA needs sufficiently extensive, deep cartilage loss. Effusion and Baker's cyst require more than a small collection. Synovitis needs synovial inflammation/thickening, rather than fluid alone. Contusion needs trauma-related marrow signal without a fracture line; fracture needs an acute cortical break or fracture line. These distinctions are essential when parsing reports.

Interpretation for L2: an explicit mild/below-threshold finding is a negative *for this competition*. It differs from a report that does not mention the finding, or one whose wording does not resolve the threshold. Do not infer missing positive criteria from co-occurring findings. Reports and image-based reference labels may legitimately disagree.

Keep separate report states for every finding:

| State | Meaning | Starting training treatment |
|---|---|---|
| Positive | Report explicitly meets the host criterion | Strong report target, with evidence span |
| Negative | Explicit absence, or clearly below the host threshold | Strong negative report target |
| Uncertain | Wording or severity does not resolve the host criterion | Soft evidence, reduced weight; image evidence can help |
| Unmentioned | No relevant report evidence | Abstain from report supervision |
| Source missing | This source did not process this study/finding | Abstain; this is a coverage condition |

Positive and negative rows also need an `explicitness` field. A public binary table usually cannot tell whether a zero means explicit absence or unmentioned; do not pretend that distinction is recoverable from its numeric values alone.

The CheXpert primary project distinguishes unmentioned, negative, uncertain, and positive report outputs, and studies multiple uncertainty policies, including image-based self-training. It supports preserving these states, but its best policy for a chest finding is not proof of the best policy for knee MRI. [CheXpert project](https://stanfordmlgroup.github.io/competitions/chexpert/).

## A reproducible conservative combiner

Use a manifest for each source: name, exact file/version, source family, label-column mapping, state encoding, confidence-column mapping, accepted UID universe, license/provenance, and whether gold labels were used in source construction. Use only exact UID joins, never row-position joins or fuzzy matching.

1. Load the complete competition study universe as strings. Reject duplicate UIDs and malformed probabilities. Missing UIDs and unknown states remain missing. Never apply `fillna(0)` to label sources.
2. Collapse exact duplicate sources and known derived/related sources into families. First combine within a family, then combine families. Report pairwise agreement, coverage, and disagreement rates. Seven parsers reading the same report are not seven independent annotations. Snorkel's primary paper explicitly models source accuracy and dependence to avoid double counting. [Ratner et al., Snorkel](https://arxiv.org/abs/1711.10160).
3. Calculate report target `R[i,c]` as the reliability/confidence-weighted arithmetic mean of available family probabilities. The denominator contains only present, non-abstaining sources. A family with no usable evidence contributes neither numerator nor denominator. Keep its total reliability capped so source count alone cannot create certainty.
4. Calculate image target `I[i,c]` as the mean of verified eligible prediction families. Prefer out-of-fold predictions for report-only training rows and fully held-out predictions for the gold studies. Do not count different folds as independent models; each row's ordinary OOF prediction comes from its held-out fold model. If multiple independent architectures have eligible predictions, combine architecture/generation families explicitly.
5. For a transparent initial experiment, use one fixed global report/image reliability ratio, rather than optimizing twelve ratios on 58 rows. Preserve the existing round-2 recipe if its exact ratio is recovered; isolate the new labels as the experiment. If no ratio exists, evaluate a small predeclared set such as report shares 0.25, 0.50, 0.75 in label diagnostics, and do not describe the best observed share as unbiased validation.
6. Combine with `T = (a * q_R * R + (1-a) * q_I * I) / (a*q_R + (1-a)*q_I)`. If only one side exists, use that side. If neither exists, emit target 0.5 with loss weight 0. Do not sharpen uncertain targets into hard 0/1.
7. Build loss weights from evidence quality, actual coverage, and disagreement. Strong explicit evidence can receive weight 1. Unmentioned report evidence receives 0. Public labels without state provenance receive a reliability cap. Image-only/uncertain rows receive reduced weight. Strong text-image contradictions remain soft and lower-weight, and enter a review queue. Numerical values are configuration choices to validate; they are not known optimal thresholds.
8. For final training only, expert labels can override their corresponding training targets. Evaluate label quality *before* this override. Keep the 58 gold studies outside training when they are the validation set. A score of 1 after a gold override is a join check, not label-validation evidence.

Confidence must not be estimated only from `abs(2p-1)`: a confidently wrong teacher remains wrong. Combine prediction-family disagreement, usable series coverage, source reliability, and, where available, independent calibration. Any calibration learned on gold labels must be fitted within the inner training split.

Required deliverables: `labels_l2.csv` with the twelve soft targets; a same-order per-finding weight table; source-coverage and contradiction summaries; input SHA-256/version manifest; prediction-provenance manifest; and per-finding gold diagnostics before expert override. Publish neither report text nor restricted study-level tables to a public GitHub repository without checking the competition data terms.

## The 0.940 comparison does not predict a 0.962 submission

Report-label AUC measures ranking agreement of manufactured targets with the 58 expert labels. Image-model AUC measures predictions from unseen images. Kaggle AUC measures hidden test predictions. They are different experiments.

If per-finding mixture weights were chosen on the same 58 rows used to report 0.940, its increase over 0.931 includes selection bias. If teacher/student predictions for those rows came from models trained on those rows or their reference labels, the comparison also includes leakage. AUC calculated on L2 targets after direct gold override is invalid as quality evidence. Check all three possibilities explicitly.

Even a clean 0.009 increase in label-target AUC does not transfer additively to leaderboard AUC. The existing round-2 students already use related pseudo-labels and report consensus, and the current final ensemble gives students only 10% weight. The plausible mechanism is improved supervision and complementary errors, not a guaranteed numeric gain.

Report every finding's positive/negative counts and AUC, with study-level paired bootstrap intervals for the difference between label sets or prediction sets. Resample whole studies so correlations among the twelve findings are preserved. Identify resamples or folds with single-class targets; do not silently score them as 0.5 or 1. A 58-study evaluation cannot reliably justify a distinction as narrow as 0.960 versus 0.964. Model-selection overfitting on small evaluation samples is an established issue. [Cawley and Talbot, JMLR](https://www.jmlr.org/beta/papers/v11/cawley10a.html).

## Prevent leakage throughout teacher/student training

Use a fixed study-level fold file for every generation. If patient links or duplicate-image evidence exist, keep linked studies in the same fold. Use scanner/site grouping as a robustness stress test where a defensible grouping exists; do not call scanner fingerprints institution IDs. The host warns that label prevalence may differ between train/public/private sets. [Official dataset description](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data).

For a strict outer validation fold, the teacher, label-combiner parameters, student training, calibration, and blend selection must exclude that fold's reference labels and dependent label information. A held-out *prediction* file alone does not make the whole pseudo-label pipeline held out. A full-data teacher can pass validation information indirectly into the targets used to train students on other rows. Recover training UID lists and fold provenance before using such predictions to claim clean OOF validation.

For the current competition's small gold set, a practical honest alternative is to keep all 58 outside every image-model training generation and report performance on them using a frozen recipe. That remains a development set if repeatedly used for hyperparameter selection. For weight selection, cross-fit the meta-blender on gold predictions, or use fixed low weights and report the resulting diagnostic honestly. Do not retrain on gold and continue labeling the old diagnostic an untouched holdout.

Pooling OOF predictions is preferable to averaging tiny per-fold AUCs. Keep a primary gold-prediction score and a separate silver-label OOF score. Silver-label AUC mainly measures reproduction of report/pseudo-label conventions and can reward site/reporting-style effects.

## Student training and two predeclared submission candidates

Start with the recovered ConvNeXt-T five-fold cache/preprocessing contract. Save the exact backbone, resolution, number of windows, series-selection rules, image normalization, JPEG round-trip behavior, and fold file in every checkpoint. Training and inference must use the same contract. Use soft binary cross-entropy with explicit per-finding weights. Normalize each finding's loss by its active weight mass, and average over active findings. A missing target has zero weight. Do not use aggressive rare-class positive weights without evidence.

Use stable augmentation and a separately initialized or larger student; avoid merely distilling the same predictions with no source of generalization. The Noisy Student primary work uses teacher predictions with student augmentation/noise and an equal-or-larger student. It demonstrates a mechanism, not an expected RSNA score. [Xie et al., Noisy Student](https://arxiv.org/abs/1911.04252).

Freeze checkpoints and blend weights before scoring. Rank-blend the distinct L2 family into the exact scored 0.956 base using a modest global weight, e.g. 0.10 as an initial candidate, with per-study fallback if L2 preparation fails. Use exact input/output UID validation and verify L2 actually ran on every study. A notebook titled `0.962` must disclose that the number is a target until Kaggle returns a score.

For two submissions, predeclare one conservative candidate and one materially different candidate. If new L2 weights are ready and independent diagnostics support them, a possible pair is L2 at 0.10 and a slightly larger fixed share such as 0.15. If L2 is not trained, do not label old checkpoints as L2: use a ready, independently sourced complementary model for the second candidate and continue the L2 training work separately. Increasing the same correlated student leg without evidence is unlikely to establish a reliable gain. Store scored notebook version IDs, score status, runtime, failed-study counts, checkpoint hashes, and the exact final CSV validation receipt.

Any score forecast remains conditional. Naming a notebook, using more subagents, or submitting twice cannot ensure 0.960–0.964.
