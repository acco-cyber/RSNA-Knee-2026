# RSNA Knee notebook audit — 2026-10-15 KST

## Actionable result

The measured incumbent is `gwey1025/rsna-knee-v224-r7-reader-students10`, script version `358274276`, public score **0.956**. Its visible preview executed Raptor384, R7, Raptor224, the three-fold goodpjw reader, and 15 existing students successfully on three example studies. That preview does not establish hidden-test runtime or a new score.

Two concrete artifacts were prepared under `RSNA-Knee-2026/notebooks/l2_submission`:

- `rsna-knee-0962-target-crop224-audit.ipynb`: same learned incumbent graph and coefficients, corrected crop224 failure/rank handling. Kaggle canonical ID: `gwey1025/rsna-knee-0-962-target-crop224-audit`.
- `actual_l2_fold0/rsna-knee-l2-0-962-target-fold0.ipynb`: adds only explicit `student_l2_convnext_tiny_f0.pt` from `gwey1025/rsna-knee-l2-smoke-20261015`, a separate L2 family at 0.05 rank weight. Requires at least one completed L2 training epoch and L2 provenance. It cannot substitute older students or claim five trained folds.

Names containing 0.962 describe a target. Neither generated artifact has an established score in this audit.

## Verified incumbent graph

Raptor384's checkpoint is SHA256 `7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef`; Raptor224's checkpoint is `3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6`. The hybrid uses Raptor384/Raptor224 shares 0.65/0.35 inside a 0.50 Raptor / 0.50 R7 junction. The goodpjw reader keeps its established per-finding weights. Existing student mean probabilities are ranked and mixed at 0.10.

R7 bundle `8c7b666e27dc0e8f` manifest SHA256 is `c4bb0f7399ada9a020e0c15cae6252033cf0c4335e999961793a176063074ec6`. Embedded worker and preprocessing source hashes are checked at runtime. The new controlled candidate also pins this entire manifest before its member hashes are used.

The existing 15-student family comprises five round-1 ConvNeXt-T folds, five round-2 ConvNeXt-T folds, and five round-2 EfficientNet-V2-S folds. No recovered repo artifact contains the earlier claimed completed L2 checkpoint set or all 44 report-annotation batches.

The current six-slot student preprocessing matches the recovered A1/r2 training source: select actual plane × fat-suppression slots by longest series with stable ties, retain at most 32 slices, area resize to 256, JPEG quality 95 round trip, 12 evaluation windows, clipped adjacent-slice triplets. The unused `build_study_table` helper contains a missing-FS fallback, but A1 and the actual r2 dataset do not call it. Adding that fallback to student inference would introduce a mismatch.

## Concrete bugs and corrections

The incumbent's 224 stage saves only ranked predictions. Its blend applies the 384 model's failure mask to the 224 model and cannot identify 224-only failures. Because the internal ranking uses `argsort().argsort()`, tied neutral 224 predictions can receive arbitrary ordinal ranks. The controlled candidate saves 224 raw probabilities, ranks each view's successful cohort with average ties, and independently handles 384-only, 224-only, both-Raptor, and R7 failures. Numeric coefficients are unchanged.

The incumbent can skip/fail the 224, public-reader, or student stage and still submit a reduced graph. The controlled candidate requires those selected stages and records their actual execution. Up to the documented three-percent per-study failure policy remains explicit; a whole-stage exception cannot masquerade as the complete candidate.

Older `rsna-knee-c-v20-student5f-w40` preview output contains a multiprocessing `PicklingError` after dynamically importing its crop224 script, followed by a caught `crop_exit_1` and `public_crop_used=false`. The current incumbent already uses a real subprocess and successfully runs the crop224 stage; do not revive that older branch.

## Public 0.957 source caveat

The newly retrieved heliosli notebook has no execution counts/outputs. Its title mentions 0.9573, while the official read-only frontend metadata retrieved by the public-research agent reports bestPublicScore **0.957**, current run `358425369`. Four-decimal 0.9573 has not been independently established here.

Compared with the current incumbent core, its primary Raptor384 script and controller source are identical. The R7 manifest/source/checkpoints are unchanged; the frozen R6 dataset version 2 repack is the same R7 bundle. Changes are crop224 raw output, mandatory stage ordering, per-view failure/rank handling, and an explicit no-flip-TTA switch. It supplies no additional trained model capable of establishing a 0.960 gain by itself.

## Older Library variants

All twelve recovered Library notebooks have empty execution counts. They are generated plans/challengers, not verified scored submissions.

| Notebook | Implemented hypothesis | Evidence status |
|---|---|---|
| `rsna-knee-mast-master` | bounded alternate-acquisition DINO views | unrun; older 0.942-family anchor |
| `mast-v2-global96` | Global96 at 0.25 of the CoAt family | unrun; requires extra assets |
| `mast-v3-series-coherent` | suppress windows crossing acquisition boundaries | unrun; trained head convention changes |
| `mast-v4-gap-coverage` | up to 12 added geometric triplets, 0.25 gap mixture | unrun; more image forwards |
| `mast-v5-mass-preserving` | per-acquisition log priors on v4 attention features | unrun; no extra image forwards beyond v4 |
| `RSNA_Knee_Audited_Ensemble` | preserved graph, diagnostics, optional Global96 | unrun |
| `RSNA_Knee_Fixed_GPU_v2` | CPU replay of four frozen alternatives | unrun; no additional trained model |

These models are substantially older than the measured 0.956 Raptor/R7 family. An additional MAST complexity layer has no validation advantage for today's limited runs.

## Checks actually executed

All 30 recovered notebooks parse after handling IPython cell syntax correctly. Relevant embedded Python source and `%%writefile` bodies were also parsed. Initial naive magic stripping misread multiline `!=` and `%` operators; those false positives were corrected before reporting.

MAST master/v2/v3/v4/v5 built-in CPU checks passed with scratch output paths. They cover acquisition selection, fusion/ranking, immutability, series boundaries, novel geometric gap centers, bounded windows, and per-acquisition sampling mass. They are synthetic helper checks and cannot validate real image decoding, checkpoint compatibility, runtime, or leaderboard improvement.

The controlled crop224 fusion passed an independent 500-study synthetic calculation including all four missing-branch cases. Its primary Raptor384 source is unchanged. The preview compile policy passed cohort-size checks at 0, 1, 3, 4, and 1300 studies. All artifact code/writefile bodies parse; saved outputs are cleared.

R7's existing compile switch is disabled only for the three-study public preview through its normal worker interface, leaving hash-verified worker source unchanged. Hidden tests above three studies keep the manifest's compilation and compiled/eager parity check. This avoids the incumbent's roughly 320-second preview startup; the earlier saved complete preview took about eight minutes including reader/students. Actual optimized runtime still requires the Kaggle run.

## Schema and metric

Official outputs are exactly `StudyInstanceUID, ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture`. There are no PCL/LCL output targets. Test reports are unavailable. The official task uses 12-target macro AUC, approximately 1300 hidden studies, offline notebooks, a nine-hour limit, and `submission.csv`.

Official references: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview and https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data .
