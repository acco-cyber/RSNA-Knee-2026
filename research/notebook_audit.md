# RSNA Knee notebook audit — 2026-10-15 KST

## Actionable result

The measured incumbent is `gwey1025/rsna-knee-v224-r7-reader-students10`, script version `358274276`, public score **0.956**. Its visible preview executed Raptor384, R7, Raptor224, the three-fold goodpjw reader, and 15 existing students successfully on three example studies. That preview does not establish hidden-test runtime or a new score.

Two concrete artifacts were prepared under `RSNA-Knee-2026/notebooks/l2_submission`:

- `rsna-knee-0962-target-crop224-audit.ipynb`: same learned incumbent graph and coefficients, corrected crop224 failure/rank handling. Kaggle canonical ID: `gwey1025/rsna-knee-0-962-target-crop224-audit`.
- `actual_l2_fold0/rsna-knee-l2-0-962-target-fold0.ipynb`: adds only explicit `student_l2_convnext_tiny_f0.pt` from `gwey1025/rsna-knee-l2-smoke-20261015`, a separate L2 family at 0.05 rank weight. Requires exactly one completed fixed smoke epoch, matching label/source/partition hashes, training manifest, completion receipt and checkpoint digest. It cannot substitute older students or claim five trained folds.

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

R7's existing compile switch is disabled only for the three-study public preview through its normal worker interface, leaving hash-verified worker source unchanged. Hidden tests above three studies keep the manifest's compilation and compiled/eager parity check. The parent's controlled preview completed in 317 seconds, and submission reference 57194274 was pending when inspected. Successful preview execution does not establish hidden-test runtime or an improved score.

The L2 gate accepted a valid synthetic smoke artifact and rejected 18 materially wrong fixtures: old mode, K8 training, K4 evaluation, zero completed epochs, gold leakage, reversed targets, wrong label/weight/model/trainer digests, wrong partition/warm-start digests, mismatched completion checkpoint, false five-fold claims, extra receipts, incomplete pilots and absent final source pins. Actual L2 GPU training/inference remains unverified until parent retrieves the completed job artifacts.

## L2 smoke identity and label generation

Smoke training uses **four windows per slot**, a quota compromise relative to the original eight-window recipe. Evaluation retains twelve windows and the unchanged six-slot, area-256, JPEG-q95 preprocessing. The variant embeds the exact recovered training `knee.py`, requires its SHA256 `314c2af387a5cbd1d25f192f19f4ecc6d4ae4b30403e92e9d9b3ed2c325880b1`, and checks evaluation-window parity for every retained-series length from 3 through 32.

The existing-student stage in the L2 variant resolves the original four training kernels and 15 filenames explicitly. Its former broad `student*.pt` glob would otherwise also count an attached L2 checkpoint as an incumbent student and blend it twice.

The fold-0 variant was regenerated against the final private dataset version 1 with all 4,349 report-only studies annotated. Labels SHA256 is `4e28bbc05c08ae4509d0eab264ebb88365a0b2cb0683dcb40b63bce5b9dab315`; confidence weights SHA256 is `7d0d5f9a11c28f1ab58bc1dbb4220ea87c34bc75c0104fdc91977353f0db4839`. It also pins original consensus/folds, exact knee/trainer sources and training/validation UID digests. Regeneration against an incomplete pilot blocks execution before GPU work.

Actual L2 probabilities are computed as a separate sidecar in the existing 15-student loop using the exact same prepared `x`, `mask` and `pos`. The original preprocessing function, 15-model mean and old rank-blend ASTs are unchanged. Only the sole L2 model is held on the second GPU when available; a single GPU remains supported. The append stage then applies the unchanged separate 0.05 rank blend using its own raw table and failure mask. No second DICOM preprocessing pass or new preprocessing worker pool remains.

Updated CPU fixtures passed for unchanged input bytes and old predictions, empty-study and nonfinite sidecars, independent masks, average ties and an independent NumPy calculation of the 100-study/12-target blend. All 15 code/writefile cells parse and 18 malformed provenance artifacts remain rejected using the final dataset pins. These checks cannot establish checkpoint loading or GPU predictions; the real smoke output and Kaggle preview remain required.

This audit agent completed private batches b024–b027, 400 reports, with literal evidence and exact UID/schema checks. Eight ambiguous ligament tears were changed to uncertain during semantic review where the report did not resolve the host's substantial/high-grade extent. Reports and report-derived annotations are kept out of the public repository. Annotation b025 row 84 is a shoulder report and abstains on all knee findings.

## Schema and metric

Official outputs are exactly `StudyInstanceUID, ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture`. There are no PCL/LCL output targets. Test reports are unavailable. The official task uses 12-target macro AUC, approximately 1300 hidden studies, offline notebooks, a nine-hour limit, and `submission.csv`.

Official references: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview and https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data .

## Completed actual L2 preview, version 2

The previously pending GPU training and visible inference checks have now passed. `gwey1025/rsna-knee-l2-0-962-target-fold0`, version 2, completed with frozen fast-preview source SHA256 `991141baf7616eb20d5385642fc089b331af7c4a535dd8cc9b1d00a9cb880764`; its normalized remote source matches. The normal full-graph notebook remains frozen at `ac65226ad73d3dad8ed3974b5fd9648ec0d7995a9f8b9fc52005d105b7b03dd8`. Both contain 16 executable code cells and a genuine executable configuration cell.

The three visible studies satisfied the exact first-preview UID guard. Their unchanged base CSV was reused after all four first-preview CSV/receipt digests passed. New L2 probabilities were computed from the actual MRI tensors on `cuda:1`, with strict state-dict loading and smoke checkpoint SHA256 `aa4c9020522dbc7aa6449247d045a9a477c34519133a664489357ad3f555c0a6`. Completion, source, partition and warm-start provenance checks passed against the completed 4,349-report label generation, dataset version 1. Offline timm 1.0.25 and pydicom 3.0.1 match the successful first-base preview.

All 36 saved L2 probabilities are finite and within [0, 1], with no failed studies and distinct probabilities across the three studies for every finding. The real forward passes took 14.389 seconds; the selected CSV receipt was written after 52.158 seconds. Raw L2 CSV SHA256 is `40d3ef517beadc336494b52f97137bbf9f8fcbe564124699bc147cc36650dab0`. Schema, exact visible row order, source hashes, one-fold identity, dependency parity, independent average-rank blend and final CSV digest were checked from the downloaded artifacts.

Final CSV SHA256 `c044757d4c18a28acf70446065c88111d230b5eca0d28b0bb7390972e2f26c00` equals the first-preview CSV. This is expected: a separate 0.05 rank blend cannot swap three distinct base percentile ranks. Real raw L2 probabilities and the GPU execution receipt establish inference. This preview reuses the already executed base; it does not rerun the complete incumbent family. For different visible UIDs or more than three studies, the cache is disabled and the notebook retains the complete base, reader, original 15-student mean, shared-tensor L2 forward and separate 0.05 blend. Hidden-test runtime and score remain unverified.

The smoke contains one completed fold and one fixed epoch, with four training windows rather than the original eight and twelve evaluation windows. It is not a completed five-fold ensemble. Gold58 was excluded from gradients but its earlier use in selecting label mixtures prevents an independent accuracy claim; diagnostic macro AUC changed from 0.923075 to 0.920092. No mixture coefficient or checkpoint was selected from that change. A title containing 0.962 remains a target, with no new leaderboard score established by this preview.

`l2_preview_validation.json` records the sanitized PASS result and artifact digests. It omits study identifiers, raw predictions, reports, report annotations, label tables and credentials. The audit permits submitting the exact completed version 2; it makes no accuracy-gain claim.
