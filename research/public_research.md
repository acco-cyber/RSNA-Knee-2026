# RSNA Knee public research — 2026-10-14 UTC

The current account baseline is stronger than the older 0.941–0.954 public recipes. The strongest public notebook with an independently checked displayed score is **0.957**, not a verified 0.960–0.964. The fourth decimal in a notebook title is not scoring evidence.

## Coverage and evidence

Authenticated Kaggle `kernels list`, competition `rsna-knee-abnormality-detection`, `scoreDescending`, 100 results/page: **1,100 unique public notebook records** across pages 1–11. Pages 12–16 returned `Not found`. This is a metadata inventory; it is not an assertion that every notebook's full source was manually reviewed. The promising leading sources were downloaded and compared against the recovered account notebook. Raw inventory and read-only score metadata remain outside this public report.

Official public frontend metadata was read using `kernels.KernelsService/GetKernel`. It reports displayed best scores and linked-submission presence. Save-and-run times on three visible test studies do not establish full hidden-test runtime.

| Public source | Actual displayed best | Current successful run ID | Relevant result |
|---|---:|---:|---|
| [heliosli refreshed three-model recipe](https://www.kaggle.com/code/heliosli/0-9573-latest-public-dataset-score-in-one-click) | 0.957 | 358425369 | Public inputs; strict 224 stage and separate raw failure masks |
| [R7 + Raptor + knee-centred ConvNeXt](https://www.kaggle.com/code/bliverrigpdoge/r7-raptor-knee-centred-convnext) | 0.957 | 358269940 | The extra models are private; a fork skips them and yields the base |
| [RSNA Versia 7](https://www.kaggle.com/code/evgendvorkin/rsna-versia-7) | 0.956 | 358019432 | Raptor + public R7; no additional learned arm |
| [degnonguidi reproduction](https://www.kaggle.com/code/degnonguidi/rsna-knee-abnormality-detection) | 0.956 | 357933579 | Same public Raptor/R7 families |
| [224 anatomical mirror addition](https://www.kaggle.com/code/varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up) | 0.956 | 358309699 | Additional plain/mirror average for the small CoAtNet |
| [224 mirror reproduction](https://www.kaggle.com/code/sushanthtiruvaipati/rsna-knee-r7-raptor384-raptor224-mirror-blend) | 0.956 | 358396996 | Same mirror change |

Search-index evidence also confirms [E0052](https://www.kaggle.com/code/guoxiangwangs/e0052-heliosli-rank-blend-public-adaptation) at 0.954 V2 (3m05s sample run, T4×2), and [Apex](https://www.kaggle.com/code/sujanmajhisuzan/rsna-knee-apex-grandmaster-stack) at 0.950 V1. Neither is a stronger starting point than the recovered 0.956 ensemble. The notebook named [Fast Parent 0.957 v2](https://www.kaggle.com/code/fuadmuhammad/rsna-knee-fast-parent-0-957-v2) actually displays 0.939: titles must be checked.

## Concrete differences from the recovered baseline

The recovered `rsna-knee-v224-r7-reader-students10` already combines Raptor 384, Raptor 224, R7 ConvNeXt-small, the goodpjw reader and 15 student checkpoints. Most public blends reuse these same weights, so counting notebook names overstates model diversity.

The refreshed heliosli source uses the **same** R7 bundle `8c7b666e27dc0e8f`, trainer/worker hashes and learned files. Its frozen dataset version 2 is equivalent to the compact R7 dataset. Exact manifest SHA-256:

`c4bb0f7399ada9a020e0c15cae6252033cf0c4335e999961793a176063074ec6`

Required sources: `heliosli/rsna-knee-r6-frozen-fe055406/2`, `nartaa/rsna-knee-publication-swa-weights-20261007`, and `heliosli/rsna-orthofoundation-fold0-epoch7-20261004`. The refreshed notebook's latest run ID is 358426898; the successful scored source is 358425369. Pin the successful source and verify every hash.

Both old and refreshed R7 use no flip TTA when this manifest is mounted. The explicit new `FLIP_TTA=False` is not a new scientific change. Meaningful differences are:

1. Run the 224 CoAtNet before R7, so its contribution is protected from a later timeout.
2. Dump 224 raw probabilities and rank them on that model's own valid studies.
3. Track 384/224 failures separately. If one fails on a study, use the other's Raptor rank instead of mixing a fallback value into it.

The 224 mirror addition is a separate, cheap candidate using existing weights. It reverses adjacent-slice channels for sagittal windows and flips image width for coronal/axial windows, before normalization, then averages probabilities. It changes the image representation, so test it as a controlled variant. The source's claimed movement within the 0.956 rank block does not establish a 0.960 score.

The private top notebook describes potentially useful future diversity: compartment-centred sagittal/coronal crops plus axial views, trained with ConvNeXt-small and EfficientNetV2-S, blended 20% into its base. Its unpublished weights prevent direct reproduction of the reported uplift.

## Labels and validation

[Single-model discussion 735304](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304) reports competitive small models, improved report labels, more training/regularization and OOF pseudo-labels. These are participant observations, not a complete reproduced recipe. [Ensemble discussion 742926](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/742926) emphasizes different errors and full-ensemble OOF validation; its small gold experiment preferred fixed blending over tuned weights. [Image-to-report discussion 743148](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743148) provides negative VLM round-trip results and a positive slice-coverage observation.

The [0.941 reproducibility repository](https://github.com/NTejas-1/RSNA-Knee-Abnormality-Detection) documents a negative label-repair experiment: broad repair of silent synovitis/effusion targets worsened gold performance. Thus L2 should preserve abstention, evaluate per-finding changes, and use image predictions held out from the study being labeled. A target's agreement with the 58 gold cases is not an independent estimate of a newly trained student's leaderboard AUC. Repeated selection on those cases also makes the gold panel optimistic.

## Source provenance and execution limits

[Host announcement 733965](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965) permits hosted LLM labeling when service/access requirements are satisfied. Earlier participant objections are superseded by that announcement.

[Host response 741819](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741819) makes OAI conditional on free, general access without institutional review/sign-off. The incumbent's nartaa CoAtNet weights explicitly acknowledge OAI; R7 documents fastMRI auxiliary training, and its anatomy path references SKM-TEA pretrained assets. Public availability of a derivative model does not by itself resolve its original data access obligations. Record exact provenance and retain acknowledgements.

[CoAtNet author discussion 746792](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792) gives the 0.949 single-model recipe and negative results: 384 training, native 320 inference crop, anatomical mirror, selected-epoch averaging and masked unavailable external labels. This source also warns that the 58 gold cases were reused for selection.

All leading source metadata specifies T4 GPU, internet disabled. [Execution discussion 740474](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740474) distinguishes the nine-hour inference budget from separate training. [DICOM environment discussion 735854](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735854) identifies offline codec wheels, T4/P100 incompatibility and nested input paths. Validate raw-DICOM inference, row identities, finite probabilities and full hidden-budget estimates before spending a scoring slot.

## Decision

For today's attempts, merge the refreshed strict failure handling into the recovered ensemble, retain exact source/data hashes, and compare a controlled L2 student or 224 mirror variant with that base. The public evidence supports an immediate **0.957 reproducible reference** and several plausible small gains. It does not establish a ready-made 0.960–0.964 solution.
