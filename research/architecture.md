# RSNA Knee 2026: L2 architecture and deployment recommendations

Research checked 2026-10-14 UTC / 2026-10-15 KST. This is an architecture review, not a measured score forecast. The requested 0.960–0.964 is a target. No source found here proves that a specific recovered L2 student will reach it.

## Immediate decision

For today's two chances, preserve the best existing image pipeline, preprocessing, weights, and sample ordering. Add L2 as a deliberately bounded complementary student only after its actual weights and clean held-out predictions are recovered. Changing orientation, slice selection, MRI intensity transformation, channels, pooling, or resolution at inference invalidates assumptions of an existing checkpoint unless the change was used in that checkpoint's training.

The L2 target's stated Gold58 macro AUC of 0.940 evaluates a derived label array; it is not the AUC of an image model trained on those labels. Selecting 12 separate report/image mixture coefficients on the same 58 studies also makes that estimate optimistic. The decision requires predictions produced by an image student that did not train on those gold studies, with teacher and label provenance audited.

## Current public signals worth recovering through the Kaggle API

These pages are indexed publicly, but Kaggle's dynamic page bodies were not returned by the web reader. Consequently the implementation and score cannot be verified from the page opens alone.

| Public item | Verified search-index signal | Action |
| --- | --- | --- |
| [antoniorotundo2/rsna-knee-convnext-tiny](https://www.kaggle.com/code/antoniorotundo2/rsna-knee-convnext-tiny) | The description says it retains the MRI pipeline and replaces ResNet with a pretrained ConvNeXt-Tiny multilabel classifier. | Pull notebook source and inputs; inspect actual training, checkpoint names, output schema, and score. |
| [matterhorn3838/rsna-knee-v2-velciraptor-dinosaur-speed](https://www.kaggle.com/code/matterhorn3838/rsna-knee-v2-velciraptor-dinosaur-speed) | Indexed as containing a 2.5D ConvNeXt reader. | Compare preprocessing/channel construction and supplied checkpoints with our recovered code. |
| [Discussion 742926](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/742926) | A user asks for a controlled experiment beyond public AUC 0.940; indexed text describes 20 DINO MRI members, five DINOv3/A5 folds, RadImageNet heads and probability TTA with per-target overrides. | Treat a large correlated DINO ensemble as already exploited; inspect what genuine independent ConvNeXt or series information is missing. These are that author's pipeline claims, not reproduced measurements. |
| [Discussion 735304](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304) | A comment says v10 labels beat v9 with DINOv2 but v9 beat v10 with ConvNeXt. | Test labels with the intended architecture. A better report-label gold metric does not imply better ConvNeXt training. |

## L2 student training: low-risk implementation

The following is a proposed engineering recipe, not a claim from a competition winner.

1. Keep identical study IDs, 12 target order, image cache, series grouping, train/validation folds and architecture to the strongest existing ConvNeXt student. Warm-start from each student's corresponding previous-round fold checkpoint when its fold membership is identical.
2. Require `labels_l2.csv` to have one row for every intended training study, unique IDs, finite probabilities in [0,1], all 12 targets and provenance metadata. Join on study ID with one-to-one validation. Never use dataframe row order as the alignment contract.
3. Train with BCE on soft targets; do not round confidence away. Keep existing class weighting initially. Per-target asymmetric loss/positive weighting is a second experiment because it can distort uncertain positive labels and ranking.
4. If source disagreement and report certainty already exist, use them as a loss mask or bounded reliability weight. Unknown report findings are not reliable negative labels. Consensus across public sources may reflect shared extraction errors, so six identical labels should not be assumed to be six independent votes.
5. Produce held-out predictions for every fold keyed by study ID and save both raw logits and probabilities. Save the fold IDs and exact target order alongside checkpoints.
6. Check a small warm-start run before committing all five folds. Use measured clean-gold/student validation AUC and error correlation with the current base to decide whether full training is useful.
7. For a same-day warm-start candidate, prefer a small number of additional epochs with all five matching folds over a new huge backbone. Select a global L2 blend coefficient coarsely; shrink any per-target differences toward that global coefficient because Gold58 is tiny.

### Leakage that must be checked

- A teacher prediction for study i is clean for i only if its image model and label-training sources excluded i's gold annotation. OOF image predictions exclude the image but do not automatically exclude leaked gold labels in their training targets.
- A new student's fold-f validation can still be indirectly contaminated if training targets for other studies were generated by teachers trained on fold f's gold labels. Reusing cross-fitted teacher labels without a nested provenance design is insufficient for a strict independent student evaluation.
- If all Gold58 are reserved from training of every image teacher/student, Gold58 can still support a sanity check. If it was used to select mixture weights, losses, epochs or target-specific blend coefficients, report it as a tuning set; it cannot also establish an unbiased validation gain.
- Bootstrap at study level for uncertainty and inspect per-target positive/negative counts. Some targets will have too few positives to support precise coefficient search. Do not claim a 0.002 gain is real without considering this uncertainty.

## Image upgrades after an actual baseline is recovered

### Multi-series, 2.5D and attention

[Stanford MRNet](https://stanfordmlgroup.github.io/projects/mrnet/) applies a 2D CNN independently to slices, pools slice features, predicts each of three sequences and combines sequence predictions. It establishes a practical MRI bag-of-slices baseline. The different sequences and diagnostic scope differ from this RSNA competition; its reported AUCs are not a score prediction here.

For a new ConvNeXt-T student, a candidate is three adjacent slices as input channels, shared encoder, and masked study-level aggregation. Keep acquisition sequences distinct: neighboring slices must be from the same series, properly ordered, and compatible geometry. Use a simple mean+max pool baseline before introducing attention. Class-specific gated attention is a plausible way to avoid rare focal lesions disappearing into a global mean; [Ilse et al.](https://proceedings.mlr.press/v80/ilse18a.html) supplies the general permutation-invariant MIL operator, not RSNA evidence.

Suggested progression, requiring retraining and measured comparison:

| Stage | New information | Cost and uncertainty |
| --- | --- | --- |
| Existing 2D student with L2 labels | Better supervision while retaining proven image representation | Lowest integration risk. |
| 2.5D student, adjacent-slice channels | Local cross-slice morphology | Must preserve consistent slice order; natural-image pretrained RGB semantics change. |
| Series-aware pooling and missing-series masks | Distinguish sagittal/coronal/axial and acquisition contrasts | More engineering but reuses encoded features. |
| Class-specific attention across slices/series | Different findings can emphasize different locations | Potential overfit with noisy labels; compare to mean+max. |
| Complementary pretrained encoder | A different feature family for blending | Highest same-day risk if weights and offline dependencies are not already attached. |

Use missing-series masks rather than silently replacing absent acquisitions with all-zero bags that are pooled normally. For an all-missing study, generate a explicitly logged fallback and retain its ID. A study with usable sequences should never be discarded because one sequence failed decoding.

### Series selection

Inspect all recovered descriptions and geometry before assigning acquisition slots. A fixed substring list alone is fragile across multinational sites. Prefer DICOM geometry for plane classification and descriptions/sequence tags for contrast; exclude localizers/scouts, derived screenshots and tiny stacks. Allow at least one fallback series per plane when a preferred PD/T2 fat-suppressed sequence is absent. Preserve a series inventory with skip reasons. Whether more sequences improve score is an experiment, not an assumption.

Sampling uniformly over the physical extent of the joint is a sensible baseline. Stratified sampling with mild per-epoch jitter covers more slices during training. Extra inference sampling/TTA needs identical aggregation semantics and a measured incremental gain. Do not use per-target clinical plane preferences as hard exclusions; secondary planes can still add evidence.

### Geometry and intensity

The [DICOM Image Plane Module](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html) defines patient-space geometry using orientation direction cosines, image position and pixel spacing. Derive a slice normal from the cross product of the row/column vectors and sort slices by their position projected on that normal. The dominant absolute normal axis provides a plane classification. `SliceLocation` uses an implementation-specific reference; filenames or `InstanceNumber` alone do not establish anatomical order. This is an implementation inference from the geometry definition.

Canonical reorientation should use the patient-space affine and preserve the MRI's physical aspect ratio; any rotations/flips must be validated visually on real left/right knees. Oblique acquisitions may require affine resampling rather than an unconditional transpose. Do not transpose/flip already trained inputs based only on plane names.

Use actual pixel decoding, honor rescale/LUT if present, normalize according to the recovered training code, and handle MONOCHROME1 deliberately. The [pydicom compression guide](https://pydicom.github.io/pydicom/stable/guides/user/image_data_handlers.html) documents optional decoding dependencies; its [modality-LUT API](https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.pixels.apply_modality_lut.html) covers rescale/LUT application. An offline notebook must include the precise decoder wheels used in its successful run. Blanket exception-to-black-image handling hides distribution shifts and should emit study/series counters with failure reasons.

Avoid normalizing MRI as CT Hounsfield units or imposing one CT-style window. A bounded volume-level percentile normalization is a candidate for a new model; per-slice normalization may erase relative contrast, and a trained model's existing normalization must be preserved.

## Complementary encoders and checkpoint compatibility

- [Official ConvNeXt](https://github.com/facebookresearch/ConvNeXt) lists Tiny at 28M parameters and 4.5 GFLOPs per 224×224 image. It is a practical encoder for multiple slices, but runtime is dominated by total encoded slices, five folds, TTA and DICOM decoding.
- [Official ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2) supplies masked-autoencoder pretrained checkpoints. Tiny/Base is a reasonable independent experiment before considering Large/Huge. An ImageNet score is not an MRI score.
- [Official DINOv3](https://github.com/facebookresearch/dinov3) provides ViTs and ConvNeXt Tiny/Small/Base/Large, local checkpoint loading, and model-specific preprocessing. The source says timm support began at 1.0.20. Its weights use the DINOv3 license and gated access; require actual permitted checkpoint acquisition and an offline attachable copy before planning a submission. DINOv3 ConvNeXt has a different self-supervised training history even if the architecture overlaps an ImageNet ConvNeXt.
- Existing task-trained ConvNeXt or RadImageNet checkpoints, with verified input preprocessing and out-of-fold predictions, may be more actionable today than a fresh foundation encoder. Do not combine models just because names differ: compare validation errors and prediction ranks.

Each checkpoint must include `model_name`, constructor configuration, targets/order, fold membership, input shape/channel convention, image transform version, slice/series sampling, pooling, training labels hash, source checkpoints, epoch and validation selection method. Restore with strict state-dict loading. `strict=False` can silently ignore a missing classifier or unmatched keys, producing a finite but invalid submission.

## Runtime planning and offline submission contract

The later-recovered round-2 ConvNeXt run provides a real 2×T4 benchmark: about 470 seconds per steady-state training epoch, about 587 seconds for its first epoch, and 29,866 seconds for the full five-fold, twelve-epoch job. The four-epoch L2 warm-start is consequently an hours-long job; a single-fold, one-epoch smoke run uses four training windows rather than eight to fit the remaining quota. Its speed and score are not measured until execution completes.

For N studies, F folds, E epochs and K encoded images per study:

`training_image_count = N * (F - 1) * E * K`

For N=4,407, F=5, E=3 and K=24, the count is 1,269,216 encoded images. For ten epochs it is 4,230,720. Divide by measured training images/second, then add preprocessing, validation, model startup and checkpoint writes. These counts are a workload estimate, not an accelerator throughput claim. At 384 resolution the image area is about 2.94× the area at 224; higher resolution and twice the slices can consume the budget quickly.

Cache compressed/resized study arrays once where allowed, then reuse them across folds. Raw competition input is hundreds of GB; never plan to rebuild the entire cache independently per fold. Cache keys must include all preprocessing and sampling settings so a stale cache cannot silently masquerade as new preprocessing. Keep recoverable checkpoints and metric logs after each epoch.

[PyTorch AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html) documents autocast and gradient scaling. Use a verified dtype for the accelerator, `inference_mode` for scoring, batched CNN encoding, pinned asynchronous transfers and bounded image batches. New `torch.compile` or untested BF16 settings can add compilation failures and startup cost; do not add them to an already reliable scoring kernel without a real benchmark.

Before either submission:

1. Load exact local attached checkpoints and offline packages without any network lookup.
2. Test at least a small real set with all model families, decoder backends, missing sequences and fallback paths.
3. Verify every sample/submission ID is present exactly once, target columns match the official template, values are finite and bounded, and no target became constant accidentally.
4. Log decoded/failed/empty studies, per-target probability spread, active ensemble members and checkpoint hashes.
5. Estimate total scoring time from a representative sample with extra allowance for slow compressed and unusual studies. Bound batch sizes and avoid materializing every study/fold/TTA tensor at once.
6. Keep the best proven base as the default. If L2 is trained and meaningfully complementary, the first candidate can add a conservative L2 weight; use the second chance only after seeing an actual result and diagnosing it. Never submit a filename or title such as `0.962` as if that is a measured outcome.

## Sources and interpretation

All architecture recommendations beyond the sources' explicit descriptions are engineering inferences and proposed experiments. The current competition source is [RSNA's challenge page](https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge), which confirms MRI images and multilingual training reports. The more detailed current competition rules, scoring runtime, final target schema, and exact recovered model weights still need authoritative inspection from the actual Kaggle/repository files.

## Recovered repository review (same-session update)

Local sources now available at `RSNA-Knee-2026/`. The following findings come from actual notebook source and its saved outputs, not assumptions from model names:

### What is already implemented

`rsna-knee-h957-reader-students10.ipynb` and `rsna-knee-v224-r7-reader-students10.ipynb` already have the 2.5D architecture described above. `KneeNet` uses six acquisition slots (plane × fat suppression), a shared encoder over three-slice windows, slot and continuous-position embeddings, a two-layer transformer, and independent attention pooling for the 12 findings. Therefore a generic “add MIL” or “add 2.5D” recommendation does not add a new capability to this reader.

Their latest student stage has three five-fold groups: round-1 ConvNeXt-T, round-2 ConvNeXt-T, and round-2 EfficientNetV2-S. It averages all 15 probabilities and blends the resulting rank at 0.10 into the already combined public/own-reader base. It caps each series at 32 slices, selects 12 windows, and explicitly uses a 256-area-resize followed by JPEG-quality-95 encode/decode to match training. The saved run has 3 visible test studies and 15 series; a 70-second stage time only describes that visible sample.

`rsna-knee-stack-2-5d-convnext-mil-lb-0-944.ipynb` documents the reader's earlier public results: a reader alone at 0.929, community stack at 0.943, and minority blend at 0.944. Its accompanying narrative reports that mixed labels improved a single reader from 0.922 to 0.927 but reduced a five-fold ensemble from 0.930 to 0.929 while improving its pseudo-label CV score. Those are recorded author claims that the account's submission history should verify. They directly support caution about assuming L2's better target AUC yields a better ensemble.

### Highest-priority checks before altering student inference

1. Actual A1 cache and r2 training sources are recovered. They manually select the most-file series for each real acquisition slot and leave absent slots masked. The student-stage `_prep` matches that behavior. An unused `build_study_table` contains fat-sat/non-fat-sat fallback duplication, but it does not establish a mismatch in the actual training path. Preserve the verified six-slot cache recipe, JPEG quality 95, resolution 256, and twelve evaluation windows.
2. Checkpoint discovery is a broad glob of `student*.pt` across all attached inputs. Attaching five L2 checkpoints would silently change the existing student's 15-model equal average to a 20-model average. L2 would receive only one-quarter of the 0.10 student group, while all existing students are also reweighted. This is not an isolated “10% L2 blend” experiment. Use an explicit checkpoint manifest and separate `r1_convnext`, `r2_convnext`, `r2_effv2`, `l2_convnext` prediction arrays and weights.
3. Student model constructors recover only the backbone name and assume default head dimensions/depth. Require exact architecture metadata and strict state-dict loading before final scoring. Saved errors currently trigger an outer fallback to the prior submission; that must be visible in the notebook name/report rather than described as a successful L2 result.
4. Record the original base before every rank stage. Re-ranking several nested blends is non-associative; changing where L2 enters can change rankings even if nominal final weights look equal. An isolated L2 candidate should compare against precisely the same base and current 15-student group, with only its explicit L2 coefficient changed.
5. The recovered mainline explicitly states that the Raptor was trained on all training studies and selected on Gold58. Consequently a Gold58 fit of the complete stack's coefficients cannot be called an independent holdout validation. Use it as a labeled diagnostic, limit coefficient flexibility, and verify real scores.

### Best next architecture experiment

After reconstructing labels and checkpoint provenance, warm-start the existing ConvNeXt-T folds on L2 while keeping the exact JPEG cache, six-slot behavior and class head. Log pure L2 predictions separately. If image pseudo-labels still contain mostly the same architecture's information, a confidence-weighted report target or a distinctly trained teacher may add more independence than repeating identical distillation. A larger encoder can wait until the current L2 student has a measured result.
