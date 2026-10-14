# RSNA Knee 2026: recovered history and L2 pipeline

The supplied Kaggle account is `gwey1025`, Team Alicia. The initial authenticated audit found 71 unique submissions, 70 with scores and one rerun error. The best recorded public score at the start of this work was **0.956**. A notebook named **0.962 Target** expresses a goal; it does not establish a score.

This repository preserves the original 18 notebooks, output-free copies of 56 accessible notebook references from the submission history, and 12 earlier source notebooks. One history reference, `aadigupta7686/rsna-knee-d4-blend`, could not be recovered because Kaggle returned HTTP 403. History copies are the latest accessible source, and are not assumed byte-identical to every older submitted version. Their source manifest retains author and Kaggle URL credits.

The public research inventoried 1,100 API-visible notebook entries and read the relevant high-scoring source families. It found a reproducible public recipe with a **0.957 displayed best public score**, but no reviewed code establishing 0.960–0.964. Notebook titles containing more digits were checked against scored metadata.

The new files include:

- `scripts/build_l2_labels.py`: validates report annotation coverage, literal evidence, target order, source joins, and recovered prediction folds; builds confidence-weighted L2 labels from the existing six-source consensus, a seventh semantic report source, and teacher/round-two predictions.
- `notebooks/l2_train/`: full five-fold warm-start ConvNeXt-T training and a clearly identified quota-limited fold-zero pilot. These use the recovered JPEG cache, actual six-slot series assignment, and twelve-window inference.
- `notebooks/l2_submission/`: a controlled crop224 ranking/failure-handling challenger and an actual-L2 candidate that requires a completed new checkpoint and blends it as a separate model family.
- `notebooks/l2_tpu_probe/`: a synthetic compatibility check for the unused TPU quota. A compatibility probe is not L2 training.
- `research/`: submission history, public-source findings, the official severity rubric, model/runtime audit and validation caveats.

All 44 annotation batches are complete: 4,349 reports and 52,188 finding decisions. The confidence-weighted L2 dataset joins all 4,407 studies, preserves the authoritative folds and excludes the 58 expert-labeled studies from gradient training. The private dataset is [`gwey1025/rsna-knee-l2-labels-20261015`](https://www.kaggle.com/datasets/gwey1025/rsna-knee-l2-labels-20261015), version 1. `research/l2_build_summary.json` records source hashes, coverage and aggregate validation.

The actual [fold-0 L2 training run](https://www.kaggle.com/code/gwey1025/rsna-knee-l2-smoke-20261015) completed one fixed epoch with four training windows and twelve evaluation windows. Its new checkpoint, all 870 held-out weak-study predictions and all 58 Gold58 predictions were recovered and checked. The Gold58 diagnostic decreased from **0.923075 to 0.920092**; this does not establish an improvement. Checkpoint inspection found all 30,741,112 stored floating values finite. This is one trained fold, not a completed five-fold ensemble. The full [five-fold TPU job](https://www.kaggle.com/code/gwey1025/rsna-knee-l2-tpu-train-20261015) has been launched and remains queued at the latest check.

The first new competition attempt, [crop224 audit](https://www.kaggle.com/code/gwey1025/rsna-knee-0-962-target-crop224-audit), passed its real GPU preview and was submitted as reference **57194274** on October 14, 2026 UTC. Its leaderboard evaluation is pending. The second candidate, [L2 0.962 Target Fold0](https://www.kaggle.com/code/gwey1025/rsna-knee-l2-0-962-target-fold0), version 2, is running its preview. It preserves the original 15 students and adds the actual new L2 checkpoint as a separate 0.05 rank-weight family. For the exact three visible preview studies only, it reuses the first candidate's validated base predictions by hash and still performs real L2 MRI inference. Other study IDs execute the full base and student pipeline. A preview cache is never used on hidden studies.

Kaggle's remaining GPU allowance is exhausted. The accepted preview is still running; its live-log endpoint returned HTTP 500 and persisted outputs are not yet available. The second competition submission has not been sent. No 0.960–0.964 score has been demonstrated. Failed, unsubmitted source variants are preserved in `notebooks/experiments/` with their failure descriptions.

Reports, study-level label tables, DICOMs, checkpoint binaries and credentials are kept out of this public repository. The new Kaggle label dataset is private. Public source copies have execution outputs and attachments removed.

The 58 official labeled studies are excluded from the new gradient training. They have nevertheless influenced upstream models and target choices, so their AUC is a tuning diagnostic. Weak-study teacher/student predictions are not assumed to constitute fully nested, independent distillation validation. The earlier label AUC of about 0.940 is reproducible before adding any new report annotations and does not prove a 0.962 image-model score.

Competition evaluation uses mean ROC-AUC across the twelve findings. Scoring notebooks have internet disabled and a nine-hour runtime limit. The final deadline is October 22, 2026 at 23:59 UTC, or October 23 at 08:59 KST. Public pretrained weights retain their original authors' terms; the OAI host ruling imposes accessibility conditions rather than granting blanket permission to every rehost.

Publication uses `scripts/publish_github.py` and GitHub's Git Data API. It scans credential patterns, verifies uploaded blob hashes, preserves the prior tree, and updates the branch without forcing or rewriting history.
