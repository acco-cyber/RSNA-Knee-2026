# RSNA Knee 2026: recovered history and L2 pipeline

The supplied Kaggle account is `gwey1025`, Team Alicia. An authenticated audit found 71 unique submissions, 70 with scores and one rerun error. The best recorded public score at the start of this work was **0.956**. A notebook named **0.962 Target** expresses a goal; it does not establish a score.

This repository preserves the original 18 notebooks, output-free copies of 56 accessible notebook references from the submission history, and 12 earlier source notebooks. One history reference, `aadigupta7686/rsna-knee-d4-blend`, could not be recovered because Kaggle returned HTTP 403. History copies are the latest accessible source, and are not assumed byte-identical to every older submitted version. Their source manifest retains author and Kaggle URL credits.

The public research inventoried 1,100 API-visible notebook entries and read the relevant high-scoring source families. It found a reproducible public recipe with a **0.957 displayed best public score**, but no reviewed code establishing 0.960–0.964. Notebook titles containing more digits were checked against scored metadata.

The new files include:

- `scripts/build_l2_labels.py`: validates report annotation coverage, literal evidence, target order, source joins, and recovered prediction folds; builds confidence-weighted L2 labels from the existing six-source consensus, a seventh semantic report source, and teacher/round-two predictions.
- `notebooks/l2_train/`: full five-fold warm-start ConvNeXt-T training and a clearly identified quota-limited fold-zero pilot. These use the recovered JPEG cache, actual six-slot series assignment, and twelve-window inference.
- `notebooks/l2_submission/`: a controlled crop224 ranking/failure-handling challenger and an actual-L2 candidate that requires a completed new checkpoint and blends it as a separate model family.
- `notebooks/l2_tpu_probe/`: a synthetic compatibility check for the unused TPU quota. A compatibility probe is not L2 training.
- `research/`: submission history, public-source findings, the official severity rubric, model/runtime audit and validation caveats.

Reports, study-level label tables, DICOMs, checkpoint binaries and credentials are kept out of this public repository. The new Kaggle label dataset is private. Public source copies have execution outputs and attachments removed.

The 58 official labeled studies are excluded from the new gradient training. They have nevertheless influenced upstream models and target choices, so their AUC is a tuning diagnostic. Weak-study teacher/student predictions are not assumed to constitute fully nested, independent distillation validation. The earlier label AUC of about 0.940 is reproducible before adding any new report annotations and does not prove a 0.962 image-model score.

Competition evaluation uses mean ROC-AUC across the twelve findings. Scoring notebooks have internet disabled and a nine-hour runtime limit. The final deadline is October 22, 2026 at 23:59 UTC, or October 23 at 08:59 KST. Public pretrained weights retain their original authors' terms; the OAI host ruling imposes accessibility conditions rather than granting blanket permission to every rehost.

Publication uses `scripts/publish_github.py` and GitHub's Git Data API. It scans credential patterns, verifies uploaded blob hashes, preserves the prior tree, and updates the branch without forcing or rewriting history.
