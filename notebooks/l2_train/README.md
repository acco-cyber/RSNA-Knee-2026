# L2 ConvNeXt training

The notebooks embed the recovered, known working round-2 `knee.py` source and use
the original A1 image cache: actual six acquisition slots, 32 retained slices,
256 area resize, JPEG quality 95, neighboring-slice channels, 8 training windows
and 12 evaluation windows. They warm-start each matching r2 ConvNeXt fold.

The only supervision change is `labels_l2.csv` plus per-target confidence in
`weights_l2.csv`. Both require unique `StudyInstanceUID` and the 12 target
columns, finite values in [0, 1], and all current competition training IDs. A1's
original report-group folds remain authoritative. Completely annotated official
gold studies never enter gradient training. Confidence zero masks a target;
loss is the confidence-weighted BCE sum divided by the weight sum.

`l2_convnext_5fold_train.ipynb` requests folds 0–4 and four fixed additional
epochs, encoder LR 3e-5 and head LR 1e-4. It has no short time cap and requires
hours of available GPU quota. The recovered r2 run spent about 470 seconds per
epoch on 2×T4; the full warm-start job is estimated around three hours, not
guaranteed.

`smoke/l2_convnext_smoke.ipynb` requests fold 0 only and one fixed epoch with an
600-second trainer budget and 650-second notebook budget. Training uses four
windows per slot (full training uses eight); evaluation retains twelve. It requires 2×T4
because that is the measured environment used to size the budget. The halved
training sampling budget is an unmeasured quota compromise. It stops
normally with the last completed EMA, discards any partial extra epoch and can
omit the optional full weak-val predictions. A smoke run is not a five-fold
ensemble. No checkpoint or leaderboard result exists merely by creating these
files.

Exact output files:

- `student_l2_convnext_tiny_f{fold}.pt`
- `oof_l2_convnext_tiny_f{fold}.csv` (check its `set` column; a capped run can
  contain gold diagnostics only)
- `training_l2_f{fold}.json`
- `completion_l2_f{fold}.json`
- `init_r2_gold_f{fold}.csv`
- `l2_training_manifest.json`

Checkpoints include ordered targets, source hashes, fold ID hashes, preserved
preprocessing, warm-start provenance and the count of completed epochs. Last
fixed completed epoch is used; Gold58 does not select the checkpoint. Gold58
has informed upstream labels and architecture choices and is a tuning
diagnostic. Weak-fold predictions are not independent nested-distillation
validation because upstream teacher predictions may contain fold information.
No 0.962 score is claimed.

Source `knee_r2.py` is the exact r2 kernel output (SHA256
`314c2af387a5cbd1d25f192f19f4ecc6d4ae4b30403e92e9d9b3ed2c325880b1`).
`python package_notebooks.py` regenerates both notebooks and metadata. No
credential appears in these files. The training metadata uses internet enabled;
warm-start model construction itself does not download pretrained weights.
