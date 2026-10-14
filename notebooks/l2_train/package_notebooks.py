"""Build self-contained Kaggle L2 notebooks from the recovered r2 source."""

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
KNEE_PATH = HERE / "knee_r2.py"
KNEE = KNEE_PATH.read_text()
TRAINER = (HERE / "train_student_l2.py").read_text()


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def build(smoke):
    slug = "rsna-knee-l2-smoke-20261015" if smoke else "rsna-knee-l2-train-20261015"
    folder = HERE / "smoke" if smoke else HERE
    folder.mkdir(exist_ok=True)
    filename = "l2_convnext_smoke.ipynb" if smoke else "l2_convnext_5fold_train.ipynb"
    config = f'''import time
NB_T0 = time.monotonic()
FOLDS = {[0] if smoke else [0, 1, 2, 3, 4]!r}
EPOCHS = {1 if smoke else 4}
K_TRAIN = {4 if smoke else 8}
ENCODER_LR = 3e-5
HEAD_LR = 1e-4
FOLD_BUDGET_SECONDS = {600 if smoke else 0}
NOTEBOOK_BUDGET_SECONDS = {650 if smoke else 0}
SMOKE_ONLY = {smoke!r}
L2_DATASET = "gwey1025/rsna-knee-l2-labels-20261015"
L2_SLUG = L2_DATASET.split("/")[-1]
print("L2 training configuration", FOLDS, EPOCHS, "smoke", SMOKE_ONLY)
'''
    setup = '''import glob, os, sys, subprocess, json, hashlib
from pathlib import Path
import torch

def mount(slug, required):
    candidates = [Path("/kaggle/input") / slug]
    for pattern in (f"/kaggle/input/notebooks/*/{slug}", f"/kaggle/input/datasets/*/{slug}", f"/kaggle/input/*/{slug}"):
        candidates += [Path(path) for path in sorted(glob.glob(pattern))]
    for candidate in candidates:
        if (candidate / required).is_file():
            return candidate
    raise FileNotFoundError(f"{slug}/{required} is not attached")

A1 = mount("rsna-knee-a1-train-cache-labels", "labels_cons.csv")
R2 = mount("rsna-knee-b-student-f01234-r2-convnext", "student_r2_convnext_tiny_f0.pt")
L2 = mount(L2_SLUG, "labels_l2.csv")
if not (L2 / "weights_l2.csv").is_file():
    raise FileNotFoundError("L2 dataset lacks weights_l2.csv")
COMP = next((Path(path) for path in ("/kaggle/input/competitions/rsna-knee-abnormality-detection", "/kaggle/input/rsna-knee-abnormality-detection")
             if (Path(path) / "train.csv").is_file()), None)
if COMP is None:
    raise FileNotFoundError("Competition train.csv is not attached")
SRC = Path("/kaggle/working/l2_src")
SRC.mkdir(exist_ok=True)
(SRC / "knee.py").write_text(KNEE_SOURCE)
(SRC / "train_student_l2.py").write_text(TRAINER_SOURCE)
sys.path.insert(0, str(SRC))
from train_student_l2 import load_training_tables
from knee import LABELS
EXPECTED_LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
                   "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
assert LABELS == EXPECTED_LABELS, "Recovered target order changed"
for fold in FOLDS:
    checkpoint = R2 / f"student_r2_convnext_tiny_f{fold}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    lab, targets, confidence, tr, va, gold, gold_y = load_training_tables(A1, L2, COMP, LABELS, fold)
    print("PREFLIGHT", fold, "training", len(tr), "val", len(va), "gold", len(gold),
          "confidence avg", round(float(confidence.to_numpy().mean()), 4), flush=True)
del lab, targets, confidence, tr, va, gold, gold_y
print("Recovered knee.py SHA256", hashlib.sha256(KNEE_SOURCE.encode()).hexdigest(), flush=True)
NPROC = min(2, torch.cuda.device_count())
if NPROC == 0:
    raise RuntimeError("L2 image training requires a GPU")
if SMOKE_ONLY and NPROC < 2:
    raise RuntimeError("Smoke budget was sized from the recovered 2xT4 run; select 2xT4")
'''
    run = '''receipts = []
for fold in FOLDS:
    elapsed = time.monotonic() - NB_T0
    budget = FOLD_BUDGET_SECONDS
    if NOTEBOOK_BUDGET_SECONDS:
        remaining = NOTEBOOK_BUDGET_SECONDS - elapsed - 30
        if remaining < 430:
            raise RuntimeError("Too little quota budget remains for a completed L2 epoch")
        budget = min(budget, remaining) if budget else remaining
    command = ["torchrun", "--standalone", "--nproc_per_node", str(NPROC), str(SRC / "train_student_l2.py"),
               "--a1", str(A1), "--l2", str(L2), "--competition", str(COMP), "--src", str(SRC),
               "--init", str(R2 / f"student_r2_convnext_tiny_f{fold}.pt"), "--fold", str(fold),
               "--epochs", str(EPOCHS), "--k-train", str(K_TRAIN), "--lr", str(ENCODER_LR), "--lr-head", str(HEAD_LR),
               "--budget-seconds", str(budget)]
    run_start = time.monotonic()
    result = subprocess.run(command, check=False, timeout=budget + 25 if budget else None)
    if result.returncode:
        raise RuntimeError(f"L2 fold {fold} training failed with status {result.returncode}")
    completion_file = Path("/kaggle/working") / f"completion_l2_f{fold}.json"
    if not completion_file.is_file():
        raise RuntimeError("Training returned without a completion receipt")
    receipt = json.loads(completion_file.read_text())
    if receipt["completed_epochs"] < 1:
        raise RuntimeError("No L2 epoch completed")
    receipt["kernel_wall_seconds"] = time.monotonic() - run_start
    receipts.append(receipt)
    print("L2_FOLD_COMPLETE", json.dumps(receipt), flush=True)
manifest = {
    "model_family": "l2_convnext_tiny", "requested_folds": FOLDS, "requested_epochs": EPOCHS,
    "k_train": K_TRAIN, "k_eval": 12,
    "smoke_only": SMOKE_ONLY, "five_folds_complete": len(receipts) == 5,
    "dataset": L2_DATASET, "targets": LABELS, "receipts": receipts,
    "gold_auc_is_independent_validation": False,
    "preprocessing": "Recovered A1/r2 exact: actual slots; 32 retained slices; area256; JPEGq95; eval12 windows",
}
Path("/kaggle/working/l2_training_manifest.json").write_text(json.dumps(manifest, indent=2))
print("L2_TRAINING_COMPLETE", json.dumps(manifest), flush=True)
'''
    description = (
        "# RSNA Knee L2: fold-0 quota-limited training\n\n"
        "One fixed epoch warm-starts the matching round-2 fold-0 checkpoint. This is a single-fold smoke candidate; "
        "it is not a five-fold ensemble and no leaderboard score is claimed. The trainer caps itself at 600 seconds "
        "and keeps only completed-epoch weights. Training uses four windows per slot to reduce the workload; evaluation remains "
        "at twelve windows. Optional full weak-validation prediction is omitted if time is short.\n\n"
        if smoke else
        "# RSNA Knee L2: five-fold ConvNeXt training\n\n"
        "Four fixed epochs warm-start each matching round-2 ConvNeXt-T fold. Uses the recovered A1 JPEG cache and "
        "KneeNet source unchanged, with new L2 soft labels and per-target confidence weights. No leaderboard score is claimed.\n\n"
    )
    description += (
        "The 58 official labeled studies never enter gradient training. They have informed upstream target choices, so their "
        "AUC is a tuning diagnostic. Weak-study held-out predictions are also not clean nested-distillation validation, because "
        "the teacher labels can contain information from those folds. Final weights are from the last fixed completed epoch; "
        "Gold58 does not select the checkpoint.\n\n"
        "Required inputs: the private L2 labels dataset, the A1 cached training kernel, the round-2 ConvNeXt kernel, "
        "and the competition. Full five-fold training takes hours; quota must be checked before launching.\n"
    )
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3.11"}},
                "cells": [markdown(description), code(config), code("KNEE_SOURCE = " + repr(KNEE) + "\nTRAINER_SOURCE = " + repr(TRAINER) + "\n"), code(setup), code(run)]}
    (folder / filename).write_text(json.dumps(notebook, indent=1))
    metadata = {
        "id": "gwey1025/" + slug,
        "title": "RSNA Knee L2 Smoke 20261015" if smoke else "RSNA Knee L2 Train 20261015",
        "code_file": filename, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": ["gwey1025/rsna-knee-l2-labels-20261015"],
        "kernel_sources": ["gwey1025/rsna-knee-a1-train-cache-labels", "gwey1025/rsna-knee-b-student-f01234-r2-convnext"],
        "competition_sources": ["rsna-knee-abnormality-detection"], "model_sources": [],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f",
        "machine_shape": "NvidiaTeslaT4",
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{filename}:cell{index}", "exec")
    return str(folder / filename)


if __name__ == "__main__":
    print(build(False))
    print(build(True))
