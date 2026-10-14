"""Warm-start recovered round-2 ConvNeXt folds using explicit L2 targets.

The 256px JPEG-q95 study cache and KneeNet implementation are the recovered r2
ones. Gold58 is excluded from training, but has already informed upstream labels
and model choices; its AUC is a diagnostic, not an independent validation score.
"""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.distributed as dist
import torch.nn.functional as F


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_indexed(path):
    frame = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    if "StudyInstanceUID" not in frame:
        raise ValueError(f"{path}: missing StudyInstanceUID")
    if frame.StudyInstanceUID.isna().any() or frame.StudyInstanceUID.duplicated().any():
        raise ValueError(f"{path}: missing or duplicate study IDs")
    return frame.set_index("StudyInstanceUID")


def checked_targets(frame, labels, ids, name):
    if not set(labels).issubset(frame.columns):
        raise ValueError(f"{name}: expected all 12 target columns")
    if set(frame.index) != set(ids):
        missing = len(set(ids) - set(frame.index))
        extra = len(set(frame.index) - set(ids))
        raise ValueError(f"{name}: study ID mismatch; missing {missing}, extra {extra}")
    value = frame.loc[ids, labels].apply(pd.to_numeric, errors="raise").astype(np.float32)
    if not np.isfinite(value.to_numpy()).all():
        raise ValueError(f"{name}: non-finite values")
    if not ((value.to_numpy() >= 0) & (value.to_numpy() <= 1)).all():
        raise ValueError(f"{name}: values outside [0, 1]")
    return value


def load_training_tables(a1, l2, competition, labels, fold):
    lab = read_indexed(Path(a1) / "labels_cons.csv")
    official = read_indexed(Path(competition) / "train.csv")
    ids = official.index.tolist()
    if set(lab.index) != set(ids):
        raise ValueError("A1 labels_cons and official train study IDs differ")
    lab = lab.loc[ids]
    official_gold = official[labels].notna().all(axis=1)
    if not np.array_equal(official_gold.to_numpy(), lab.is_gold.to_numpy().astype(bool)):
        raise ValueError("A1 gold flags differ from the current official labels")
    folds = pd.to_numeric(lab.fold, errors="raise").astype(int)
    if set(folds[~official_gold].unique()) != set(range(5)):
        raise ValueError("A1 must have five existing weak-study folds")
    if not (folds[official_gold] == -1).all():
        raise ValueError("Gold studies must have A1 fold=-1")
    target = checked_targets(read_indexed(Path(l2) / "labels_l2.csv"), labels, ids, "L2 labels")
    confidence = checked_targets(read_indexed(Path(l2) / "weights_l2.csv"), labels, ids, "L2 confidence")
    cache_meta = read_indexed(Path(a1) / "cache256_meta.csv")
    cached = set(cache_meta.index[cache_meta.n_slots > 0])
    weak = lab[~official_gold & lab.index.isin(cached)]
    train = weak[(weak.fold != fold) & (confidence.loc[weak.index].sum(axis=1) > 0)]
    val = weak[weak.fold == fold]
    gold = lab[official_gold & lab.index.isin(cached)]
    if not len(train) or not len(val) or not len(gold):
        raise ValueError("Empty train/val/gold partition")
    if set(train.index) & (set(val.index) | set(gold.index)):
        raise ValueError("Train partition overlaps validation or gold")
    gold_y = official.loc[gold.index, labels].to_numpy(dtype=np.int64)
    return lab, target, confidence, train, val, gold, gold_y


def macro(y, probabilities, labels):
    result = {}
    for i, name in enumerate(labels):
        if np.unique(y[:, i]).size == 2:
            result[name] = float(roc_auc_score(y[:, i], probabilities[:, i]))
    if not result:
        raise ValueError("No evaluable target has both classes")
    return float(np.mean(list(result.values()))), result


def main():
    process_started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1", required=True)
    parser.add_argument("--l2", required=True)
    parser.add_argument("--competition", required=True)
    parser.add_argument("--src", required=True)
    parser.add_argument("--init", required=True)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--k-train", type=int, default=8, choices=range(1, 17))
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--budget-seconds", type=float, default=0,
                        help="0 means unlimited. Stop normally with the last completed epoch when budget is tight.")
    parser.add_argument("--final-eval-reserve", type=float, default=160)
    parser.add_argument("--output", default="/kaggle/working")
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    deadline = process_started + args.budget_seconds if args.budget_seconds > 0 else float("inf")
    sys.path.insert(0, args.src)
    from knee import LABELS, N_SLOTS, KneeNet, window_centres

    res, k_train, k_eval, batch_size = 256, args.k_train, 12, 2
    backbone, weight_decay, ema_decay = "convnext_tiny", 0.02, 0.998
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if ddp:
        dist.init_process_group("nccl")
    rank = dist.get_rank() if ddp else 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    torch.manual_seed(args.fold * 100 + rank)
    np.random.seed(args.fold * 100 + rank)
    cv2.setNumThreads(1)
    say = (lambda message: print(message, flush=True)) if rank == 0 else (lambda message: None)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    lab, target, confidence, train, val, gold, gold_y = load_training_tables(
        args.a1, args.l2, args.competition, LABELS, args.fold
    )
    cache = Path(args.a1) / "cache256"
    say(f"L2 fold {args.fold}: train {len(train)} val {len(val)} gold {len(gold)}; confidence weighted BCE")
    say("Gold AUC is training-excluded but tuning-exposed; weak val is not clean nested distillation OOF.")

    class StudyDataset(torch.utils.data.Dataset):
        # Same JPEG cache decoding, slots and window sampling as recovered r2.
        def __init__(self, ids, y, weights, k, training):
            self.ids, self.y, self.weights = list(ids), y, weights
            self.k, self.training = k, training

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, index):
            study = self.ids[index]
            x = np.zeros((N_SLOTS, self.k, 3, res, res), np.uint8)
            mask = np.zeros(N_SLOTS, bool)
            pos = np.zeros((N_SLOTS, self.k), np.float32)
            with np.load(cache / f"{study}.npz", allow_pickle=False) as archive:
                for slot in range(N_SLOTS):
                    if f"s{slot}_o" not in archive.files:
                        continue
                    data, offsets = archive[f"s{slot}_d"], archive[f"s{slot}_o"]
                    n = len(offsets) - 1
                    if n < 3:
                        continue
                    centers = window_centres(n, self.k, self.training)
                    decoded = {}
                    required = sorted({int(t) for center in centers for t in np.clip([center - 1, center, center + 1], 0, n - 1)})
                    for j in required:
                        image = cv2.imdecode(data[offsets[j]:offsets[j + 1]], cv2.IMREAD_GRAYSCALE)
                        if image is None or image.shape != (res, res):
                            raise ValueError(f"{study} slot {slot} slice {j}: invalid JPEG cache")
                        decoded[j] = image
                    x[slot] = np.stack([
                        np.stack([decoded[int(t)] for t in np.clip([center - 1, center, center + 1], 0, n - 1)])
                        for center in centers
                    ])
                    mask[slot], pos[slot] = True, centers / max(n - 1, 1)
            if not mask.any():
                raise ValueError(f"{study}: cache metadata claimed nonempty study but no slot is usable")
            y = self.y[index] if self.y is not None else np.zeros(len(LABELS), np.float32)
            w = self.weights[index] if self.weights is not None else np.ones(len(LABELS), np.float32)
            return tuple(torch.from_numpy(value) for value in (x, mask, pos, np.asarray(y, np.float32), np.asarray(w, np.float32)))

    training = StudyDataset(train.index, target.loc[train.index, LABELS].to_numpy(),
                            confidence.loc[train.index, LABELS].to_numpy(), k_train, True)
    sampler = torch.utils.data.distributed.DistributedSampler(training, shuffle=True, drop_last=True) if ddp else None
    train_loader = torch.utils.data.DataLoader(
        training, batch_size=batch_size, shuffle=sampler is None, sampler=sampler, drop_last=True,
        num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=3,
    )
    initial = torch.load(args.init, map_location="cpu", weights_only=False)
    saved_args = initial.get("args", {})
    if saved_args.get("backbone") != backbone or int(saved_args.get("fold", -1)) != args.fold:
        raise ValueError("Warm-start checkpoint is not the matching r2 ConvNeXt fold")
    if saved_args.get("mode") != "r2" or int(saved_args.get("res", -1)) != res:
        raise ValueError("Warm-start checkpoint mode/resolution differ from recovered r2")
    model = KneeNet(backbone, pretrained=False).cuda().to(memory_format=torch.channels_last)
    model.load_state_dict(initial["model"], strict=True)
    del initial
    ema = copy.deepcopy(model).eval()
    for param in ema.parameters():
        param.requires_grad_(False)
    encoder = list(model.enc.parameters())
    head = [param for name, param in model.named_parameters() if not name.startswith("enc.")]
    optimizer = torch.optim.AdamW([{"params": encoder, "lr": args.lr}, {"params": head, "lr": args.lr_head}], weight_decay=weight_decay)
    network = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank]) if ddp else model
    steps, warm = args.epochs * len(train_loader), max(1, len(train_loader) // 4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / max(1, steps))))
    )
    scaler = torch.cuda.amp.GradScaler()

    @torch.no_grad()
    def predict(net, ids):
        loader = torch.utils.data.DataLoader(StudyDataset(ids, None, None, k_eval, False),
                                            batch_size=4, num_workers=2, pin_memory=True)
        predictions = []
        net.eval()
        for x, mask, pos, _, _ in loader:
            with torch.autocast("cuda", dtype=torch.float16):
                probability = net(x.cuda(non_blocking=True), mask.cuda(), pos.cuda(), res=res).float().sigmoid()
            if not torch.isfinite(probability).all():
                raise ValueError("Non-finite L2 validation predictions")
            predictions.append(probability.cpu())
        return torch.cat(predictions).numpy()

    hashes = {
        "labels_l2.csv": digest(Path(args.l2) / "labels_l2.csv"),
        "weights_l2.csv": digest(Path(args.l2) / "weights_l2.csv"),
        "labels_cons.csv": digest(Path(args.a1) / "labels_cons.csv"),
        "knee.py": digest(Path(args.src) / "knee.py"),
        "warm_start": digest(args.init),
        "trainer": digest(__file__),
    }
    init_gold = None
    history, started, step = [], time.time(), 0
    completed_epochs, prior_training_seconds = 0, None
    checkpoint_path = out_dir / f"student_l2_convnext_tiny_f{args.fold}.pt"
    if rank == 0:
        init_gold = predict(ema, gold.index)
        value, per_target = macro(gold_y, init_gold, LABELS)
        say(f"INIT r2 fold {args.fold}: gold diagnostic {value:.5f}")
        pd.DataFrame(init_gold, index=pd.Index(gold.index, name="StudyInstanceUID"), columns=LABELS).to_csv(out_dir / f"init_r2_gold_f{args.fold}.csv")
    if ddp:
        dist.barrier()
    for epoch in range(args.epochs):
        cannot_complete = int(bool(completed_epochs) and time.monotonic() + prior_training_seconds + 35 > deadline)
        stop_epoch = torch.tensor([cannot_complete], device="cuda")
        if ddp:
            dist.all_reduce(stop_epoch, op=dist.ReduceOp.MAX)
        if bool(stop_epoch.item()):
            say("L2_TIME_GUARD: insufficient time for another completed epoch; retaining last completed EMA")
            break
        if sampler is not None:
            sampler.set_epoch(epoch)
        network.train()
        total, count, epoch_start = 0.0, 0, time.time()
        partial_epoch = False
        for x, mask, pos, y, weights in train_loader:
            stop = torch.tensor([int(time.monotonic() + 35 > deadline)], device="cuda")
            if ddp:
                dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if bool(stop.item()):
                partial_epoch = True
                break
            x, mask, pos = x.cuda(non_blocking=True), mask.cuda(), pos.cuda()
            y, weights = y.cuda(), weights.cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                logits = network(x, mask, pos, res=res, train=True)
            per_target_loss = F.binary_cross_entropy_with_logits(logits.float(), y, reduction="none")
            loss = (per_target_loss * weights).sum() / weights.sum().clamp_min(1e-8)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite confidence-weighted training loss")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            # A pretrained warm start should not reset EMA to the first few batches.
            with torch.no_grad():
                for averaged, current in zip(ema.parameters(), model.parameters()):
                    averaged.lerp_(current, 1 - ema_decay)
                for averaged, current in zip(ema.buffers(), model.buffers()):
                    averaged.copy_(current)
            total += loss.item()
            count += 1
            if rank == 0 and count % 100 == 0:
                say(f"ep {epoch} {count}/{len(train_loader)} loss {total/count:.4f} {(time.time()-epoch_start)/count:.2f}s/it")
        if partial_epoch:
            if completed_epochs == 0:
                raise RuntimeError("L2_TIME_GUARD: no full L2 epoch completed; no trained L2 checkpoint is available")
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            ema.load_state_dict(saved["model"], strict=True)
            del saved
            say("L2_TIME_GUARD: discarded partial epoch and restored last completed EMA")
            if ddp:
                dist.barrier()
            break
        prior_training_seconds = time.time() - epoch_start
        completed_epochs += 1
        if rank == 0:
            pg = predict(ema, gold.index)
            value, per_target = macro(gold_y, pg, LABELS)
            item = {"epoch": epoch, "loss": total / max(1, count), "gold_diagnostic_auc": value,
                    "gold_diagnostic_per_target": per_target, "seconds": time.time() - epoch_start}
            history.append(item)
            say(f"ep {epoch}: loss {item['loss']:.4f}; gold diagnostic {value:.5f}; elapsed {time.time()-started:.0f}s")
            provenance = {
                "targets": LABELS, "source_hashes": hashes, "fold": args.fold,
                "train_ids_sha256": hashlib.sha256("\n".join(train.index).encode()).hexdigest(),
                "val_ids_sha256": hashlib.sha256("\n".join(val.index).encode()).hexdigest(),
                "gold_excluded_from_gradient_training": True,
                "gold_auc_independent_validation": False,
                "weak_val_independent_of_label_generation": False,
                "selection": "last fixed epoch; no gold-best selection",
                "preprocessing": "A1 cached32 slices; area256; JPEGq95; real six slots without FS fallback",
                "source_knee": "recovered r2 ConvNeXt kernel output src/knee.py",
                "completed_epochs": completed_epochs,
                "requested_epochs": args.epochs,
                "label_source": "private L2 labels_l2.csv, confidence weights_l2.csv",
                "fold_source": "A1 labels_cons.csv; preserved weak-study report-group folds",
                "gold_source": "current official train.csv; all completely annotated studies excluded",
            }
            checkpoint = {"model": ema.state_dict(), "args": {
                "backbone": backbone, "res": res, "k_eval": k_eval, "k_train": k_train,
                "mode": "l2", "fold": args.fold, "epoch": epoch, "epochs": args.epochs,
                "lr": args.lr, "lr_head": args.lr_head, "dim": 384, "depth": 2,
                "drop": 0.2, "drop_path": 0.1, "confidence_weighted_bce": True,
            }, "provenance": provenance}
            path = checkpoint_path
            temporary = path.with_suffix(".pt.tmp")
            torch.save(checkpoint, temporary)
            os.replace(temporary, path)
            (out_dir / f"training_l2_f{args.fold}.json").write_text(json.dumps({"args": vars(args), "history": history, "provenance": provenance}, indent=2))
        if ddp:
            dist.barrier()
    if rank == 0:
        frames = [pd.DataFrame(pg, index=pd.Index(gold.index, name="StudyInstanceUID"), columns=LABELS).assign(set="gold")]
        full_val_saved = False
        if time.monotonic() + args.final_eval_reserve < deadline:
            pv = predict(ema, val.index)
            vy = (lab.loc[val.index, LABELS].to_numpy() > 0.5).astype(np.int64)
            val_auc, _ = macro(vy, pv, LABELS)
            frames.insert(0, pd.DataFrame(pv, index=pd.Index(val.index, name="StudyInstanceUID"), columns=LABELS).assign(set="val"))
            full_val_saved = True
            say(f"FINAL L2 fold {args.fold}: weak consensus diagnostic {val_auc:.5f}")
        else:
            say("L2_TIME_GUARD: full weak-val prediction omitted; completed checkpoint and gold diagnostics retained")
        gold_auc, gold_per_target = macro(gold_y, pg, LABELS)
        say(f"FINAL L2 fold {args.fold}: completed epochs {completed_epochs}; gold diagnostic {gold_auc:.5f}")
        say(json.dumps(gold_per_target))
        frame = pd.concat(frames)
        frame.to_csv(out_dir / f"oof_l2_convnext_tiny_f{args.fold}.csv")
        (out_dir / f"completion_l2_f{args.fold}.json").write_text(json.dumps({
            "fold": args.fold, "completed_epochs": completed_epochs, "requested_epochs": args.epochs,
            "k_train": k_train, "k_eval": k_eval,
            "full_weak_val_saved": full_val_saved, "gold_diagnostic_auc": gold_auc,
            "checkpoint": checkpoint_path.name, "checkpoint_sha256": digest(checkpoint_path),
            "seconds": time.monotonic() - process_started,
        }, indent=2))
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
