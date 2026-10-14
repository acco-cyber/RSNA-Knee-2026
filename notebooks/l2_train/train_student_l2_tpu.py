"""Distributed TPU warm-start of all five L2 ConvNeXt-T folds.

The JPEG cache, targets, model head and checkpoints are the recovered r2 ones.
Views are rendered on CPU; the static TPU encoder evaluates all six slots and
then zeros missing-slot features. ConvNeXt uses LayerNorm rather than BatchNorm.
Gold58 is a tuning diagnostic and never enters the training loader.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('PJRT_DEVICE', 'TPU')
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr


def worker(index, args):
    sys.path.insert(0, args.src)
    from knee import KneeNet, LABELS, N_SLOTS, window_centres, make_views
    from train_student_l2 import load_training_tables, macro, digest
    device = xm.xla_device()
    rank, world = xr.global_ordinal(), xr.world_size()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    folds = [int(value) for value in args.folds.split(',')]
    global_batch = args.batch_size * world
    ema_decay = 0.998 ** (global_batch / 4)
    say = (lambda message: print(message, flush=True)) if rank == 0 else (lambda message: None)
    say(f'TPU world={world} global_batch={global_batch} EMA={ema_decay:.6f}')
    receipts = []

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, ids, targets, weights, k, training):
            self.ids, self.targets, self.weights = list(ids), targets, weights
            self.k, self.training = k, training

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, position):
            study = self.ids[position]
            raw = np.zeros((N_SLOTS, self.k, 3, 256, 256), np.uint8)
            mask = np.zeros(N_SLOTS, bool)
            locations = np.zeros((N_SLOTS, self.k), np.float32)
            with np.load(Path(args.a1) / 'cache256' / f'{study}.npz', allow_pickle=False) as archive:
                for slot in range(N_SLOTS):
                    if f's{slot}_o' not in archive.files:
                        continue
                    data, offsets = archive[f's{slot}_d'], archive[f's{slot}_o']
                    n = len(offsets) - 1
                    if n < 3:
                        continue
                    centers = window_centres(n, self.k, self.training)
                    required = sorted({int(t) for center in centers for t in np.clip([center-1, center, center+1], 0, n-1)})
                    decoded = {}
                    for number in required:
                        image = cv2.imdecode(data[offsets[number]:offsets[number+1]], cv2.IMREAD_GRAYSCALE)
                        if image is None or image.shape != (256, 256):
                            raise ValueError('Invalid JPEG cache for ' + study)
                        decoded[number] = image
                    raw[slot] = np.stack([np.stack([decoded[int(t)] for t in np.clip([center-1, center, center+1], 0, n-1)]) for center in centers])
                    mask[slot], locations[slot] = True, centers / max(n-1, 1)
            if not mask.any():
                raise ValueError('Empty cached study: ' + study)
            # Fixed all-slot shape avoids data-dependent boolean indexing on XLA.
            views = make_views(torch.from_numpy(raw).flatten(0, 1), 256, self.training, self.k)
            target = self.targets[position] if self.targets is not None else np.zeros(12, np.float32)
            weight = self.weights[position] if self.weights is not None else np.ones(12, np.float32)
            return views, torch.from_numpy(mask), torch.from_numpy(locations), torch.from_numpy(np.asarray(target, np.float32)), torch.from_numpy(np.asarray(weight, np.float32))

    def forward(model, views, mask, locations, k):
        batch = mask.shape[0]
        features = model.enc(views.flatten(0, 1)).reshape(batch, N_SLOTS, k, -1)
        features = features * mask.unsqueeze(-1).unsqueeze(-1)
        return model.head(features, mask, locations)

    def predict(model, ids, tag):
        ids = list(ids)
        per_replica = (len(ids) + world - 1) // world
        assigned = [ids[position] for position in range(rank, len(ids), world)]
        actual = len(assigned)
        assigned += [ids[0]] * (per_replica - actual)
        dataset = Dataset(assigned, None, None, 12, False)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)
        predictions = []
        model.eval()
        with torch.no_grad():
            for views, mask, locations, _, _ in loader:
                with torch.autocast('xla', dtype=torch.bfloat16):
                    logits = forward(model, views.to(device), mask.to(device), locations.to(device), 12)
                xm.mark_step()
                predictions.append(logits.float().sigmoid().cpu().numpy()[0].tolist())
        chunks = xm.mesh_reduce(tag, (assigned[:actual], predictions[:actual]), lambda pieces: pieces)
        if rank != 0:
            return None
        lookup = {}
        for names, probabilities in chunks:
            for study, probability in zip(names, probabilities):
                if study in lookup:
                    raise ValueError('Duplicate distributed prediction ID')
                lookup[study] = probability
        if set(lookup) != set(ids):
            raise ValueError('Distributed prediction coverage differs from requested IDs')
        result = np.asarray([lookup[study] for study in ids], np.float32)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite TPU predictions')
        return result

    for fold in folds:
        started = time.monotonic()
        torch.manual_seed(fold*100 + rank)
        np.random.seed(fold*100 + rank)
        lab, targets, confidence, training, val, gold, gold_y = load_training_tables(args.a1, args.l2, args.competition, LABELS, fold)
        dataset = Dataset(training.index, targets.loc[training.index, LABELS].to_numpy(), confidence.loc[training.index, LABELS].to_numpy(), 8, True)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, drop_last=True, num_workers=0)
        checkpoint_path = Path(args.r2) / f'student_r2_convnext_tiny_f{fold}.pt'
        initial = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        saved_args = initial.get('args', {})
        if saved_args.get('mode') != 'r2' or saved_args.get('backbone') != 'convnext_tiny' or int(saved_args.get('fold', -1)) != fold:
            raise ValueError('TPU warm start requires the matching r2 ConvNeXt fold')
        model = KneeNet('convnext_tiny', pretrained=False)
        model.load_state_dict(initial['model'], strict=True)
        del initial
        model.to(device)
        ema = copy.deepcopy(model).eval()
        for parameter in ema.parameters():
            parameter.requires_grad_(False)
        encoder = list(model.enc.parameters())
        head = [parameter for name, parameter in model.named_parameters() if not name.startswith('enc.')]
        optimizer = torch.optim.AdamW([{'params': encoder, 'lr': args.lr}, {'params': head, 'lr': args.lr_head}], weight_decay=0.02)
        history = []
        for epoch in range(args.epochs):
            sampler.set_epoch(epoch)
            model.train()
            tick = time.monotonic()
            # Fixed LR per epoch avoids recompiling for every Python scheduler value.
            decay = 0.5 * (1 + np.cos(np.pi * epoch / max(1, args.epochs)))
            for group, base_lr in zip(optimizer.param_groups, [args.lr, args.lr_head]):
                group['lr'] = base_lr * decay
            total = torch.zeros((), device=device)
            for step, (views, mask, locations, target, weight) in enumerate(loader):
                views, mask, locations = views.to(device), mask.to(device), locations.to(device)
                target, weight = target.to(device), weight.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('xla', dtype=torch.bfloat16):
                    logits = forward(model, views, mask, locations, 8)
                loss = (F.binary_cross_entropy_with_logits(logits.float(), target, reduction='none') * weight).sum() / weight.sum().clamp_min(1)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                xm.optimizer_step(optimizer)
                with torch.no_grad():
                    for shadow, parameter in zip(ema.parameters(), model.parameters()):
                        shadow.mul_(ema_decay).add_(parameter, alpha=1-ema_decay)
                    for shadow, buffer in zip(ema.buffers(), model.buffers()):
                        shadow.copy_(buffer)
                total += loss.detach()
                xm.mark_step()
                if step % 100 == 0:
                    number = float(loss.cpu())
                    if not np.isfinite(number):
                        raise ValueError('Nonfinite L2 TPU training loss')
                    say(f'TPU fold{fold} epoch{epoch} step{step}/{len(loader)} loss={number:.5f} seconds={time.monotonic()-tick:.1f}')
            mean_loss = float((total / len(loader)).cpu())
            history.append({'epoch': epoch+1, 'weighted_loss': mean_loss, 'seconds': time.monotonic()-tick})
            say(f'TPU_EPOCH_COMPLETE fold={fold} epoch={epoch+1} loss={mean_loss:.6f} seconds={time.monotonic()-tick:.1f}')
        pg = predict(ema, gold.index, f'l2-f{fold}-gold')
        pv = predict(ema, val.index, f'l2-f{fold}-val')
        saved = {'model': ema.state_dict(), 'args': {'backbone': 'convnext_tiny', 'res': 256, 'k_eval': 12,
                 'k_train': 8, 'mode': 'l2', 'fold': fold, 'epoch': args.epochs-1, 'backend': 'xla'},
                 'provenance': {'target_order': LABELS, 'gold_excluded_from_gradient_training': True,
                 'gold_auc_is_independent_validation': False, 'completed_epochs': args.epochs,
                 'labels_sha256': digest(Path(args.l2)/'labels_l2.csv'), 'weights_sha256': digest(Path(args.l2)/'weights_l2.csv'),
                 'initial_checkpoint_sha256': digest(checkpoint_path)}}
        path = output / f'student_l2_convnext_tiny_f{fold}.pt'
        xm.save(saved, str(path), master_only=True)
        xm.rendezvous(f'l2-f{fold}-saved')
        if rank == 0:
            gold_auc, per_class = macro(gold_y, pg, LABELS)
            predictions = pd.concat([pd.DataFrame(pv, index=val.index, columns=LABELS).assign(set='val'),
                                     pd.DataFrame(pg, index=gold.index, columns=LABELS).assign(set='gold')])
            predictions.index.name = 'StudyInstanceUID'
            predictions.to_csv(output / f'oof_l2_convnext_tiny_f{fold}.csv')
            receipt = {'fold': fold, 'completed_epochs': args.epochs, 'backend': 'xla', 'world_size': world,
                       'global_batch': global_batch, 'k_train': 8, 'k_eval': 12, 'gold_auc_diagnostic': gold_auc,
                       'gold_auc_is_independent_validation': False, 'gold_per_class': per_class,
                       'checkpoint_sha256': digest(path), 'checkpoint': path.name,
                       'elapsed_seconds': time.monotonic()-started, 'history': history,
                       'target_order': LABELS, 'gold_excluded_from_gradient_training': True}
            (output / f'completion_l2_f{fold}.json').write_text(json.dumps(receipt, indent=2))
            receipts.append(receipt)
            say('TPU_L2_FOLD_COMPLETE ' + json.dumps(receipt))
        xm.rendezvous(f'l2-f{fold}-receipt')
        del model, ema, optimizer, loader, dataset
        xm.mark_step()
    if rank == 0:
        manifest = {'model_family': 'l2_convnext_tiny', 'backend': 'xla', 'requested_folds': folds,
                    'requested_epochs': args.epochs, 'five_folds_complete': folds == [0,1,2,3,4] and len(receipts)==5,
                    'smoke_only': False, 'targets': LABELS, 'receipts': receipts,
                    'gold_auc_is_independent_validation': False}
        (output/'l2_training_manifest.json').write_text(json.dumps(manifest, indent=2))
        say('TPU_L2_TRAINING_COMPLETE ' + json.dumps(manifest))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['a1','l2','r2','competition','src']:
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--folds', default='0,1,2,3,4')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--lr-head', type=float, default=1e-4)
    parser.add_argument('--output', default='/kaggle/working')
    configuration = parser.parse_args()
    if configuration.epochs < 1:
        raise ValueError('epochs must be positive')
    torch_xla.launch(worker, args=(configuration,))
