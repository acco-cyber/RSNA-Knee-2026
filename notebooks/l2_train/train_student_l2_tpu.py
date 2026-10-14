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
import torch_xla.debug.metrics as xla_metrics


EXPECTED_LABEL_GENERATION = {
    'dataset': 'gwey1025/rsna-knee-l2-labels-20261015',
    'version': 1,
    'annotations_complete': True,
    'annotated_studies': 4349,
    'report_only_studies': 4349,
    'studies': 4407,
    'gold_excluded_from_training': 58,
    'is_pilot': False,
    'algorithm': 'l2_fixed_round2_report_shares_v1',
    'hashes': {
        'manifest.json': '477c5902b982ed8dc67d730fbf55f807df1a40fbed1de38bc1211c980cd4964c',
        'labels_l2.csv': '4e28bbc05c08ae4509d0eab264ebb88365a0b2cb0683dcb40b63bce5b9dab315',
        'weights_l2.csv': '7d0d5f9a11c28f1ab58bc1dbb4220ea87c34bc75c0104fdc91977353f0db4839',
        'row_provenance.csv': '45564b7e483be02da6ab0a0913888ad307d74d7a8acaaf4c77652b3b27ca63fe',
    },
}


def verify_label_generation(folder):
    folder = Path(folder)
    for name, expected in EXPECTED_LABEL_GENERATION['hashes'].items():
        actual = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('TPU label generation digest differs: ' + name)
    manifest = json.loads((folder / 'manifest.json').read_text())
    for name in ('annotations_complete', 'annotated_studies', 'report_only_studies',
                 'studies', 'gold_excluded_from_training', 'is_pilot', 'algorithm'):
        if manifest.get(name) != EXPECTED_LABEL_GENERATION[name]:
            raise ValueError('TPU requires the final complete label generation: ' + name)
    for name in ('labels_l2.csv', 'weights_l2.csv', 'row_provenance.csv'):
        if manifest['outputs'].get(name) != EXPECTED_LABEL_GENERATION['hashes'][name]:
            raise ValueError('TPU label manifest output digest differs: ' + name)
    return dict(EXPECTED_LABEL_GENERATION)


def worker(index, args):
    sys.path.insert(0, args.src)
    from knee import KneeNet, LABELS, N_SLOTS, window_centres, make_views
    from train_student_l2 import load_training_tables, macro, digest
    label_generation = verify_label_generation(args.l2)
    device = torch_xla.device()
    rank, world = xr.global_ordinal(), xr.world_size()
    if xm.xla_device_hw(device) != 'TPU':
        raise RuntimeError('This notebook requires an actual TPU runtime')
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
    common_hashes = {
        'manifest.json': digest(Path(args.l2) / 'manifest.json'),
        'row_provenance.csv': digest(Path(args.l2) / 'row_provenance.csv'),
        'labels_l2.csv': digest(Path(args.l2) / 'labels_l2.csv'),
        'weights_l2.csv': digest(Path(args.l2) / 'weights_l2.csv'),
        'labels_cons.csv': digest(Path(args.a1) / 'labels_cons.csv'),
        'knee.py': digest(Path(args.src) / 'knee.py'),
        'training_tables_helper': digest(Path(args.src) / 'train_student_l2.py'),
        'trainer': digest(__file__),
    }
    if common_hashes['knee.py'] != '314c2af387a5cbd1d25f192f19f4ecc6d4ae4b30403e92e9d9b3ed2c325880b1':
        raise ValueError('TPU source differs from the recovered r2 knee.py')

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

    def save_epoch(model, fold, completed, provenance):
        # Every replica reaches both rendezvous calls; only global rank zero
        # serializes CPU tensors. Commit an EMA before optional final validation.
        saved = {'model': model.state_dict(), 'args': {
            'backbone': 'convnext_tiny', 'res': 256, 'k_eval': 12, 'k_train': 8,
            'mode': 'l2', 'fold': fold, 'epoch': completed - 1, 'epochs': args.epochs,
            'backend': 'xla', 'dim': 384, 'depth': 2, 'drop': 0.2, 'drop_path': 0.1,
            'lr': args.lr, 'lr_head': args.lr_head, 'confidence_weighted_bce': True,
        }, 'provenance': provenance}
        path = output / f'student_l2_convnext_tiny_f{fold}.pt'
        temporary = path.with_suffix('.pt.tmp')
        xm.save(saved, str(temporary), master_only=True, global_master=True)
        xm.rendezvous(f'l2-f{fold}-epoch{completed}-written')
        if rank == 0:
            os.replace(temporary, path)
        xm.rendezvous(f'l2-f{fold}-epoch{completed}-committed')
        return path

    for fold in folds:
        started = time.monotonic()
        torch.manual_seed(fold*100 + rank)
        xm.set_rng_state(fold*100 + rank, device=device)
        np.random.seed(fold*100 + rank)
        lab, targets, confidence, training, val, gold, gold_y = load_training_tables(args.a1, args.l2, args.competition, LABELS, fold)
        dataset = Dataset(training.index, targets.loc[training.index, LABELS].to_numpy(), confidence.loc[training.index, LABELS].to_numpy(), 8, True)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, drop_last=True, num_workers=0)
        if not len(loader):
            raise ValueError('Global TPU batch leaves no full training batch')
        checkpoint_path = Path(args.r2) / f'student_r2_convnext_tiny_f{fold}.pt'
        initial = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        saved_args = initial.get('args', {})
        if saved_args.get('mode') != 'r2' or saved_args.get('backbone') != 'convnext_tiny' or int(saved_args.get('fold', -1)) != fold or int(saved_args.get('res', -1)) != 256:
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
        # XLA supports the single-tensor capturable AdamW path. Its step and
        # bias corrections remain tensors instead of changing Python constants.
        optimizer = torch.optim.AdamW([{'params': encoder, 'lr': args.lr}, {'params': head, 'lr': args.lr_head}],
                                      weight_decay=0.02, capturable=True, foreach=False)
        hashes = {**common_hashes, 'warm_start': digest(checkpoint_path)}
        provenance = {
            'targets': LABELS, 'target_order': LABELS, 'source_hashes': hashes,
            'fold': fold, 'backend': 'xla',
            'train_ids_sha256': hashlib.sha256('\n'.join(training.index).encode()).hexdigest(),
            'val_ids_sha256': hashlib.sha256('\n'.join(val.index).encode()).hexdigest(),
            'gold_excluded_from_gradient_training': True,
            'gold_auc_independent_validation': False,
            'gold_auc_is_independent_validation': False,
            'weak_val_independent_of_label_generation': False,
            'selection': 'last fixed epoch; no gold-best selection',
            'preprocessing': 'A1 cached32 slices; area256; JPEGq95; real six slots without FS fallback',
            'source_knee': 'recovered r2 ConvNeXt kernel output src/knee.py',
            'requested_epochs': args.epochs, 'completed_epochs': 0,
            'label_source': 'private L2 labels_l2.csv, confidence weights_l2.csv',
            'label_generation': label_generation,
            'fold_source': 'A1 labels_cons.csv; preserved weak-study report-group folds',
            'gold_source': 'current official train.csv; all completely annotated studies excluded',
            'loss_reduction': 'sum confidence-weighted BCE / global replica confidence mass',
            'global_batch': global_batch, 'ema_decay': ema_decay,
            'precision': 'bfloat16 autocast with float32 loss and parameters',
            'optimizer': 'AdamW capturable=True foreach=False; clip after replica gradient reduction',
            'torch_version': torch.__version__, 'torch_xla_version': torch_xla.__version__,
        }
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
                weight_mass = xm.all_reduce(xm.REDUCE_SUM, weight.sum())
                numerator = (F.binary_cross_entropy_with_logits(logits.float(), target, reduction='none') * weight).sum()
                # reduce_gradients averages gradients. Multiplication by world
                # gives the confidence-weighted objective over the whole batch.
                loss = numerator * world / weight_mass.clamp_min(1e-8)
                loss.backward()
                xm.reduce_gradients(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0, foreach=False)
                optimizer.step()
                with torch.no_grad():
                    for shadow, parameter in zip(ema.parameters(), model.parameters()):
                        shadow.mul_(ema_decay).add_(parameter, alpha=1-ema_decay)
                    for shadow, buffer in zip(ema.buffers(), model.buffers()):
                        shadow.copy_(buffer)
                reduced_loss = xm.all_reduce(xm.REDUCE_SUM, loss.detach(), scale=1.0/world)
                total += reduced_loss
                xm.mark_step()
                if step % 100 == 0:
                    number = float(reduced_loss.cpu())
                    if not np.isfinite(number):
                        raise ValueError('Nonfinite L2 TPU training loss')
                    say(f'TPU fold{fold} epoch{epoch} step{step}/{len(loader)} loss={number:.5f} seconds={time.monotonic()-tick:.1f}')
            mean_loss = float((total / len(loader)).cpu())
            if not np.isfinite(mean_loss):
                raise ValueError('Nonfinite completed TPU epoch loss')
            history.append({'epoch': epoch+1, 'weighted_loss': mean_loss, 'seconds': time.monotonic()-tick})
            say(f'TPU_EPOCH_COMPLETE fold={fold} epoch={epoch+1} loss={mean_loss:.6f} seconds={time.monotonic()-tick:.1f}')
            provenance['completed_epochs'] = epoch + 1
            path = save_epoch(ema, fold, epoch + 1, provenance)
            if rank == 0:
                progress = {'fold': fold, 'completed_epochs': epoch + 1, 'requested_epochs': args.epochs,
                            'checkpoint': path.name, 'checkpoint_sha256': digest(path),
                            'history': history, 'provenance': provenance, 'final_validation_complete': False}
                (output / f'training_l2_f{fold}.json').write_text(json.dumps(progress, indent=2))
                (output / f'xla_metrics_f{fold}.txt').write_text(xla_metrics.metrics_report())
            xm.rendezvous(f'l2-f{fold}-epoch{epoch+1}-progress')
        pg = predict(ema, gold.index, f'l2-f{fold}-gold')
        pv = predict(ema, val.index, f'l2-f{fold}-val')
        if rank == 0:
            gold_auc, per_class = macro(gold_y, pg, LABELS)
            predictions = pd.concat([pd.DataFrame(pv, index=val.index, columns=LABELS).assign(set='val'),
                                     pd.DataFrame(pg, index=gold.index, columns=LABELS).assign(set='gold')])
            predictions.index.name = 'StudyInstanceUID'
            predictions.to_csv(output / f'oof_l2_convnext_tiny_f{fold}.csv')
            receipt = {'fold': fold, 'completed_epochs': args.epochs, 'requested_epochs': args.epochs,
                       'backend': 'xla', 'world_size': world,
                       'global_batch': global_batch, 'k_train': 8, 'k_eval': 12, 'gold_auc_diagnostic': gold_auc,
                       'gold_auc_is_independent_validation': False, 'gold_per_class': per_class,
                       'checkpoint_sha256': digest(path), 'checkpoint': path.name,
                       'elapsed_seconds': time.monotonic()-started, 'history': history,
                       'target_order': LABELS, 'gold_excluded_from_gradient_training': True,
                       'full_weak_val_saved': True, 'source_hashes': hashes,
                       'provenance': provenance}
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
                    'gold_auc_is_independent_validation': False,
                    'label_generation': label_generation}
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
    parsed_folds = [int(value) for value in configuration.folds.split(',')]
    if len(set(parsed_folds)) != len(parsed_folds) or not set(parsed_folds).issubset(range(5)):
        raise ValueError('folds must contain unique values between zero and four')
    if configuration.batch_size < 1:
        raise ValueError('batch-size must be positive')
    torch_xla.launch(worker, args=(configuration,))
