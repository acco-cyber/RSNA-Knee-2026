# TPU launch audit

Static review completed on 2026-10-14 UTC. No TPU execution has been validated.
The queued synthetic probe tests a single-device model gradient path; it does
not establish eight-replica collectives, real-data throughput, five-fold training
completion, or any leaderboard score.

The full notebook runs five matching r2 warm starts and four fixed epochs. CPU
`make_views` produces the original image transforms. Static all-slot ConvNeXt
encoding avoids data-dependent boolean indexing; masked features are zeroed
before the original head. Inference still uses twelve windows, resolution 256,
and the original six real acquisition slots. The recovered `knee.py` SHA is
checked before training. Gold studies remain excluded from the training loader.
The notebook and worker both require the final complete private dataset version
1: 4,349 annotated report-only studies and 4,407 total studies. Manifest, labels,
weights and row-provenance hashes are pinned. The standard-library preflight
runs before dependency installation and rejects a pilot or altered generation.

TPU-only audit fixes:

- Pin compatible torch 2.8.0, torch-xla 2.8.0, torchvision 0.23.0 and timm
  1.0.22 before imports. Check missing OpenCV/sklearn even when XLA is already
  installed. Python 3.11 or newer is required by this pinned setup.
- Use AdamW with `capturable=True, foreach=False`. The supported XLA
  single-tensor path keeps the step and bias corrections in the graph instead
  of changing Python coefficients on every batch. An epoch-wise learning rate
  can still cause one new compile per epoch.
- Divide confidence-weighted loss by global replica confidence mass, reduce
  gradients before clipping, and report loss averaged over replicas.
- Seed the XLA RNG explicitly. Validate warm-start resolution and nonempty
  full training batches.
- Atomically commit the EMA after each complete epoch, using the global master
  and all-replica rendezvous calls. This retains usable weights if final
  prediction fails. Optimizer state is not saved for exact training resumption.
- Record source hashes, fold memberships, model configuration, precision,
  optimization changes and excluded gold provenance. Final completion receipts
  require complete gold and weak-fold predictions. Epoch checkpoints alone are
  not full-fold completion receipts.

`training_l2_f{fold}.json` records completed epoch progress and
`xla_metrics_f{fold}.txt` exposes compiler/fallback diagnostics. Watch the first
real-data epoch before extrapolating duration: sequential CPU JPEG decoding and
augmentation may bottleneck eight workers, and encoding absent slots spends
additional TPU compute. Global batch size and adjusted EMA differ from the GPU
recipe. bfloat16 predictions and optimization are an unmeasured model change.

Notebook cells and embedded sources pass Python compilation and exact source
parity checks. Local torch/XLA/TPU libraries are unavailable, so this is static
verification. GPU trainer, GPU notebooks and their hashes were not modified.
The actual attached complete generation passes the isolated preflight; incorrect
label hashes and completeness expectations are rejected. The run may be queued
as an unvalidated candidate and will abort on an unsupported runtime.

Primary implementation references:

- [PyTorch/XLA 2.8 multi-device execution and save semantics](https://docs.pytorch.org/xla/release/r2.8/learn/pytorch-on-xla-devices.html)
- [PyTorch/XLA 2.8 gradient reduction and global-master serialization source](https://docs.pytorch.org/xla/release/r2.8/_modules/torch_xla/core/xla_model.html)
- [PyTorch 2.8 capturable Adam implementation](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/adam.py)
- [PyTorch 2.8 supported capturable devices, including XLA](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/optimizer.py)
