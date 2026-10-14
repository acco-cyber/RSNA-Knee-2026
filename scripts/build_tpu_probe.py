"""Package a read-only model compatibility check for the remaining TPU quota."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
destination = root / 'notebooks' / 'l2_tpu_probe'
destination.mkdir(parents=True, exist_ok=True)

source = '''import os, sys, time, json, glob, subprocess, importlib.util
from pathlib import Path
started = time.monotonic()
os.environ.setdefault("PJRT_DEVICE", "TPU")
if importlib.util.find_spec("torch_xla") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "torch==2.8.0", "torch_xla[tpu]==2.8.0", "timm==1.0.22"], check=True)
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.debug.metrics as met
print("VERSIONS", torch.__version__, torch_xla.__version__, flush=True)
device = xm.xla_device()
print("XLA_DEVICE", device, "world", xr.world_size(), "runtime_devices", xr.global_runtime_device_count(), flush=True)
paths = glob.glob("/kaggle/input/**/src/knee.py", recursive=True)
assert paths, "Recovered r2 model source missing"
source = next((p for p in paths if "r2-convnext" in p), paths[0])
sys.path.insert(0, str(Path(source).parent))
from knee import KneeNet, make_views, LABELS
checkpoint_paths = glob.glob("/kaggle/input/**/student_r2_convnext_tiny_f0.pt", recursive=True)
assert len(checkpoint_paths) == 1, checkpoint_paths
checkpoint = torch.load(checkpoint_paths[0], map_location="cpu", weights_only=False)
model = KneeNet("convnext_tiny", pretrained=False)
model.load_state_dict(checkpoint["model"], strict=True)
del checkpoint
torch.backends.mha.set_fastpath_enabled(False)
model.to(device).train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
batch, slots, windows, resolution = 1, 6, 8, 256
raw = torch.randint(0, 256, (batch * slots * windows, 3, resolution, resolution), dtype=torch.uint8)
# CPU augmentation produces the same 256px views without dynamic TPU masking.
# The ConvNeXt backbone has no BatchNorm, so running masked slots and zeroing their
# features preserves the model's inference graph while fixing XLA tensor shapes.
views = make_views(raw, resolution, False, windows).to(device)
mask = torch.ones(batch, slots, dtype=torch.bool).to(device)
positions = torch.linspace(0.04, 0.96, windows).repeat(batch, slots, 1).to(device)
target = torch.full((batch, len(LABELS)), 0.5).to(device)
times = []
for step in range(2):
    tick = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("xla", dtype=torch.bfloat16):
        features = model.enc(views).reshape(batch, slots, windows, -1)
        features = features * mask.unsqueeze(-1).unsqueeze(-1)
        logits = model.head(features, mask, positions)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), target)
    loss.backward()
    xm.optimizer_step(optimizer)
    xm.mark_step()
    number = float(loss.cpu())
    assert number == number, "Nonfinite TPU loss"
    times.append(time.monotonic() - tick)
    print("TPU_GRADIENT_STEP", step, number, times[-1], flush=True)
receipt = {"compatible_gradient_steps": len(times), "torch": torch.__version__, "torch_xla": torch_xla.__version__,
           "step_seconds": times, "elapsed_seconds": time.monotonic() - started,
           "world_size": xr.world_size(), "runtime_device_count": xr.global_runtime_device_count(),
           "training_complete": False, "learned_l2_checkpoint": False}
Path("/kaggle/working/tpu_probe_receipt.json").write_text(json.dumps(receipt, indent=2))
Path("/kaggle/working/tpu_metrics.txt").write_text(met.metrics_report())
print("TPU_PROBE_COMPLETE", json.dumps(receipt), flush=True)
'''
compile(source, 'tpu_probe', 'exec')
notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}},
            'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source': ['# RSNA Knee L2 TPU compatibility probe\n\nTwo synthetic gradient steps test the recovered model; they do not train L2 or establish a score.\n']},
                      {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': source.splitlines(keepends=True)}]}
(destination / 'l2_tpu_probe.ipynb').write_text(json.dumps(notebook, indent=1))
metadata = {'id': 'gwey1025/rsna-knee-l2-tpu-probe', 'title': 'RSNA Knee L2 TPU Probe', 'code_file': 'l2_tpu_probe.ipynb',
            'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
            'enable_gpu': False, 'enable_tpu': True, 'enable_internet': True,
            'machine_shape': 'TpuV5E8', 'dataset_sources': [],
            'kernel_sources': ['gwey1025/rsna-knee-b-student-f01234-r2-convnext'], 'competition_sources': [], 'model_sources': []}
(destination / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
print(destination)
