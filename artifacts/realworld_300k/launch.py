"""Launch full-data tokenizer runs and preserve earlier output directories."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
name = sys.argv[1]
assert name in {"btt", "oat", "fast"}
config_dir = ROOT / "artifacts/realworld_300k"
cfg = OmegaConf.load(config_dir / f"{name}.yaml")
manifest = json.loads((Path(cfg.dataset.root) / "source_manifest.json").read_text())
assert manifest["total_episodes"] == 405
assert manifest["total_frames"] == 287854
assert manifest["action_dim"] == 14
assert cfg.trainer.val_ratio == 0.0
assert cfg.trainer.batch_size == cfg.trainer.global_batch_size == 512
assert cfg.trainer.checkpoint_every == 0
assert cfg.trainer.checkpoint_candidates.best.k == 0
assert cfg.trainer.checkpoint_candidates.recent.k == 0
output = Path(cfg.output_dir)
stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
job_id = os.environ.get("PBS_JOBID", f"manual_{os.getpid()}")
logs = config_dir / "logs"
logs.mkdir(exist_ok=True)
if name != "fast":
    import torch
    assert torch.cuda.is_available(), "A CUDA GPU is required"
    assert torch.cuda.device_count() == 1
    gpu_name = torch.cuda.get_device_name(0)
    assert "H200" in gpu_name, gpu_name
    print(f"GPU: {gpu_name}", flush=True)
if output.exists() and any(output.iterdir()):
    archive = output.with_name(f"AT_before_300k_{stamp}_{job_id}")
    output.rename(archive)
    print(f"Preserved previous output at {archive}", flush=True)
command = [str(ROOT / ".venv/bin/python"), "-u", "-m", "praxis.scripts.train",
           "--config-path", str(config_dir), "--config-name", name]
logfile = logs / f"{name}_{job_id}.log"
print(f"Starting {name}: output={output}, log={logfile}", flush=True)
with logfile.open("x") as log:
    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
if result.returncode:
    print(logfile.read_text()[-12000:], flush=True)
    raise SystemExit(result.returncode)
latest = output / "checkpoints/latest"
step = json.loads((latest / "training_state/training_step.json").read_text())["step"]
assert step == (0 if name == "fast" else 300000), step
target = latest.with_name("fitted" if name == "fast" else "step_300000")
latest.rename(target)
print(f"Complete: {target}", flush=True)
