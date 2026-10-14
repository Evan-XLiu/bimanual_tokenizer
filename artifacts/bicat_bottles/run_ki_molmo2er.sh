#!/usr/bin/env bash
set -Eeuo pipefail
SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$SOURCE_ROOT"
RUN_ID=${PBS_JOBID:-manual}
RUN_MODE=${1:-memory-test}
OUT="$SOURCE_ROOT/outputs/BiCAT_bottles_molmo2ER_${RUN_MODE}/$RUN_ID"
mkdir -p "$OUT" "$SOURCE_ROOT/.t/$RUN_ID"
exec > >(tee -a "$OUT/train.log") 2>&1
export HF_HOME="$SOURCE_ROOT/.cache/huggingface"
export HF_MODULES_CACHE="$SOURCE_ROOT/.cache/huggingface/modules"
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=online
export TMPDIR="$SOURCE_ROOT/.t/$RUN_ID"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CONFIG="${CONFIG_OVERRIDE:-$SOURCE_ROOT/artifacts/bicat_bottles/ki_molmo2er_ce_ki_joint.yaml}"
cp "$CONFIG" "$OUT/submitted_config.yaml"
"$SOURCE_ROOT/.venv/bin/python" - <<'PYGUARD'
import json
from pathlib import Path
p=Path("/scratch/xuanweiliu/bimanual_tokenizer/outputs/tokenizer_bottles_btt_50k/662436.hopper-m-02/train/checkpoints/latest")
step=json.loads((p/'training_state/training_step.json').read_text())['step']
assert step == 50000, f"Expected tokenizer step 50000, got {step}"
for name in ('policy_io.json','norm_stats.pt'):
    assert (p/'pretrained_model'/name).is_file(), f"Missing {name}"
print(f"Verified tokenizer checkpoint: {p}, step={step}")
PYGUARD
nvidia-smi
if [[ "$RUN_MODE" != train || "${MONITOR_GPU_MEMORY:-0}" == 1 || "$CONFIG" == *ki_oat_molmo2er_fft.yaml ]]; then
  nvidia-smi --query-gpu=timestamp,index,uuid,name,memory.used,memory.total,utilization.gpu \
    --format=csv --loop-ms=1000 > "$OUT/gpu_memory.csv" 2> "$OUT/gpu_monitor.err" &
  GPU_MONITOR_PID=$!
  trap 'kill "$GPU_MONITOR_PID" 2>/dev/null || true; wait "$GPU_MONITOR_PID" 2>/dev/null || true' EXIT
fi
if [[ "$RUN_MODE" == "memory-test" ]]; then
  "$SOURCE_ROOT/.venv/bin/python" artifacts/bicat_box/test_ki_molmo2er_memory.py "$CONFIG" "$OUT"
elif [[ "$RUN_MODE" == "smoke-test" ]]; then
  "$SOURCE_ROOT/.venv/bin/python" artifacts/bicat_box/test_ki_molmo2er_memory.py "$CONFIG" "$OUT" --smoke
elif [[ "$RUN_MODE" == "single-resize-test" ]]; then
  "$SOURCE_ROOT/.venv/bin/python" artifacts/bicat_box/test_ki_molmo2er_memory.py "$CONFIG" "$OUT" --smoke --single-resize
elif [[ "$RUN_MODE" == "train" ]]; then
  "$SOURCE_ROOT/.venv/bin/python" -m praxis.scripts.train \
    --config-path "$(dirname "$CONFIG")" --config-name "$(basename "$CONFIG" .yaml)" \
    output_dir="$OUT/train"
else
  echo "Unknown run mode: $RUN_MODE" >&2
  exit 2
fi
