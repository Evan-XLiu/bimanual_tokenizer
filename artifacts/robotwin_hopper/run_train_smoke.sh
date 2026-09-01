#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$SOURCE_ROOT"

RUN_ID=${PBS_JOBID:-manual}
OUTPUT_BASE=${OUTPUT_BASE:-$SOURCE_ROOT/outputs/hopper_smoke}
OUT="$OUTPUT_BASE/$RUN_ID"
SMOKE_BATCH_SIZE=${SMOKE_BATCH_SIZE:-2}
SMOKE_MAX_STEPS=${SMOKE_MAX_STEPS:-20}
SMOKE_USE_WANDB=${SMOKE_USE_WANDB:-false}
SMOKE_VAL_EVERY=${SMOKE_VAL_EVERY:-$SMOKE_MAX_STEPS}
SMOKE_MAX_VAL_BATCHES=${SMOKE_MAX_VAL_BATCHES:-5}
SMOKE_RECONST_EVERY=${SMOKE_RECONST_EVERY:-$SMOKE_MAX_STEPS}
SMOKE_MAX_RECONST_BATCHES=${SMOKE_MAX_RECONST_BATCHES:-1}
SMOKE_CHECKPOINT_EVERY=${SMOKE_CHECKPOINT_EVERY:-$SMOKE_MAX_STEPS}
SMOKE_USE_LORA=${SMOKE_USE_LORA:-false}
SMOKE_FREEZE_VISION_ENCODER=${SMOKE_FREEZE_VISION_ENCODER:-false}
mkdir -p "$OUT"
exec > >(tee "$OUT/smoke.log") 2>&1

export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
export HF_HOME=${HF_HOME:-$SOURCE_ROOT/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}

echo "SMOKE_START job=$RUN_ID host=$(hostname)"
nvidia-smi
"$SOURCE_ROOT/.venv/bin/python" - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"gpu_count={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable inside the Hopper container")
p = torch.cuda.get_device_properties(0)
print(f"gpu={p.name} total_memory_gib={p.total_memory / 2**30:.2f}")
PY

"$SOURCE_ROOT/.venv/bin/python" - <<'PY'
from transformers import AutoConfig
AutoConfig.from_pretrained("google/paligemma2-3b-mix-224")
print("PALIGEMMA_ACCESS_OK")
PY

(
  while true; do
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits
    sleep 2
  done
) >"$OUT/gpu_usage.csv" &
GPU_MONITOR_PID=$!
cleanup() {
  kill "$GPU_MONITOR_PID" 2>/dev/null || true
  wait "$GPU_MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

"$SOURCE_ROOT/.venv/bin/praxis-train" \
  policy=bar_bin_pali \
  dataset=robotwin2_clean \
  policy.action_tokenizer.pretrained_path="$SOURCE_ROOT/outputs/hopper_smoke/bin_robotwin_bootstrap/checkpoints/latest/pretrained_model" \
  policy.use_lora="$SMOKE_USE_LORA" \
  policy.freeze_vision_encoder="$SMOKE_FREEZE_VISION_ENCODER" \
  policy.gradient_checkpointing=true \
  trainer.batch_size="$SMOKE_BATCH_SIZE" \
  trainer.gradient_accumulation_steps=1 \
  trainer.num_workers=2 \
  trainer.max_steps="$SMOKE_MAX_STEPS" \
  trainer.log_every=1 \
  trainer.val_every="$SMOKE_VAL_EVERY" \
  trainer.max_val_batches="$SMOKE_MAX_VAL_BATCHES" \
  trainer.reconst_every="$SMOKE_RECONST_EVERY" \
  trainer.max_reconst_batches="$SMOKE_MAX_RECONST_BATCHES" \
  trainer.checkpoint_every="$SMOKE_CHECKPOINT_EVERY" \
  trainer.use_wandb="$SMOKE_USE_WANDB" \
  trainer.wandb_project=praxis-vla \
  output_dir="$OUT/train"

cleanup
trap - EXIT INT TERM
awk -F, 'BEGIN { max=0 } { gsub(/ /, "", $4); if ($4+0 > max) max=$4+0 } END { print "peak_nvidia_smi_mib=" max }' "$OUT/gpu_usage.csv"
echo "SMOKE_COMPLETE output=$OUT"
