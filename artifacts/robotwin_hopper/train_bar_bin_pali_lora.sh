#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$SOURCE_ROOT"

RUN_ID=${PBS_JOBID:-manual}
OUTPUT_BASE=${OUTPUT_BASE:-$SOURCE_ROOT/outputs/hopper_train_lora}
OUT="$OUTPUT_BASE/$RUN_ID"
mkdir -p "$OUT"
exec > >(tee "$OUT/train.log") 2>&1

export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
export HF_HOME=${HF_HOME:-$SOURCE_ROOT/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}

echo "TRAIN_START job=$RUN_ID host=$(hostname)"
nvidia-smi

(
  while true; do
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits
    sleep 10
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
  policy.use_lora=true \
  policy.freeze_vision_encoder=true \
  policy.gradient_checkpointing=true \
  trainer.batch_size=64 \
  trainer.gradient_accumulation_steps=1 \
  trainer.num_workers=2 \
  trainer.max_steps=15000 \
  trainer.val_every=200 \
  trainer.reconst_every=200 \
  trainer.checkpoint_every=5000 \
  trainer.use_wandb=true \
  trainer.wandb_project=praxis-vla \
  output_dir="$OUT/train"

cleanup
trap - EXIT INT TERM
echo "TRAIN_COMPLETE output=$OUT"
