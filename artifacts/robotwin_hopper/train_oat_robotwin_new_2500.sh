#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$SOURCE_ROOT"

RUN_ID=${PBS_JOBID:-manual}
DATASET_ROOT=${ROBOTWIN_NEW_DATA_ROOT:-$SOURCE_ROOT/data/lerobot/local/robotwin_new_demo_clean}
OUTPUT_BASE=${OUTPUT_BASE:-$SOURCE_ROOT/outputs/hopper_oat_robotwin_new_2500}
OUT="$OUTPUT_BASE/$RUN_ID"

test -f "$DATASET_ROOT/meta/info.json"
test -f "$DATASET_ROOT/meta/stats.json"
mkdir -p "$OUT"
exec > >(tee "$OUT/train.log") 2>&1

export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
export HF_HOME=${HF_HOME:-$SOURCE_ROOT/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export WANDB_MODE=online

echo "OAT_TRAIN_START job=$RUN_ID host=$(hostname) dataset_root=$DATASET_ROOT"
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
  --config-name train_codecs \
  'codecs@codec=tokens/oat' \
  +codec.latent_dim=4 \
  'codec.quantizer_levels=[8,8,6,5]' \
  dataset=libero \
  dataset.name=robotwin_new_demo_clean \
  dataset.repo_id=local/robotwin_new_demo_clean \
  dataset.root="$DATASET_ROOT" \
  dataset.bypass_force_cache_sync=true \
  device=cuda \
  trainer.use_wandb=true \
  run_name=robotwin_new_2500_oat_tokenizer_latent4 \
  output_dir="$OUT/train"

cleanup
trap - EXIT INT TERM
echo "OAT_TRAIN_COMPLETE output=$OUT"
