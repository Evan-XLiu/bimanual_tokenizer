#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$SOURCE_ROOT"

RUN_ID=${PBS_JOBID:-manual}
DEVICE=${DEVICE:-cuda}
OUTPUT_BASE=${OUTPUT_BASE:-$SOURCE_ROOT/outputs/hopper_fast_q01_q99_noclip_v1024_filtered}
OUT="$OUTPUT_BASE/$RUN_ID"
mkdir -p "$OUT"
exec > >(tee "$OUT/train.log") 2>&1

export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
export HF_HOME=${HF_HOME:-$SOURCE_ROOT/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=online

echo "FAST_FIT_START job=$RUN_ID host=$(hostname)"
GPU_MONITOR_PID=
if [[ "$DEVICE" == cuda ]]; then
  nvidia-smi
  (
    while true; do
      nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits
      sleep 10
    done
  ) >"$OUT/gpu_usage.csv" &
  GPU_MONITOR_PID=$!
fi
cleanup() {
  if [[ -n "$GPU_MONITOR_PID" ]]; then
    kill "$GPU_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_MONITOR_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"$SOURCE_ROOT/.venv/bin/praxis-train" \
  --config-name train_codecs \
  'codecs@codec=tokens/fast' \
  dataset=robotwin2_clean \
  dataset.exclude_episodes_file="$SOURCE_ROOT/artifacts/robotwin_hopper/fast_episode_filter/excluded_episodes.json" \
  dataset.action_stats_file="$SOURCE_ROOT/artifacts/robotwin_hopper/fast_episode_filter/retained_action_stats.json" \
  codec.bpe_tokenizer_path=null \
  codec.vocab_size=1024 \
  codec.action_horizon=32 \
  codec.max_fit_chunks=600000 \
  codec.action_normalization_mode=quantile_01_99 \
  trainer.val_ratio=0.0 \
  trainer.batch_size=512 \
  trainer.num_workers=8 \
  trainer.use_wandb=true \
  trainer.wandb_project=tokenizer_pretrain \
  device="$DEVICE" \
  run_name=robotwin_fast_q01_q99_noclip_v1024_filtered \
  output_dir="$OUT/train"

cleanup
trap - EXIT INT TERM
echo "FAST_FIT_COMPLETE output=$OUT"
