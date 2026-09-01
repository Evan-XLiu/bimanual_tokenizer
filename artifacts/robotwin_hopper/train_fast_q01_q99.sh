#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$SOURCE_ROOT"

RUN_ID=${PBS_JOBID:-manual}
OUTPUT_BASE=${OUTPUT_BASE:-$SOURCE_ROOT/outputs/hopper_fast_q01_q99}
OUT="$OUTPUT_BASE/$RUN_ID"
mkdir -p "$OUT"
exec > >(tee "$OUT/train.log") 2>&1

export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
export HF_HOME=${HF_HOME:-$SOURCE_ROOT/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}

echo "FAST_FIT_START job=$RUN_ID host=$(hostname)"
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
  'codecs@codec=tokens/fast' \
  dataset=robotwin2_clean \
  codec.bpe_tokenizer_path=null \
  codec.action_horizon=32 \
  codec.max_fit_chunks=600000 \
  codec.action_normalization_mode=quantile_01_99 \
  codec.clip_quantile_normalized_actions=true \
  trainer.val_ratio=0.0 \
  trainer.batch_size=512 \
  trainer.num_workers=8 \
  trainer.use_wandb=true \
  trainer.wandb_project=tokenizer_pretrain \
  device=cuda \
  run_name=robotwin_fast_q01_q99_full \
  output_dir="$OUT/train"

cleanup
trap - EXIT INT TERM
echo "FAST_FIT_COMPLETE output=$OUT"
