#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$SOURCE_ROOT"

RUN_ID=${PBS_JOBID:-manual}
TRAIN_VARIANT=${TRAIN_VARIANT:?Set TRAIN_VARIANT to lora or fft}
FAST_CHECKPOINT=${FAST_CHECKPOINT:-$SOURCE_ROOT/outputs/hopper_fast_q01_q99_noclip_v1024_filtered/598729.hopper-m-02/train/checkpoints/latest/pretrained_model}
EXCLUDED_EPISODES_FILE=${EXCLUDED_EPISODES_FILE:-$SOURCE_ROOT/artifacts/robotwin_hopper/fast_episode_filter/excluded_episodes.json}
ACTION_STATS_FILE=${ACTION_STATS_FILE:-$SOURCE_ROOT/artifacts/robotwin_hopper/fast_episode_filter/retained_action_stats.json}
FREEZE_VISION_ENCODER=${FREEZE_VISION_ENCODER:-true}

case "$TRAIN_VARIANT" in
  lora)
    USE_LORA=true
    OUTPUT_BASE=${OUTPUT_BASE:-$SOURCE_ROOT/outputs/hopper_bar_fast_pali_lora}
    ;;
  fft)
    USE_LORA=false
    OUTPUT_BASE=${OUTPUT_BASE:-$SOURCE_ROOT/outputs/hopper_bar_fast_pali_fft}
    ;;
  *)
    echo "Unsupported TRAIN_VARIANT=$TRAIN_VARIANT (expected lora or fft)" >&2
    exit 2
    ;;
esac

test -d "$FAST_CHECKPOINT"
test -f "$FAST_CHECKPOINT/tokenizer.json"
test -f "$EXCLUDED_EPISODES_FILE"
test -f "$ACTION_STATS_FILE"
OUT="$OUTPUT_BASE/$RUN_ID"
mkdir -p "$OUT"
exec > >(tee "$OUT/train.log") 2>&1

export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
export HF_HOME=${HF_HOME:-$SOURCE_ROOT/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export WANDB_MODE=online

echo "TRAIN_START job=$RUN_ID variant=$TRAIN_VARIANT host=$(hostname) fast_checkpoint=$FAST_CHECKPOINT"
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
  policy=bar_fast_pali \
  dataset=robotwin2_clean \
  dataset.exclude_episodes_file="$EXCLUDED_EPISODES_FILE" \
  dataset.action_stats_file="$ACTION_STATS_FILE" \
  policy.action_tokenizer.pretrained_path="$FAST_CHECKPOINT" \
  policy.action_tokenizer.bpe_tokenizer_path=null \
  policy.freeze_action_tokenizer=true \
  policy.use_lora="$USE_LORA" \
  policy.freeze_vision_encoder="$FREEZE_VISION_ENCODER" \
  policy.gradient_checkpointing=true \
  trainer.batch_size=64 \
  trainer.gradient_accumulation_steps=1 \
  trainer.num_workers=2 \
  trainer.max_steps=15000 \
  trainer.log_every=1 \
  trainer.val_every=100 \
  trainer.reconst_every=100 \
  trainer.checkpoint_every=1000 \
  trainer.checkpoint_candidates.best.k=3 \
  trainer.checkpoint_candidates.recent.k=2 \
  trainer.use_wandb=true \
  trainer.wandb_project=praxis-vla \
  run_name="robotwin_bar_fast_pali_${TRAIN_VARIANT}_q01_q99_v1024_filtered_15000" \
  output_dir="$OUT/train"

cleanup
trap - EXIT INT TERM
echo "TRAIN_COMPLETE output=$OUT"
