#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PBS_SCRIPT="$SOURCE_ROOT/artifacts/robotwin_hopper/eval_best_checkpoint_5tasks_20ep.pbs"
OUTPUT_BASE="$SOURCE_ROOT/outputs/robotwin_eval/best-checkpoint-5tasks-20ep"

cd "$SOURCE_ROOT"

submit_eval() {
  local job_name=$1
  local model_label=$2
  local checkpoint=$3

  test -f "$checkpoint/config.json"
  test -f "$checkpoint/model.safetensors"

  qsub \
    -N "$job_name" \
    -v "EVAL_MODEL_LABEL=$model_label,PRAXIS_CHECKPOINT=$checkpoint,ROBOTWIN_OUTPUT_BASE=$OUTPUT_BASE" \
    "$PBS_SCRIPT"
}

submit_eval \
  oatfft_5task20 \
  bar_oat_pali_fft_step13000 \
  "$SOURCE_ROOT/outputs/hopper_bar_oat_pali_fft_new_2500/601387.hopper-m-02/train/checkpoints/best/step_13000/pretrained_model"

submit_eval \
  bin_5task20 \
  bar_bin_pali_step17000 \
  "$SOURCE_ROOT/outputs/hopper_train/594020.hopper-m-02/train/checkpoints/best/step_17000/pretrained_model"

submit_eval \
  fastfft_5task20 \
  bar_fast_pali_fft_step5200 \
  "$SOURCE_ROOT/outputs/hopper_bar_fast_pali_fft/600669.hopper-m-02/train/checkpoints/best/step_5200/pretrained_model"
