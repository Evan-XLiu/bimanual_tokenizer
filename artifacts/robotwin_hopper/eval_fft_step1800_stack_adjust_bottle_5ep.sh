#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$SOURCE_ROOT"

export ROBOTWIN_ROOT=${ROBOTWIN_ROOT:-/scratch/xuanweiliu/RoboTwin}
export ROBOTWIN_PYTHON=${ROBOTWIN_PYTHON:-/scratch/xuanweiliu/conda_envs/robotwin/bin/python}
export PRAXIS_CHECKPOINT=${PRAXIS_CHECKPOINT:-/scratch/xuanweiliu/bimanual_tokenizer/outputs/hopper_bar_fast_pali_fft/598979.hopper-m-02/train/checkpoints/best/step_1800/pretrained_model}
export EPISODES_PER_TASK=5
export RECORD_EPISODES_PER_TASK=5
export MAX_SEED_ATTEMPTS=${MAX_SEED_ATTEMPTS:-250}
export ROBOTWIN_OUTPUT_DIR=${ROBOTWIN_OUTPUT_DIR:-/scratch/xuanweiliu/bimanual_tokenizer/outputs/robotwin_eval/fft-step1800-stack-adjust-bottle-5ep}
export POLICY_PORT=${POLICY_PORT:-50100}

for task_name in stack_blocks_two adjust_bottle; do
  export ROBOTWIN_TASK="$task_name"
  export ROBOTWIN_RUN_ID="fft-step1800-${task_name}-5ep"
  bash artifacts/robotwin_hopper/run_eval.sh
done
