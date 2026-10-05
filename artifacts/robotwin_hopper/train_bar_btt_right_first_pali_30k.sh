#!/usr/bin/env bash
set -Eeuo pipefail
cd "${PBS_O_WORKDIR:?}"
OUT="$PWD/outputs/hopper_bar_btt_right_first_pali_robotwin_new_30k/${PBS_JOBID:?}"
test ! -e "$OUT/resolved_config.yaml"
mkdir -p "$OUT" "$JOB_TMPDIR"
exec > >(tee "$OUT/train.log") 2>&1
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
export HF_HOME=${HF_HOME:-$PWD/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export WANDB_MODE=online
export WANDB_RUN_GROUP=btt_arm_order_ablation
export WANDB_TAGS=btt,right_then_left,30k
export WANDB_RUN_ID="btt-right-first-${PBS_JOBID%%.*}"
export PYTHONUNBUFFERED=1
.venv/bin/python - <<'PY'
import wandb
assert wandb.Api().api_key, 'W&B credentials are required for online training'
print('WANDB_ONLINE_AUTH_CONFIGURED')
PY
.venv/bin/python scripts/tokenizer/prepare_btt_right_first_training.py --output-dir "$OUT"
tar -cf "$OUT/training_source.tar" src/praxis scripts/tokenizer/prepare_btt_right_first_training.py
nvidia-smi
(
  while true; do
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    sleep 10
  done
) > "$OUT/gpu_usage.csv" &
GPU_MONITOR_PID=$!
cleanup() {
  kill "$GPU_MONITOR_PID" 2>/dev/null || true
  wait "$GPU_MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT
.venv/bin/praxis-train --config-path "$OUT" --config-name resolved_config
.venv/bin/python scripts/tokenizer/prepare_btt_right_first_training.py --output-dir "$OUT" --finalize
echo "BTT_RIGHT_FIRST_TRAIN_COMPLETE output=$OUT"
