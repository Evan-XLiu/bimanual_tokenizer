#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$SOURCE_ROOT"

RUN_ID=${PBS_JOBID:-manual-btt-independent-robotwin-new}
DATASET_ROOT=${ROBOTWIN_NEW_DATA_ROOT:-$SOURCE_ROOT/data/lerobot/local/robotwin_new_demo_clean}
OUTPUT_BASE=${OUTPUT_BASE:-$SOURCE_ROOT/outputs/hopper_btt_independent_robotwin_new}
RUN_NAME=${RUN_NAME:-BTT_independent_arm_ablation}
OUT="$OUTPUT_BASE/$RUN_ID"
JOB_TMPDIR=${JOB_TMPDIR:-$SOURCE_ROOT/.t/$RUN_ID}

test -f "$DATASET_ROOT/meta/info.json"
test -f "$DATASET_ROOT/meta/stats.json"
mkdir -p "$OUT" "$JOB_TMPDIR"
exec > >(tee "$OUT/train.log") 2>&1

export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
export HF_HOME=${HF_HOME:-$SOURCE_ROOT/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export WANDB_MODE=${WANDB_MODE:-online}
export TMPDIR="$JOB_TMPDIR"
export PYTHONUNBUFFERED=1

TRAIN_ARGS=(
  --config-name train_codecs
  'codecs@codec=tokens/btt_independent'
  'dataset=robotwin_new_demo_clean'
  dataset.root="$DATASET_ROOT"
  trainer.lr=5.0e-05
  trainer.max_steps=300000
  trainer.wandb_project=tokenizer_pretrain
  run_name="$RUN_NAME"
  output_dir="$OUT/train"
)

echo "BTT_INDEPENDENT_TRAIN_START job=$RUN_ID host=$(hostname) dataset_root=$DATASET_ROOT lr=5.0e-05 max_steps=300000"
nvidia-smi

"$SOURCE_ROOT/.venv/bin/python" -m praxis.scripts.train \
  "${TRAIN_ARGS[@]}" --cfg job > "$OUT/resolved_config.yaml"

"$SOURCE_ROOT/.venv/bin/praxis-train" "${TRAIN_ARGS[@]}"

echo "BTT_INDEPENDENT_TRAIN_COMPLETE output=$OUT"
