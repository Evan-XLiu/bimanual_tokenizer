#!/usr/bin/env bash
set -Eeuo pipefail
SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$SOURCE_ROOT"
RUN_ID=${PBS_JOBID:-manual-btt-oneblock-$(date +%Y%m%d-%H%M%S)}
OUTPUT_BASE=${OUTPUT_BASE:-$SOURCE_ROOT/outputs/hopper_btt_oneblock_robotwin_new}
OUT="$OUTPUT_BASE/$RUN_ID"
RUN_NAME=${RUN_NAME:-robotwin_new_btt_oneblock_tokenizer}
JOB_TMPDIR=${JOB_TMPDIR:-$SOURCE_ROOT/.t/$RUN_ID}
test ! -e "$OUT/resolved_config.yaml"
mkdir -p "$OUT" "$JOB_TMPDIR"
exec > >(tee "$OUT/train.log") 2>&1
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
export HF_HOME=${HF_HOME:-$SOURCE_ROOT/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export WANDB_MODE=${WANDB_MODE:-online}
export TMPDIR="$JOB_TMPDIR"
export PYTHONUNBUFFERED=1

"$SOURCE_ROOT/.venv/bin/python" scripts/tokenizer/prepare_btt_oneblock_training.py \
  --stage tokenizer --output-dir "$OUT/train" --run-name "$RUN_NAME" \
  --save-config "$OUT/resolved_config.yaml"
"$SOURCE_ROOT/.venv/bin/praxis-train" --config-path "$OUT" --config-name resolved_config --cfg job \
  > "$OUT/hydra_config.yaml"
if [[ ${PREPARE_ONLY:-0} == 1 ]]; then
  echo "BTT_ONEBLOCK_PREPARED output=$OUT"
  exit 0
fi
echo "BTT_ONEBLOCK_TRAIN_START job=$RUN_ID output=$OUT"
nvidia-smi
"$SOURCE_ROOT/.venv/bin/praxis-train" --config-path "$OUT" --config-name resolved_config
echo "BTT_ONEBLOCK_TRAIN_COMPLETE output=$OUT"
