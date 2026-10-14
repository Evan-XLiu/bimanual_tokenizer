#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$SOURCE_ROOT"
RUN_ID=${PBS_JOBID:-manual-tokenizer-plate-btt-50k}
OUT=${TRAIN_OUTPUT_PARENT:-"$SOURCE_ROOT/outputs/tokenizer_plate_btt_50k/$RUN_ID"}
JOB_TMPDIR="$SOURCE_ROOT/.t/$RUN_ID"
mkdir -p "$OUT" "$JOB_TMPDIR"
exec > >(tee -a "$OUT/train.log") 2>&1

export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
export HF_HOME=${HF_HOME:-$SOURCE_ROOT/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
export TMPDIR="$JOB_TMPDIR"
export PYTHONUNBUFFERED=1

TRAIN_ARGS=(
  --config-path "$SOURCE_ROOT/artifacts/bicat_plate"
  --config-name tokenizer_plate_btt_50k
  output_dir="$OUT/train"
)
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  test -f "$RESUME_CHECKPOINT/training_state/training_step.json"
  TRAIN_ARGS+=(run_mode=resume resume_checkpoint="$RESUME_CHECKPOINT")
fi
test -f "$SOURCE_ROOT/data/BiCAT-BimanualYAM-Dataset/plate/meta/info.json"
echo "BTT_PLATE_TRAIN_START job=$RUN_ID host=$(hostname) project=BiCAT_RealWorld name=tokenizer_plate_btt_50k"
nvidia-smi
"$SOURCE_ROOT/.venv/bin/python" -m praxis.scripts.train \
  "${TRAIN_ARGS[@]}" --cfg job --resolve > "$OUT/resolved_config.yaml"
cp artifacts/bicat_plate/normalization.json "$OUT/normalization.json"
"$SOURCE_ROOT/.venv/bin/praxis-train" "${TRAIN_ARGS[@]}"
echo "BTT_PLATE_TRAIN_COMPLETE output=$OUT"
