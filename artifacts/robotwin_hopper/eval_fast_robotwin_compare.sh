#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$SOURCE_ROOT"

RUN_ID=${PBS_JOBID:-manual}
OUTPUT_BASE=${OUTPUT_BASE:-$SOURCE_ROOT/outputs/hopper_fast_eval_compare}
OUT="$OUTPUT_BASE/$RUN_ID"
CHECKPOINT=${FAST_CHECKPOINT:-$SOURCE_ROOT/outputs/hopper_fast_q01_q99_noclip_v1024_filtered/598729.hopper-m-02/train/checkpoints/latest/pretrained_model}
DATASET_ROOT=${ROBOTWIN_DATA_ROOT:-$SOURCE_ROOT/data/lerobot/local/robotwin2_depth_merged_clean}
mkdir -p "$OUT"
exec > >(tee "$OUT/eval.log") 2>&1

export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
export HF_HOME=${HF_HOME:-$SOURCE_ROOT/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}

COMMON_ARGS=(
  --checkpoint "$CHECKPOINT"
  --dataset-repo-id local/robotwin2_depth_merged_clean
  --dataset-root "$DATASET_ROOT"
  --split all
  --batch-size 512
  --num-workers 8
)

echo "FAST_EVAL_START job=$RUN_ID host=$(hostname)"
echo "FAST_EVAL_LOCAL_START"
"$SOURCE_ROOT/.venv/bin/python" scripts/tokenizer/fast/token_stats.py \
  "${COMMON_ARGS[@]}" | tee "$OUT/local_q01_q99.txt"

echo "FAST_EVAL_OFFICIAL_PI_START"
"$SOURCE_ROOT/.venv/bin/python" scripts/tokenizer/fast/token_stats.py \
  "${COMMON_ARGS[@]}" \
  --bpe-tokenizer-path physical-intelligence/fast | tee "$OUT/official_pi.txt"

echo "FAST_EVAL_COMPLETE output=$OUT"
