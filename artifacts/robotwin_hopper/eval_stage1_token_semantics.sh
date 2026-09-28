#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$SOURCE_ROOT"

RUN_ID=${PBS_JOBID:-manual-stage1-token-semantics}
OUTPUT_DIR=${OUTPUT_DIR:-$SOURCE_ROOT/outputs/stage1_token_semantics/$RUN_ID}
JOB_TMPDIR=${JOB_TMPDIR:-$SOURCE_ROOT/.t/$RUN_ID}
mkdir -p "$OUTPUT_DIR" "$JOB_TMPDIR"

export TMPDIR="$JOB_TMPDIR"
export HF_HOME=${HF_HOME:-$SOURCE_ROOT/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$JOB_TMPDIR/hf_datasets}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$JOB_TMPDIR/xdg}
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

echo "STAGE1_START job=$RUN_ID host=$(hostname) output=$OUTPUT_DIR"
nvidia-smi

if [[ ${RUN_TOKENIZER_STAGE:-1} == 1 ]]; then
  echo "TOKENIZER_STAGE_START $(date --iso-8601=seconds)"
  "$SOURCE_ROOT/.venv/bin/python" \
    "$SOURCE_ROOT/scripts/tokenizer/evaluate_stage1_semantics.py" \
    --suite robotwin5 \
    --device cuda \
    --max-samples 4096 \
    --intervention-samples 256 \
    --batch-size 256 \
    --num-workers 4 \
    --output-dir "$OUTPUT_DIR/tokenizer"
  echo "TOKENIZER_STAGE_COMPLETE $(date --iso-8601=seconds)"
fi

if [[ ${RUN_VLA_STAGE:-1} == 1 ]]; then
  echo "VLA_STAGE_START $(date --iso-8601=seconds)"
  "$SOURCE_ROOT/.venv/bin/python" \
    "$SOURCE_ROOT/scripts/robotwin/evaluate_stage1_vla.py" \
    --suite robotwin5 \
    --device cuda \
    --batch-size 8 \
    --generation-batch-size 4 \
    --num-workers 4 \
    --max-batches 16 \
    --max-generation-samples 32 \
    --output-dir "$OUTPUT_DIR/vla"
  echo "VLA_STAGE_COMPLETE $(date --iso-8601=seconds)"
fi

echo "STAGE1_COMPLETE output=$OUTPUT_DIR"
