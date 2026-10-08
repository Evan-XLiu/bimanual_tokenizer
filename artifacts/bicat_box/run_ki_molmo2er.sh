#!/usr/bin/env bash
set -Eeuo pipefail
SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$SOURCE_ROOT"
RUN_ID=${PBS_JOBID:-manual}
RUN_MODE=${1:-memory-test}
OUT="$SOURCE_ROOT/outputs/BiCAT_box_molmo2ER_${RUN_MODE}/$RUN_ID"
mkdir -p "$OUT" "$SOURCE_ROOT/.t/$RUN_ID"
exec > >(tee -a "$OUT/train.log") 2>&1
export HF_HOME="$SOURCE_ROOT/.cache/huggingface"
export HF_MODULES_CACHE="$SOURCE_ROOT/.cache/huggingface/modules"
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=online
export TMPDIR="$SOURCE_ROOT/.t/$RUN_ID"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CONFIG="${CONFIG_OVERRIDE:-$SOURCE_ROOT/artifacts/bicat_box/ki_molmo2er_fft.yaml}"
cp "$CONFIG" "$OUT/submitted_config.yaml"
nvidia-smi
if [[ "$RUN_MODE" == "memory-test" ]]; then
  "$SOURCE_ROOT/.venv/bin/python" artifacts/bicat_box/test_ki_molmo2er_memory.py "$CONFIG" "$OUT"
elif [[ "$RUN_MODE" == "train" ]]; then
  "$SOURCE_ROOT/.venv/bin/python" -m praxis.scripts.train \
    --config-path "$SOURCE_ROOT/artifacts/bicat_box" --config-name ki_molmo2er_fft \
    output_dir="$OUT/train"
else
  echo "Unknown run mode: $RUN_MODE" >&2
  exit 2
fi
