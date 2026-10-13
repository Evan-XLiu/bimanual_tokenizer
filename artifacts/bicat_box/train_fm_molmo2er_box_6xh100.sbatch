#!/usr/bin/env bash
#SBATCH --job-name=BiCAT-FM-6H100
#SBATCH --partition=debug
#SBATCH --nodelist=crane6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:6
#SBATCH --mem=240G
#SBATCH --time=7-00:00:00
#SBATCH --output=/data/home/shailesh/Evan/bimanual_tokenizer/outputs/slurm-%j.out

set -Eeuo pipefail
cd /data/home/shailesh/Evan/bimanual_tokenizer

RUN_DIR="$PWD/outputs/BiCAT_box_Molmo2ER_FM_30k/${SLURM_JOB_ID:?}"
mkdir -p "$RUN_DIR" "$PWD/.t/${SLURM_JOB_ID}"
export TMPDIR="$PWD/.t/${SLURM_JOB_ID}"
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=online
WANDB_KEY_FILE="$HOME/.config/bimanual_tokenizer/wandb_evan_lxw.key"
if [[ ! -s "$WANDB_KEY_FILE" ]]; then
  echo "Missing W&B API key file: $WANDB_KEY_FILE" >&2
  exit 1
fi
export WANDB_API_KEY="$(<"$WANDB_KEY_FILE")"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PWD/.venv/bin/python" -m praxis.scripts.train --cfg job \
  --config-path "$PWD/artifacts/bicat_box" \
  --config-name fm_molmo2er_box_30k \
  "output_dir=$RUN_DIR/train" \
  "run_name=BiCAT_box_Molmo2ER_FM_${SLURM_JOB_ID}" > "$RUN_DIR/resolved_config.yaml"

srun "$PWD/.venv/bin/torchrun" --standalone --nnodes=1 --nproc-per-node=6 \
  -m praxis.scripts.train \
  --config-path "$PWD/artifacts/bicat_box" \
  --config-name fm_molmo2er_box_30k \
  "output_dir=$RUN_DIR/train" \
  "run_name=BiCAT_box_Molmo2ER_FM_${SLURM_JOB_ID}"
