#!/usr/bin/env bash
#SBATCH --job-name=BiCAT-BTT-all-200k
#SBATCH --partition=debug
#SBATCH --nodelist=crane6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --no-requeue
#SBATCH --output=/data/home/shailesh/Evan/bimanual_tokenizer/outputs/slurm-%j-btt-all.out

set -Eeuo pipefail
cd /data/home/shailesh/Evan/bimanual_tokenizer

DATA_ROOT="$PWD/data/BiCAT-BimanualYAM-JointOnly"
KEY_FILE="$HOME/.config/bimanual_tokenizer/wandb_evan_lxw.key"
RUN_DIR="${RUN_DIR_OVERRIDE:-$PWD/outputs/BiCAT_all_BTT_200k/${SLURM_JOB_ID:?}}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

test -s "$DATA_ROOT/source_manifest.json"
test -s "$KEY_FILE"
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  test -f "$RESUME_CHECKPOINT/training_state/training_step.json"
  RUN_MODE_ARGS=("run_mode=resume" "resume_checkpoint=$RESUME_CHECKPOINT")
else
  if [[ -e "$RUN_DIR/train" ]]; then
    echo "Refusing to overwrite existing run: $RUN_DIR/train" >&2
    exit 2
  fi
  RUN_MODE_ARGS=("run_mode=overwrite" "resume_checkpoint=null")
fi

mkdir -p "$RUN_DIR" "$PWD/.t/${SLURM_JOB_ID}"
export TMPDIR="$PWD/.t/${SLURM_JOB_ID}"
export HF_HOME="$PWD/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=online
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=1

TRAIN_ARGS=(
  --config-path "$PWD/artifacts/bicat_box"
  --config-name tokenizer_bicat_all_btt_200k
  "output_dir=$RUN_DIR/train"
  "run_name=BiCAT_all_BTT_14joint_200k_${SLURM_JOB_ID}"
  "${RUN_MODE_ARGS[@]}"
)

"$PWD/.venv/bin/python" -m praxis.scripts.train \
  "${TRAIN_ARGS[@]}" --cfg job --resolve > "$RUN_DIR/resolved_config.yaml"

echo "BTT all-task training: job=${SLURM_JOB_ID} node=$(hostname -s) GPUs=${CUDA_VISIBLE_DEVICES:-unset} output=$RUN_DIR"
echo "Data: $DATA_ROOT; 405 episodes; 287854 frames; action_joint_angles only"
echo "4 GPUs x 128 samples/GPU = GBS 512; target=200000 optimizer steps"
echo "W&B entity=evan_lxw-harbin-institute-of-technology project=BiCAT_realworld"

srun "$PWD/.venv/bin/python" scripts/tokenizer/exec_with_wandb_key.py \
  "$KEY_FILE" "$PWD/.venv/bin/torchrun" \
  --standalone --nnodes=1 --nproc-per-node=4 \
  -m praxis.scripts.train "${TRAIN_ARGS[@]}"
