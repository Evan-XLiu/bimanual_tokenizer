#!/usr/bin/env bash
#SBATCH --job-name=BiCAT-BAR-Molmo-30k-6GPU
#SBATCH --partition=debug
#SBATCH --nodelist=crane6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:6
#SBATCH --mem=320G
#SBATCH --no-requeue
#SBATCH --output=/data/home/shailesh/Evan/bimanual_tokenizer/outputs/slurm-%j-bar-6gpu-bs64.out

set -Eeuo pipefail
cd /data/home/shailesh/Evan/bimanual_tokenizer

KEY_FILE="$HOME/.config/bimanual_tokenizer/wandb_evan_lxw.key"
BTT_DIR="$PWD/outputs/BiCAT_all_BTT_200k/67292/train/checkpoints/protected/step_150000/pretrained_model"
RUN_DIR="${RUN_DIR_OVERRIDE:-$PWD/outputs/BiCAT_all_BAR_BTT_Molmo2ER_30k_6gpu_bs64/${SLURM_JOB_ID:?}}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
SMOKE_TEST="${SMOKE_TEST:-0}"

test -s "$KEY_FILE"
test -d "$BTT_DIR"
test -d /data/home/shailesh/checkpoints/Molmo2-ER

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
  --config-name train_bicat_all_bar_btt_molmo_30k
  "output_dir=$RUN_DIR/train"
  "run_name=BiCAT_all_BAR_BTT_Molmo2ER_30k_${SLURM_JOB_ID}"
  "${RUN_MODE_ARGS[@]}"
)
if [[ "$SMOKE_TEST" == 1 ]]; then
  TRAIN_ARGS+=(
    trainer.max_steps=1
    trainer.checkpoint_every=0
    trainer.log_every=1
    trainer.use_wandb=false
    trainer.wandb_required=false
  )
fi

"$PWD/.venv/bin/python" -m praxis.scripts.train \
  "${TRAIN_ARGS[@]}" --cfg job --resolve > "$RUN_DIR/resolved_config.yaml"

echo "BAR all-task training: job=${SLURM_JOB_ID} node=$(hostname -s) GPUs=${CUDA_VISIBLE_DEVICES:-unset} output=$RUN_DIR smoke=$SMOKE_TEST"
echo "All 405 episodes; 14 joint dimensions; 6 GPUs x 64/GPU = GBS 384"
echo "BTT 150k frozen; Molmo2-ER FFT; 30000 optimizer steps; checkpoint from 2000 every 1000"

srun "$PWD/.venv/bin/python" scripts/tokenizer/exec_with_wandb_key.py \
  "$KEY_FILE" "$PWD/.venv/bin/torchrun" \
  --standalone --nnodes=1 --nproc-per-node=6 \
  -m praxis.scripts.train "${TRAIN_ARGS[@]}"
