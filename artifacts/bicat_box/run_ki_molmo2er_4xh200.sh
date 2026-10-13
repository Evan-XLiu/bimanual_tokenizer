#!/usr/bin/env bash
set -Eeuo pipefail
SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$SOURCE_ROOT"
RUN_MODE=${1:-train}
case "$RUN_MODE" in
  train|dry-run) ;;
  *) echo "Usage: $0 [train|dry-run]" >&2; exit 2 ;;
esac
NPROC_PER_NODE=4
# Full box dataset: 70,152 usable frames. At 4 x 48, 366 steps/epoch.
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-192}
GRAD_ACCUM=${GRAD_ACCUM:-1}
if ! [[ "$GLOBAL_BATCH_SIZE" =~ ^[1-9][0-9]*$ && "$GRAD_ACCUM" =~ ^[1-9][0-9]*$ ]]; then
  echo "GLOBAL_BATCH_SIZE and GRAD_ACCUM must be positive integers" >&2
  exit 2
fi
if (( GLOBAL_BATCH_SIZE % (NPROC_PER_NODE * GRAD_ACCUM) != 0 )); then
  echo "GLOBAL_BATCH_SIZE must be divisible by 4 * GRAD_ACCUM" >&2
  exit 2
fi
PER_RANK_BATCH=$((GLOBAL_BATCH_SIZE / (NPROC_PER_NODE * GRAD_ACCUM)))
RUN_ID=${PBS_JOBID:-manual-$(date +%Y%m%d-%H%M%S)}
OUT="$SOURCE_ROOT/outputs/BiCAT_box_molmo2ER_4xh200/$RUN_ID"
mkdir -p "$OUT" "$SOURCE_ROOT/.t/$RUN_ID"
exec > >(tee -a "$OUT/train.log") 2>&1
export HF_HOME="$SOURCE_ROOT/.cache/huggingface"
export HF_MODULES_CACHE="$HF_HOME/modules"
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=${WANDB_MODE:-online}
export TMPDIR="$SOURCE_ROOT/.t/$RUN_ID"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TRAIN_ARGS=(
  --config-path "$SOURCE_ROOT/artifacts/bicat_box" --config-name ki_molmo2er_fft
  "output_dir=$OUT/train"
  "run_name=BiCAT_box_molmo2ER_4xh200_$RUN_ID"
  "trainer.batch_size=$PER_RANK_BATCH"
  "trainer.global_batch_size=$GLOBAL_BATCH_SIZE"
  "trainer.gradient_accumulation_steps=$GRAD_ACCUM"
  trainer.distributed_strategy=ddp
  trainer.find_unused_parameters=true
  trainer.num_workers=4
  trainer.val_ratio=0.0
  trainer.val_every=0
  trainer.reconst_every=0
  trainer.sim_eval_every=null
  trainer.checkpoint_candidates.best.k=0
  trainer.checkpoint_every=500
  +trainer.checkpoint_start_step=1000
  trainer.checkpoint_candidates.recent.k=9
  policy.gradient_checkpointing=true
  "trainer.max_steps=${MAX_STEPS:-30000}"
)
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  TRAIN_ARGS+=("resume_checkpoint=$RESUME_CHECKPOINT")
fi
echo "GPUs=4 per_rank_batch=$PER_RANK_BATCH accumulation=$GRAD_ACCUM global_batch=$GLOBAL_BATCH_SIZE"
echo "All box episodes used for training; validation/reconstruction evaluation disabled."
"$SOURCE_ROOT/.venv/bin/python" -m praxis.scripts.train --cfg job "${TRAIN_ARGS[@]}" > "$OUT/resolved_config.yaml"
if [[ "$RUN_MODE" == dry-run ]]; then
  echo "Resolved config: $OUT/resolved_config.yaml"
  exit 0
fi
"$SOURCE_ROOT/.venv/bin/python" -c 'import torch; n = torch.cuda.device_count(); assert n >= 4, f"Need 4 visible GPUs, found {n}"'
nvidia-smi
exec "$SOURCE_ROOT/.venv/bin/torchrun" --standalone --nnodes=1 --nproc_per_node=4 \
  -m praxis.scripts.train "${TRAIN_ARGS[@]}"
