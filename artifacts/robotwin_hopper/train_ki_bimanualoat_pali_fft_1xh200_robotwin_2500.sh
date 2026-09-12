#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$SOURCE_ROOT"

RUN_ID=${PBS_JOBID:-manual}
DATASET_ROOT=${ROBOTWIN_NEW_DATA_ROOT:-$SOURCE_ROOT/data/lerobot/local/robotwin_new_demo_clean}
BIMANUALOAT_CHECKPOINT=${BIMANUALOAT_CHECKPOINT:-$SOURCE_ROOT/outputs/hopper_bimanualoat_robotwin_new_2500/606307.hopper-m-02/train/checkpoints/best/step_300000/pretrained_model}
OUTPUT_BASE=${OUTPUT_BASE:-$SOURCE_ROOT/outputs/hopper_ki_bimanualoat_pali_fft_1xh200_robotwin_2500}
RUN_NAME=${RUN_NAME:-robotwin_new_2500_ki_bimanualoat_pali_fft_1xh200_batch16_ga2}
WANDB_PROJECT=${WANDB_PROJECT:-praxis_vla}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
TRAIN_GRAD_ACCUM=${TRAIN_GRAD_ACCUM:-2}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
OUT="$OUTPUT_BASE/$RUN_ID"
JOB_TMPDIR=${JOB_TMPDIR:-$SOURCE_ROOT/.t/$RUN_ID}

test -f "$DATASET_ROOT/meta/info.json"
test -f "$DATASET_ROOT/meta/stats.json"
test -f "$BIMANUALOAT_CHECKPOINT/config.json"
test -f "$BIMANUALOAT_CHECKPOINT/model.safetensors"
test -f "$BIMANUALOAT_CHECKPOINT/norm_stats.pt"
test -x "$SOURCE_ROOT/.venv/bin/torchrun"
mkdir -p "$OUT" "$JOB_TMPDIR"
exec > >(tee "$OUT/train.log") 2>&1

export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
export HF_HOME=${HF_HOME:-$SOURCE_ROOT/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export WANDB_MODE=online
export TMPDIR="$JOB_TMPDIR"

TRAIN_ARGS=(
  --config-name train_policy
  policy=ki_bimanualoat_pali
  train_recipe/policy@_global_=ki
  policy.gradient_checkpointing=true
  dataset=libero
  dataset.name=robotwin_new_demo_clean
  dataset.repo_id=local/robotwin_new_demo_clean
  dataset.root="$DATASET_ROOT"
  dataset.episodes=null
  policy.action_tokenizer.pretrained_path="$BIMANUALOAT_CHECKPOINT"
  policy.freeze_action_tokenizer=true
  policy.use_lora=false
  policy.freeze_vision_encoder=false
  device=cuda
  trainer.batch_size="$TRAIN_BATCH_SIZE"
  trainer.gradient_accumulation_steps="$TRAIN_GRAD_ACCUM"
  trainer.num_workers=2
  trainer.ema_decay=0.99
  trainer.log_every=100
  trainer.val_every=500
  trainer.reconst_every=500
  trainer.checkpoint_every=500
  trainer.sim_eval_every=null
  trainer.use_wandb=true
  trainer.wandb_project="$WANDB_PROJECT"
  trainer.wandb_upload_eval_videos=false
  run_name="$RUN_NAME"
  output_dir="$OUT/train"
)

echo "TRAIN_START job=$RUN_ID host=$(hostname) dataset_root=$DATASET_ROOT bimanualoat_checkpoint=$BIMANUALOAT_CHECKPOINT wandb_project=$WANDB_PROJECT tmpdir=$TMPDIR"
nvidia-smi

"$SOURCE_ROOT/.venv/bin/python" -m praxis.scripts.train --cfg job "${TRAIN_ARGS[@]}" > "$OUT/resolved_config.yaml"
"$SOURCE_ROOT/.venv/bin/python" - "$OUT/resolved_config.yaml" <<'PY'
import os
import sys

from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
expected = {
    "policy.name": "ki_bimanualoat_pali",
    "policy.vlm_model_name": "google/paligemma2-3b-mix-224",
    "policy.action_tokenizer.name": "bimanualoat",
    "policy.action_tokenizer.kind": "tokens",
    "policy.freeze_action_tokenizer": True,
    "policy.use_lora": False,
    "policy.freeze_vision_encoder": False,
    "policy.gradient_checkpointing": True,
    "policy.ce_warmstart_steps": 10000,
    "trainer.max_steps": 30000,
    "trainer.batch_size": int(os.environ.get("TRAIN_BATCH_SIZE", "16")),
    "trainer.global_batch_size": None,
    "trainer.gradient_accumulation_steps": int(os.environ.get("TRAIN_GRAD_ACCUM", "2")),
    "trainer.distributed_strategy": "ddp",
    "trainer.num_workers": 2,
    "trainer.use_ema": True,
    "trainer.ema_decay": 0.99,
    "trainer.log_every": 100,
    "trainer.val_every": 500,
    "trainer.reconst_every": 500,
    "trainer.checkpoint_every": 500,
    "trainer.sim_eval_every": None,
    "trainer.use_wandb": True,
    "trainer.wandb_project": os.environ.get("WANDB_PROJECT", "praxis_vla"),
    "trainer.wandb_upload_eval_videos": False,
    "trainer.lr_scheduler": "constant",
    "trainer.warmup_steps": 10000,
}
for key, wanted in expected.items():
    actual = OmegaConf.select(cfg, key)
    if actual != wanted:
        raise SystemExit(
            f"CONFIG_AUDIT_FAILED {key}: expected {wanted!r}, got {actual!r}"
        )
for key in ("vlm", "vision", "bridge", "action_expert"):
    actual = OmegaConf.select(cfg, f"trainer.lrs.{key}")
    if actual != 5.0e-5:
        raise SystemExit(
            f"CONFIG_AUDIT_FAILED trainer.lrs.{key}: expected 5e-5, got {actual!r}"
        )
print("CONFIG_AUDIT_OK " + " ".join(f"{key}={value}" for key, value in expected.items()))
PY

(
  while true; do
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits
    sleep 10
  done
) >"$OUT/gpu_usage.csv" &
GPU_MONITOR_PID=$!
cleanup() {
  kill "$GPU_MONITOR_PID" 2>/dev/null || true
  wait "$GPU_MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

"$SOURCE_ROOT/.venv/bin/torchrun" \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  -m praxis.scripts.train \
  "${TRAIN_ARGS[@]}"

cleanup
trap - EXIT INT TERM
echo "TRAIN_COMPLETE output=$OUT"
