#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$SOURCE_ROOT"

RUN_ID=${PBS_JOBID:-manual-500k}
DATASET_ROOT=${ROBOTWIN_NEW_DATA_ROOT:-$SOURCE_ROOT/data/lerobot/local/robotwin_new_demo_clean}
BIMANUALOAT_CHECKPOINT=${BIMANUALOAT_CHECKPOINT:-$SOURCE_ROOT/outputs/hopper_bimanualoat_robotwin_new_2500/606307.hopper-m-02/train/checkpoints/best/step_300000/pretrained_model}
OUTPUT_BASE=${OUTPUT_BASE:-$SOURCE_ROOT/outputs/hopper_bar_bimanualoat_pali_robotwin_500k}
RUN_NAME=${RUN_NAME:-robotwin_bar_bimanualoat_pali_500k_bs16_lr2.5e-5}
OUT="$OUTPUT_BASE/$RUN_ID"
JOB_TMPDIR=${JOB_TMPDIR:-$SOURCE_ROOT/.t/$RUN_ID}

test -f "$DATASET_ROOT/meta/info.json"
test -f "$DATASET_ROOT/meta/stats.json"
test -f "$BIMANUALOAT_CHECKPOINT/config.json"
test -f "$BIMANUALOAT_CHECKPOINT/model.safetensors"
test -f "$BIMANUALOAT_CHECKPOINT/norm_stats.pt"
mkdir -p "$OUT" "$JOB_TMPDIR"
exec > >(tee "$OUT/train.log") 2>&1

export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
export HF_HOME=${HF_HOME:-$SOURCE_ROOT/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export WANDB_MODE=${WANDB_MODE:-online}
export TMPDIR="$JOB_TMPDIR"

TRAIN_ARGS=(
  policy=bar_bimanualoat_pali
  dataset=libero
  dataset.name=robotwin_new_demo_clean
  dataset.repo_id=local/robotwin_new_demo_clean
  dataset.root="$DATASET_ROOT"
  dataset.episodes=null
  dataset.bypass_force_cache_sync=true
  policy.action_tokenizer.pretrained_path="$BIMANUALOAT_CHECKPOINT"
  policy.freeze_action_tokenizer=true
  policy.use_lora=false
  policy.freeze_vision_encoder=false
  policy.gradient_checkpointing=false
  trainer.batch_size=16
  trainer.global_batch_size=null
  trainer.gradient_accumulation_steps=1
  trainer.max_steps=500000
  trainer.lr_scheduler=cosine
  trainer.warmup_steps=500
  trainer.min_lr_ratio=0.1
  trainer.lrs.vlm=2.5e-5
  trainer.lrs.vision=2.5e-5
  trainer.lrs.bridge=2.5e-5
  trainer.val_every=2000
  trainer.reconst_every=2000
  trainer.checkpoint_every=2000
  trainer.checkpoint_candidates.best.metric_key=val_reconst_loss
  trainer.checkpoint_candidates.best.mode=min
  trainer.checkpoint_candidates.best.k=1
  trainer.checkpoint_candidates.recent.k=0
  run_name="$RUN_NAME"
  output_dir="$OUT/train"
)

echo "TRAIN_START job=$RUN_ID host=$(hostname) dataset_root=$DATASET_ROOT checkpoint=$BIMANUALOAT_CHECKPOINT"
nvidia-smi

"$SOURCE_ROOT/.venv/bin/praxis-train" --cfg job "${TRAIN_ARGS[@]}" > "$OUT/resolved_config.yaml"
"$SOURCE_ROOT/.venv/bin/python" - "$OUT/resolved_config.yaml" "$DATASET_ROOT" "$BIMANUALOAT_CHECKPOINT" <<'PY'
import sys

from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
expected = {
    "policy.name": "bar_bimanualoat_pali",
    "policy.vlm_model_name": "google/paligemma2-3b-mix-224",
    "policy.action_tokenizer.name": "bimanualoat",
    "policy.action_tokenizer.pretrained_path": sys.argv[3],
    "policy.freeze_action_tokenizer": True,
    "policy.use_lora": False,
    "policy.freeze_vision_encoder": False,
    "policy.gradient_checkpointing": False,
    "dataset.root": sys.argv[2],
    "trainer.batch_size": 16,
    "trainer.global_batch_size": None,
    "trainer.gradient_accumulation_steps": 1,
    "trainer.max_steps": 500000,
    "trainer.lr_scheduler": "cosine",
    "trainer.warmup_steps": 500,
    "trainer.min_lr_ratio": 0.1,
    "trainer.val_every": 2000,
    "trainer.reconst_every": 2000,
    "trainer.checkpoint_every": 2000,
    "trainer.checkpoint_candidates.best.metric_key": "val_reconst_loss",
    "trainer.checkpoint_candidates.best.mode": "min",
    "trainer.checkpoint_candidates.best.k": 1,
    "trainer.checkpoint_candidates.recent.k": 0,
}
for key, wanted in expected.items():
    actual = OmegaConf.select(cfg, key)
    if actual != wanted:
        raise SystemExit(
            f"CONFIG_AUDIT_FAILED {key}: expected {wanted!r}, got {actual!r}"
        )
for key in ("vlm", "vision", "bridge"):
    actual = float(OmegaConf.select(cfg, f"trainer.lrs.{key}"))
    if actual != 2.5e-5:
        raise SystemExit(
            f"CONFIG_AUDIT_FAILED trainer.lrs.{key}: expected 2.5e-5, got {actual!r}"
        )
print("CONFIG_AUDIT_OK")
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

"$SOURCE_ROOT/.venv/bin/praxis-train" "${TRAIN_ARGS[@]}"

cleanup
trap - EXIT INT TERM
echo "TRAIN_COMPLETE output=$OUT"
