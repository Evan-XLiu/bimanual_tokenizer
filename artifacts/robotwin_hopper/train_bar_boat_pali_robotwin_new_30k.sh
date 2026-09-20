#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$SOURCE_ROOT"

RUN_ID=${PBS_JOBID:-manual-bar-boat-pali-30k}
DATASET_ROOT=${ROBOTWIN_NEW_DATA_ROOT:-$SOURCE_ROOT/data/lerobot/local/robotwin_new_demo_clean}
BOAT_CHECKPOINT=${BOAT_CHECKPOINT:-$SOURCE_ROOT/outputs/hopper_boat_robotwin_new/616167.hopper-m-02/train/checkpoints/best/step_300000}
OUTPUT_BASE=${OUTPUT_BASE:-$SOURCE_ROOT/outputs/hopper_bar_boat_pali_robotwin_new_30k}
RUN_NAME=${RUN_NAME:-robotwin_new_bar_boat_pali_30k}
OUT="$OUTPUT_BASE/$RUN_ID"
JOB_TMPDIR=${JOB_TMPDIR:-$SOURCE_ROOT/.t/$RUN_ID}

test -f "$DATASET_ROOT/meta/info.json"
test -f "$DATASET_ROOT/meta/stats.json"

# Accept either the checkpoint directory supplied by the user or its pretrained_model child.
if [[ -f "$BOAT_CHECKPOINT/config.json" ]]; then
  BOAT_TOKENIZER_PATH="$BOAT_CHECKPOINT"
else
  BOAT_TOKENIZER_PATH="$BOAT_CHECKPOINT/pretrained_model"
fi
test -f "$BOAT_TOKENIZER_PATH/config.json"
test -f "$BOAT_TOKENIZER_PATH/model.safetensors"
test -f "$BOAT_TOKENIZER_PATH/norm_stats.pt"

mkdir -p "$OUT" "$JOB_TMPDIR"
exec > >(tee "$OUT/train.log") 2>&1

export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
export HF_HOME=${HF_HOME:-$SOURCE_ROOT/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export WANDB_MODE=${WANDB_MODE:-online}
export TMPDIR="$JOB_TMPDIR"
export PYTHONUNBUFFERED=1

TRAIN_ARGS=(
  policy=bar_boat_pali
  dataset=robotwin_new_demo_clean
  dataset.root="$DATASET_ROOT"
  dataset.episodes=null
  dataset.bypass_force_cache_sync=true
  policy.action_tokenizer.pretrained_path="$BOAT_TOKENIZER_PATH"
  policy.gradient_checkpointing=true
  trainer.max_steps=30000
  trainer.val_every=1000
  trainer.reconst_every=1000
  trainer.checkpoint_every=1000
  trainer.checkpoint_candidates.best.k=2
  trainer.checkpoint_candidates.recent.k=0
  trainer.num_workers=4
  trainer.wandb_project=praxis-vla
  run_name="$RUN_NAME"
  output_dir="$OUT/train"
)

echo "TRAIN_START job=$RUN_ID host=$(hostname) dataset_root=$DATASET_ROOT boat_checkpoint=$BOAT_TOKENIZER_PATH output=$OUT"
nvidia-smi

"$SOURCE_ROOT/.venv/bin/praxis-train" --cfg job "${TRAIN_ARGS[@]}" > "$OUT/resolved_config.yaml"

"$SOURCE_ROOT/.venv/bin/python" - "$OUT/resolved_config.yaml" "$DATASET_ROOT" "$BOAT_TOKENIZER_PATH" <<'PY'
import sys

from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
expected = {
    "policy.name": "bar_boat_pali",
    "policy.vlm_model_name": "google/paligemma2-3b-mix-224",
    "policy.action_tokenizer.name": "boat",
    "policy.action_tokenizer.kind": "tokens",
    "policy.action_tokenizer.pretrained_path": sys.argv[3],
    "policy.use_lora": False,
    "policy.freeze_vision_encoder": False,
    "policy.freeze_action_tokenizer": True,
    "policy.gradient_checkpointing": True,
    "dataset.name": "robotwin_new_demo_clean",
    "dataset.root": sys.argv[2],
    "dataset.episodes": None,
    "trainer.max_steps": 30000,
    "trainer.val_every": 1000,
    "trainer.reconst_every": 1000,
    "trainer.checkpoint_every": 1000,
    "trainer.checkpoint_candidates.best.k": 2,
    "trainer.checkpoint_candidates.recent.k": 0,
    "trainer.num_workers": 4,
    "trainer.wandb_project": "praxis-vla",
}
for key, wanted in expected.items():
    actual = OmegaConf.select(cfg, key)
    if actual != wanted:
        raise SystemExit(
            f"CONFIG_AUDIT_FAILED {key}: expected {wanted!r}, got {actual!r}"
        )
for key in ("vlm", "vision", "bridge"):
    actual = float(OmegaConf.select(cfg, f"trainer.lrs.{key}"))
    if actual != 1.0e-4:
        raise SystemExit(
            f"CONFIG_AUDIT_FAILED trainer.lrs.{key}: expected 1e-4, got {actual!r}"
        )
print("CONFIG_AUDIT_OK")
PY

(
  while true; do
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits
    sleep 10
  done
) > "$OUT/gpu_usage.csv" &
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
