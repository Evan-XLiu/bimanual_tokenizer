#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$SOURCE_ROOT"

RUN_ID=${PBS_JOBID:-manual-bar-quest-pali-30k}
DATASET_ROOT=${ROBOTWIN_NEW_DATA_ROOT:-$SOURCE_ROOT/data/lerobot/local/robotwin_new_demo_clean}
QUEST_CHECKPOINT=${QUEST_CHECKPOINT:-$SOURCE_ROOT/outputs/hopper_quest_robotwin_new/613294.hopper-m-02/train/checkpoints/best/step_299000/pretrained_model}
OUTPUT_BASE=${OUTPUT_BASE:-$SOURCE_ROOT/outputs/hopper_bar_quest_pali_robotwin_new_30k}
RUN_NAME=${RUN_NAME:-robotwin_new_bar_quest_pali_30k}
OUT="$OUTPUT_BASE/$RUN_ID"
JOB_TMPDIR=${JOB_TMPDIR:-$SOURCE_ROOT/.t/$RUN_ID}

test -f "$DATASET_ROOT/meta/info.json"
test -f "$DATASET_ROOT/meta/stats.json"
test -f "$QUEST_CHECKPOINT/config.json"
test -f "$QUEST_CHECKPOINT/model.safetensors"
test -f "$QUEST_CHECKPOINT/norm_stats.pt"

mkdir -p "$OUT" "$JOB_TMPDIR"
exec > >(tee "$OUT/train.log") 2>&1

export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
export HF_HOME=${HF_HOME:-$SOURCE_ROOT/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export WANDB_MODE=${WANDB_MODE:-online}
export TMPDIR="$JOB_TMPDIR"
export PYTHONUNBUFFERED=1

TRAIN_ARGS=(
  policy=bar_quest_pali
  dataset=robotwin_new_demo_clean
  dataset.root="$DATASET_ROOT"
  dataset.episodes=null
  dataset.bypass_force_cache_sync=true
  policy.action_tokenizer.pretrained_path="$QUEST_CHECKPOINT"
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

echo "TRAIN_START job=$RUN_ID host=$(hostname) dataset_root=$DATASET_ROOT quest_checkpoint=$QUEST_CHECKPOINT output=$OUT"
nvidia-smi

"$SOURCE_ROOT/.venv/bin/praxis-train" --cfg job "${TRAIN_ARGS[@]}" > "$OUT/resolved_config.yaml"

"$SOURCE_ROOT/.venv/bin/python" - "$OUT/resolved_config.yaml" "$DATASET_ROOT" "$QUEST_CHECKPOINT" <<'PY'
import sys

from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
expected = {
    "policy.name": "bar_quest_pali",
    "policy.action_tokenizer.name": "quest",
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
