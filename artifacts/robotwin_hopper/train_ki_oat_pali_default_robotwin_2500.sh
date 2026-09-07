#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$SOURCE_ROOT"

RUN_ID=${PBS_JOBID:-manual}
DATASET_ROOT=${ROBOTWIN_NEW_DATA_ROOT:-$SOURCE_ROOT/data/lerobot/local/robotwin_new_demo_clean}
OAT_CHECKPOINT=${OAT_CHECKPOINT:-$SOURCE_ROOT/outputs/hopper_oat_robotwin_new_2500/600693.hopper-m-02/train/checkpoints/best/step_296000/pretrained_model}
OUTPUT_BASE=${OUTPUT_BASE:-$SOURCE_ROOT/outputs/hopper_ki_oat_pali_default_robotwin_2500}
RUN_NAME=${RUN_NAME:-robotwin_new_2500_ki_oat_pali_default}
OUT="$OUTPUT_BASE/$RUN_ID"
JOB_TMPDIR=${JOB_TMPDIR:-$SOURCE_ROOT/.t/$RUN_ID}

test -f "$DATASET_ROOT/meta/info.json"
test -f "$DATASET_ROOT/meta/stats.json"
test -f "$OAT_CHECKPOINT/config.json"
test -f "$OAT_CHECKPOINT/model.safetensors"
test -f "$OAT_CHECKPOINT/norm_stats.pt"
mkdir -p "$OUT" "$JOB_TMPDIR"
exec > >(tee "$OUT/train.log") 2>&1

export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
export HF_HOME=${HF_HOME:-$SOURCE_ROOT/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export WANDB_MODE=online
export TMPDIR="$JOB_TMPDIR"

TRAIN_ARGS=(
  policy=ki_oat_pali
  train_recipe/policy@_global_=ki
  dataset=libero
  dataset.name=robotwin_new_demo_clean
  dataset.repo_id=local/robotwin_new_demo_clean
  dataset.root="$DATASET_ROOT"
  dataset.episodes=null
  policy.action_tokenizer.pretrained_path="$OAT_CHECKPOINT"
  run_name="$RUN_NAME"
  output_dir="$OUT/train"
)
if [[ -n ${TRAIN_BATCH_SIZE:-} ]]; then
  TRAIN_ARGS+=(trainer.batch_size="$TRAIN_BATCH_SIZE")
fi

echo "TRAIN_START job=$RUN_ID host=$(hostname) dataset_root=$DATASET_ROOT oat_checkpoint=$OAT_CHECKPOINT tmpdir=$TMPDIR"
nvidia-smi

"$SOURCE_ROOT/.venv/bin/praxis-train" --cfg job "${TRAIN_ARGS[@]}" > "$OUT/resolved_config.yaml"
"$SOURCE_ROOT/.venv/bin/python" - "$OUT/resolved_config.yaml" <<'PY'
import sys

import os

from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
expected = {
    "policy.name": "ki_oat_pali",
    "policy.vlm_model_name": "google/paligemma2-3b-mix-224",
    "policy.freeze_vision_encoder": False,
    "policy.use_lora": False,
    "policy.gradient_checkpointing": False,
    "policy.ce_loss_weight": 1.0,
    "policy.flow_loss_weight": 1.0,
    "policy.flow_sampling_steps": 10,
    "policy.freeze_action_tokenizer": True,
    "policy.ce_vocab_scope": "action",
    "policy.ce_warmstart_steps": 10000,
    "trainer.max_steps": 30000,
    "trainer.batch_size": int(os.environ.get("TRAIN_BATCH_SIZE", "64")),
    "trainer.num_workers": 4,
    "trainer.gradient_accumulation_steps": 1,
    "trainer.mixed_precision": "bf16",
    "trainer.lr_scheduler": "constant",
    "trainer.warmup_steps": 10000,
    "trainer.find_unused_parameters": True,
    "trainer.log_every": 200,
    "trainer.val_every": 1000,
    "trainer.reconst_every": 1000,
    "trainer.checkpoint_every": 5000,
    "trainer.max_val_batches": 100,
    "trainer.max_reconst_batches": 100,
    "trainer.sim_eval_every": None,
    "trainer.checkpoint_candidates.best.k": 3,
    "trainer.checkpoint_candidates.recent.k": 2,
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

"$SOURCE_ROOT/.venv/bin/praxis-train" "${TRAIN_ARGS[@]}"

cleanup
trap - EXIT INT TERM
echo "TRAIN_COMPLETE output=$OUT"
