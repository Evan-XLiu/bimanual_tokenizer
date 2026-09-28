#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$SOURCE_ROOT"

: "${POLICY_NAME:?Set POLICY_NAME (for example, bar_btt_pali)}"
: "${TOKENIZER_NAME:?Set TOKENIZER_NAME (for example, btt)}"
: "${TOKENIZER_CHECKPOINT:?Set TOKENIZER_CHECKPOINT to a tokenizer checkpoint}"
: "${OUTPUT_BASE:?Set OUTPUT_BASE to the experiment output directory}"
: "${RUN_NAME:?Set RUN_NAME to the W&B run name}"

RUN_ID=${PBS_JOBID:-manual-${TOKENIZER_NAME}-step300000-30k}
DATASET_ROOT=${ROBOTWIN_NEW_DATA_ROOT:-$SOURCE_ROOT/data/lerobot/local/robotwin_new_demo_clean}
REFERENCE_CONFIG=${REFERENCE_CONFIG:-$SOURCE_ROOT/outputs/hopper_bar_btt_pali_robotwin_new_30k/622038.hopper-m-02/resolved_config.yaml}
OUT="$OUTPUT_BASE/$RUN_ID"
JOB_TMPDIR=${JOB_TMPDIR:-$SOURCE_ROOT/.t/$RUN_ID}

test -f "$DATASET_ROOT/meta/info.json"
test -f "$DATASET_ROOT/meta/stats.json"
test -f "$REFERENCE_CONFIG"

# Accept either the checkpoint directory or its pretrained_model child.
if [[ -f "$TOKENIZER_CHECKPOINT/config.json" ]]; then
  TOKENIZER_PATH="$TOKENIZER_CHECKPOINT"
else
  TOKENIZER_PATH="$TOKENIZER_CHECKPOINT/pretrained_model"
fi
test -f "$TOKENIZER_PATH/config.json"
test -f "$TOKENIZER_PATH/model.safetensors"
test -f "$TOKENIZER_PATH/norm_stats.pt"

mkdir -p "$OUT" "$JOB_TMPDIR"
exec > >(tee "$OUT/train.log") 2>&1

export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
export HF_HOME=${HF_HOME:-$SOURCE_ROOT/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export WANDB_MODE=${WANDB_MODE:-online}
export TMPDIR="$JOB_TMPDIR"
export PYTHONUNBUFFERED=1

TRAIN_ARGS=(
  policy="$POLICY_NAME"
  dataset=robotwin_new_demo_clean
  dataset.root="$DATASET_ROOT"
  dataset.episodes=null
  dataset.bypass_force_cache_sync=true
  policy.action_tokenizer.pretrained_path="$TOKENIZER_PATH"
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

echo "TRAIN_START job=$RUN_ID host=$(hostname) policy=$POLICY_NAME tokenizer=$TOKENIZER_NAME tokenizer_checkpoint=$TOKENIZER_PATH dataset_root=$DATASET_ROOT reference_config=$REFERENCE_CONFIG output=$OUT wandb_project=praxis-vla"
nvidia-smi

"$SOURCE_ROOT/.venv/bin/praxis-train" --cfg job "${TRAIN_ARGS[@]}" > "$OUT/resolved_config.yaml"

"$SOURCE_ROOT/.venv/bin/python" - "$OUT/resolved_config.yaml" "$REFERENCE_CONFIG" "$DATASET_ROOT" "$TOKENIZER_PATH" "$POLICY_NAME" "$TOKENIZER_NAME" <<'PY'
import sys

from omegaconf import OmegaConf

candidate = OmegaConf.to_container(OmegaConf.load(sys.argv[1]), resolve=False)
reference = OmegaConf.to_container(OmegaConf.load(sys.argv[2]), resolve=False)
assert isinstance(candidate, dict) and isinstance(reference, dict)

expected = {
    "policy.name": sys.argv[5],
    "policy.vlm_model_name": "google/paligemma2-3b-mix-224",
    "policy.action_tokenizer.name": sys.argv[6],
    "policy.action_tokenizer.kind": "tokens",
    "policy.action_tokenizer.pretrained_path": sys.argv[4],
    "policy.use_lora": False,
    "policy.freeze_vision_encoder": False,
    "policy.freeze_action_tokenizer": True,
    "policy.gradient_checkpointing": True,
    "dataset.name": "robotwin_new_demo_clean",
    "dataset.root": sys.argv[3],
    "dataset.episodes": None,
    "trainer.max_steps": 30000,
    "trainer.batch_size": 64,
    "trainer.weight_decay": 1.0e-6,
    "trainer.lr_scheduler": "cosine",
    "trainer.warmup_steps": 500,
    "trainer.mixed_precision": "bf16",
    "trainer.val_every": 1000,
    "trainer.reconst_every": 1000,
    "trainer.checkpoint_every": 1000,
    "trainer.checkpoint_candidates.best.k": 2,
    "trainer.checkpoint_candidates.recent.k": 0,
    "trainer.num_workers": 4,
    "trainer.wandb_project": "praxis-vla",
}
candidate_cfg = OmegaConf.create(candidate)
for key, wanted in expected.items():
    actual = OmegaConf.select(candidate_cfg, key)
    if actual != wanted:
        raise SystemExit(
            f"CONFIG_AUDIT_FAILED {key}: expected {wanted!r}, got {actual!r}"
        )
for key in ("vlm", "vision", "bridge"):
    actual = float(OmegaConf.select(candidate_cfg, f"trainer.lrs.{key}"))
    if actual != 1.0e-4:
        raise SystemExit(
            f"CONFIG_AUDIT_FAILED trainer.lrs.{key}: expected 1e-4, got {actual!r}"
        )

# Require exact parity with the completed BAR+BTT+Pali reference, except for
# tokenizer/policy identity, tokenizer checkpoint, and run output identity.
for payload in (candidate, reference):
    policy = payload["policy"]
    policy.pop("decode_keep_k", None)
    policy["name"] = "<policy-name>"
    policy["action_tokenizer"]["name"] = "<tokenizer-name>"
    policy["action_tokenizer"]["pretrained_path"] = "<tokenizer-checkpoint>"
    payload["run_name"] = "<run-name>"
    payload["output_dir"] = "<output-dir>"
if candidate != reference:
    raise SystemExit("CONFIG_AUDIT_FAILED candidate differs from BAR+BTT reference")
print("CONFIG_AUDIT_OK exact BAR+BTT hyperparameter parity")
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
