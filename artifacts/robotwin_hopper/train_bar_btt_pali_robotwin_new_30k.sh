#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$SOURCE_ROOT"

RUN_ID=${PBS_JOBID:-manual-bar-btt-pali-30k}
DATASET_ROOT=${ROBOTWIN_NEW_DATA_ROOT:-$SOURCE_ROOT/data/lerobot/local/robotwin_new_demo_clean}
BTT_CHECKPOINT=${BTT_CHECKPOINT:-$SOURCE_ROOT/outputs/hopper_btt_robotwin_new/621340.hopper-m-02/train/checkpoints/best/step_297000}
REFERENCE_CONFIG=${REFERENCE_CONFIG:-$SOURCE_ROOT/outputs/hopper_bar_boat_pali_robotwin_new_30k/616455.hopper-m-02/resolved_config.yaml}
OUTPUT_BASE=${OUTPUT_BASE:-$SOURCE_ROOT/outputs/hopper_bar_btt_pali_robotwin_new_30k}
RUN_NAME=${RUN_NAME:-robotwin_new_bar_btt_pali_30k}
OUT="$OUTPUT_BASE/$RUN_ID"
JOB_TMPDIR=${JOB_TMPDIR:-$SOURCE_ROOT/.t/$RUN_ID}

test -f "$DATASET_ROOT/meta/info.json"
test -f "$DATASET_ROOT/meta/stats.json"
test -f "$REFERENCE_CONFIG"

# Accept either the checkpoint directory or its pretrained_model child.
if [[ -f "$BTT_CHECKPOINT/config.json" ]]; then
  BTT_TOKENIZER_PATH="$BTT_CHECKPOINT"
else
  BTT_TOKENIZER_PATH="$BTT_CHECKPOINT/pretrained_model"
fi
test -f "$BTT_TOKENIZER_PATH/config.json"
test -f "$BTT_TOKENIZER_PATH/model.safetensors"
test -f "$BTT_TOKENIZER_PATH/norm_stats.pt"

mkdir -p "$OUT" "$JOB_TMPDIR"
exec > >(tee "$OUT/train.log") 2>&1

export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
export HF_HOME=${HF_HOME:-$SOURCE_ROOT/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export WANDB_MODE=${WANDB_MODE:-online}
export TMPDIR="$JOB_TMPDIR"
export PYTHONUNBUFFERED=1

TRAIN_ARGS=(
  policy=bar_btt_pali
  dataset=robotwin_new_demo_clean
  dataset.root="$DATASET_ROOT"
  dataset.episodes=null
  dataset.bypass_force_cache_sync=true
  policy.action_tokenizer.pretrained_path="$BTT_TOKENIZER_PATH"
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

echo "TRAIN_START job=$RUN_ID host=$(hostname) dataset_root=$DATASET_ROOT btt_checkpoint=$BTT_TOKENIZER_PATH reference_config=$REFERENCE_CONFIG output=$OUT"
nvidia-smi

"$SOURCE_ROOT/.venv/bin/praxis-train" --cfg job "${TRAIN_ARGS[@]}" > "$OUT/resolved_config.yaml"

"$SOURCE_ROOT/.venv/bin/python" - "$OUT/resolved_config.yaml" "$REFERENCE_CONFIG" "$DATASET_ROOT" "$BTT_TOKENIZER_PATH" <<'PY'
import sys

from omegaconf import OmegaConf

candidate = OmegaConf.to_container(OmegaConf.load(sys.argv[1]), resolve=False)
reference = OmegaConf.to_container(OmegaConf.load(sys.argv[2]), resolve=False)
assert isinstance(candidate, dict) and isinstance(reference, dict)

expected = {
    "policy.name": "bar_btt_pali",
    "policy.vlm_model_name": "google/paligemma2-3b-mix-224",
    "policy.action_tokenizer.name": "btt",
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

# Require exact parity with the completed BAR+BOAT+Pali run except for the
# policy/tokenizer identity, tokenizer checkpoint, and run output identity.
for payload in (candidate, reference):
    policy = payload["policy"]
    policy.pop("decode_keep_k", None)
    policy["name"] = "<policy-name>"
    policy["action_tokenizer"]["name"] = "<tokenizer-name>"
    policy["action_tokenizer"]["pretrained_path"] = "<tokenizer-checkpoint>"
    payload["run_name"] = "<run-name>"
    payload["output_dir"] = "<output-dir>"
if candidate != reference:
    raise SystemExit("CONFIG_AUDIT_FAILED candidate differs from BAR+BOAT reference")
print("CONFIG_AUDIT_OK exact BAR+BOAT hyperparameter parity")
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
