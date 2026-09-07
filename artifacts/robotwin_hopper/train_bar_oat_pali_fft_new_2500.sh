#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$SOURCE_ROOT"

RUN_ID=${PBS_JOBID:-manual}
DATASET_ROOT=${ROBOTWIN_NEW_DATA_ROOT:-$SOURCE_ROOT/data/lerobot/local/robotwin_new_demo_clean}
OAT_CHECKPOINT=${OAT_CHECKPOINT:-$SOURCE_ROOT/outputs/hopper_oat_robotwin_new_2500/600693.hopper-m-02/train/checkpoints/best/step_296000/pretrained_model}
OUTPUT_BASE=${OUTPUT_BASE:-$SOURCE_ROOT/outputs/hopper_bar_oat_pali_fft_new_2500}
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
  policy=bar_oat_pali
  dataset=libero
  dataset.name=robotwin_new_demo_clean
  dataset.repo_id=local/robotwin_new_demo_clean
  dataset.root="$DATASET_ROOT"
  dataset.episodes=null
  dataset.bypass_force_cache_sync=true
  policy.action_tokenizer.pretrained_path="$OAT_CHECKPOINT"
  policy.freeze_action_tokenizer=true
  policy.use_lora=false
  policy.freeze_vision_encoder=false
  policy.gradient_checkpointing=true
  trainer.batch_size=64
  trainer.gradient_accumulation_steps=1
  trainer.num_workers=2
  trainer.max_steps=20000
  trainer.log_every=1
  trainer.val_every=200
  trainer.reconst_every=200
  trainer.checkpoint_every=5000
  trainer.checkpoint_candidates.best.k=3
  trainer.checkpoint_candidates.recent.k=4
  trainer.use_wandb=true
  trainer.wandb_project=praxis-vla
  run_name=robotwin_new_2500_bar_oat_pali_fft_unfrozen_vision_20k
  output_dir="$OUT/train"
)

echo "TRAIN_START job=$RUN_ID host=$(hostname) dataset_root=$DATASET_ROOT oat_checkpoint=$OAT_CHECKPOINT tmpdir=$TMPDIR"
nvidia-smi

"$SOURCE_ROOT/.venv/bin/praxis-train" --cfg job "${TRAIN_ARGS[@]}" > "$OUT/resolved_config.yaml"
"$SOURCE_ROOT/.venv/bin/python" - "$OUT/resolved_config.yaml" <<'PY'
import sys

from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
expected = {
    "policy.name": "bar_oat_pali",
    "policy.freeze_action_tokenizer": True,
    "policy.use_lora": False,
    "policy.freeze_vision_encoder": False,
    "trainer.max_steps": 20000,
    "trainer.log_every": 1,
    "trainer.val_every": 200,
    "trainer.reconst_every": 200,
    "trainer.checkpoint_every": 5000,
    "trainer.checkpoint_candidates.best.k": 3,
    "trainer.checkpoint_candidates.recent.k": 4,
    "trainer.wandb_project": "praxis-vla",
}
for key, wanted in expected.items():
    actual = OmegaConf.select(cfg, key)
    if actual != wanted:
        raise SystemExit(f"CONFIG_AUDIT_FAILED {key}: expected {wanted!r}, got {actual!r}")
if cfg.dataset.episodes is not None:
    raise SystemExit(f"CONFIG_AUDIT_FAILED dataset.episodes: expected null, got {cfg.dataset.episodes!r}")
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
