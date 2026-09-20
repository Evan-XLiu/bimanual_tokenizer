#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$SOURCE_ROOT"

RUN_ID=${PBS_JOBID:-manual}
DATASET_ROOT=${ROBOTWIN_NEW_DATA_ROOT:-$SOURCE_ROOT/data/lerobot/local/robotwin_new_demo_clean}
OUTPUT_BASE=${OUTPUT_BASE:-$SOURCE_ROOT/outputs/hopper_boat_robotwin_new}
RUN_NAME=${RUN_NAME:-robotwin_new_boat_tokenizer}
OUT="$OUTPUT_BASE/$RUN_ID"
JOB_TMPDIR=${JOB_TMPDIR:-$SOURCE_ROOT/.t/$RUN_ID}

test -f "$DATASET_ROOT/meta/info.json"
test -f "$DATASET_ROOT/meta/stats.json"
mkdir -p "$OUT" "$JOB_TMPDIR"
exec > >(tee "$OUT/train.log") 2>&1

export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
export HF_HOME=${HF_HOME:-$SOURCE_ROOT/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export WANDB_MODE=${WANDB_MODE:-online}
export TMPDIR="$JOB_TMPDIR"
export PYTHONUNBUFFERED=1

TRAIN_ARGS=(
  --config-name train_codecs
  'codecs@codec=tokens/boat'
  dataset=robotwin_new_demo_clean
  dataset.root="$DATASET_ROOT"
  dataset.episodes=null
  dataset.bypass_force_cache_sync=true
  trainer.wandb_project=tokenizer_pretrain
  run_name="$RUN_NAME"
  output_dir="$OUT/train"
)

echo "BOAT_TRAIN_START job=$RUN_ID host=$(hostname) dataset_root=$DATASET_ROOT"
nvidia-smi

"$SOURCE_ROOT/.venv/bin/python" -m praxis.scripts.train \
  "${TRAIN_ARGS[@]}" --cfg job > "$OUT/resolved_config.yaml"

"$SOURCE_ROOT/.venv/bin/python" - "$OUT/resolved_config.yaml" "$DATASET_ROOT" <<'PY'
import sys

from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
expected = {
    "codec.name": "boat",
    "codec.action_horizon": 32,
    "codec.latent_horizon": 16,
    "codec.latent_dropout_mode": "pow2",
    "dataset.root": sys.argv[2],
    "dataset.episodes": None,
    "trainer.max_steps": 300000,
    "trainer.batch_size": 512,
    "trainer.mixed_precision": "bf16",
    "trainer.lr": 5.0e-5,
    "trainer.wandb_project": "tokenizer_pretrain",
}
for key, wanted in expected.items():
    actual = OmegaConf.select(cfg, key)
    if actual != wanted:
        raise SystemExit(
            f"CONFIG_AUDIT_FAILED {key}: expected {wanted!r}, got {actual!r}"
        )
print("CONFIG_AUDIT_OK " + " ".join(f"{key}={value}" for key, value in expected.items()))
PY

"$SOURCE_ROOT/.venv/bin/praxis-train" "${TRAIN_ARGS[@]}"

echo "BOAT_TRAIN_COMPLETE output=$OUT"
