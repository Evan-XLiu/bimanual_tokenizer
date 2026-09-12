#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$SOURCE_ROOT"

RUN_ID=${PBS_JOBID:-manual}
DATASET_ROOT=${ROBOTWIN_NEW_DATA_ROOT:-$SOURCE_ROOT/data/lerobot/local/robotwin_new_demo_clean}
OUTPUT_BASE=${OUTPUT_BASE:-$SOURCE_ROOT/outputs/hopper_bimanualoat_robotwin_new_2500}
RUN_NAME=${RUN_NAME:-robotwin_new_2500_bimanualoat_tokenizer}
OUT="$OUTPUT_BASE/$RUN_ID"
JOB_TMPDIR=${JOB_TMPDIR:-$SOURCE_ROOT/.t/$RUN_ID}

test -f "$DATASET_ROOT/meta/info.json"
test -f "$DATASET_ROOT/meta/stats.json"
mkdir -p "$OUT" "$JOB_TMPDIR"
exec > >(tee "$OUT/train.log") 2>&1

export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
export HF_HOME=${HF_HOME:-$SOURCE_ROOT/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export WANDB_MODE=online
export TMPDIR="$JOB_TMPDIR"

TRAIN_ARGS=(
  --config-name train_codecs
  'codecs@codec=tokens/bimanualoat'
  dataset=libero
  dataset.name=robotwin_new_demo_clean
  dataset.repo_id=local/robotwin_new_demo_clean
  dataset.root="$DATASET_ROOT"
  dataset.bypass_force_cache_sync=true
  trainer.wandb_project=tokenizer_pretrain
  run_name="$RUN_NAME"
  output_dir="$OUT/train"
)

echo "BIMANUALOAT_TRAIN_START job=$RUN_ID host=$(hostname) dataset_root=$DATASET_ROOT"
nvidia-smi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
if [[ "$GPU_NAME" != *H100* && "$GPU_NAME" != *H200* ]]; then
  echo "Expected an NVIDIA H100 or H200, got: $GPU_NAME" >&2
  exit 1
fi

"$SOURCE_ROOT/.venv/bin/python" - "$DATASET_ROOT" <<'PY'
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

root = Path(sys.argv[1])
info = json.loads((root / "meta/info.json").read_text())
action_feature = info["features"]["action"]
if action_feature["shape"] != [14]:
    raise SystemExit(f"Expected action shape [14], got {action_feature['shape']!r}")
data_path = next((root / "data").glob("chunk-*/file-*.parquet"))
actions = pq.read_table(data_path, columns=["action"]).slice(0, 32)["action"]
if len(actions) != 32 or actions.type.list_size != 14:
    raise SystemExit(
        f"Expected a readable [32, 14] action sample, got rows={len(actions)}, "
        f"type={actions.type}"
    )
print("DATA_AUDIT_OK action_layout=[left_0:7,right_7:14] sample_shape=[32,14]")
PY

"$SOURCE_ROOT/.venv/bin/python" -m praxis.scripts.train \
  "${TRAIN_ARGS[@]}" --cfg job > "$OUT/resolved_config.yaml"
"$SOURCE_ROOT/.venv/bin/python" - "$OUT/resolved_config.yaml" "$DATASET_ROOT" <<'PY'
import sys

from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
expected = {
    "codec.name": "bimanualoat",
    "codec.action_horizon": 32,
    "codec.latent_horizon": 16,
    "codec.emb_dim": 256,
    "codec.common_quantizer_levels": [8, 6],
    "codec.relative_quantizer_levels": [8, 6],
    "dataset.root": sys.argv[2],
    "trainer.max_steps": 300000,
    "trainer.batch_size": 512,
    "trainer.mixed_precision": "bf16",
    "trainer.gradient_accumulation_steps": 1,
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
echo "BIMANUALOAT_TRAIN_COMPLETE output=$OUT"
