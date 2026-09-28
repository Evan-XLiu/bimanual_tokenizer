#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$SOURCE_ROOT"

CHECK_ONLY=false
if [[ ${1:-} == --check-only ]]; then
  CHECK_ONLY=true
  OUT=${2:?Provide a preflight output directory}
  [[ $# -eq 2 ]]
else
  [[ $# -eq 0 ]]
  : "${PBS_JOBID:?Submit the dedicated w/o convention PBS job}"
  OUT="$SOURCE_ROOT/outputs/hopper_bar_btt_wo_convention_pali_robotwin_new_step300000_30k/$PBS_JOBID"
fi
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
exec > >(tee -a "$OUT/train.log") 2>&1

export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
export HF_HOME=${HF_HOME:-$SOURCE_ROOT/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export WANDB_MODE=online
unset WANDB_RUN_ID WANDB_RESUME
export PYTHONUNBUFFERED=1

AUDITOR="$SOURCE_ROOT/artifacts/robotwin_hopper/prepare_bar_btt_wo_convention_pali_30k.py"
PYTHON="$SOURCE_ROOT/.venv/bin/python"
TRAIN="$SOURCE_ROOT/.venv/bin/praxis-train"

"$PYTHON" "$AUDITOR" --output "$OUT"
TRAIN_ARGS=(--config-path "$OUT" --config-name resolved_config)
"$TRAIN" "${TRAIN_ARGS[@]}" --cfg job > "$OUT/hydra_config_audit.yaml"
"$PYTHON" "$AUDITOR" --output "$OUT" --candidate "$OUT/hydra_config_audit.yaml"

if $CHECK_ONLY; then
  echo "PREFLIGHT_OK output=$OUT"
  exit 0
fi

JOB_TMPDIR="$SOURCE_ROOT/.t/$PBS_JOBID"
mkdir -p "$JOB_TMPDIR"
export TMPDIR="$JOB_TMPDIR"
cp "$AUDITOR" "$OUT/prepare_ablation_snapshot.py"
cp "${BASH_SOURCE[0]}" "$OUT/launch_snapshot.sh"
cp artifacts/robotwin_hopper/train_bar_btt_wo_convention_pali_robotwin_new_step300000_30k.pbs "$OUT/submitted_job.pbs"
git rev-parse HEAD > "$OUT/git_commit.txt"
git diff HEAD -- src pyproject.toml uv.lock > "$OUT/source_changes.patch"
tar -czf "$OUT/training_source_snapshot.tar.gz" --exclude=__pycache__ src/praxis pyproject.toml uv.lock

echo "TRAIN_START job=$PBS_JOBID host=$(hostname) policy=bar_btt_pali ablation=wo_convention tokenizer_step=300000 max_steps=30000 wandb_project=praxis-vla output=$OUT"
nvidia-smi
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

"$TRAIN" "${TRAIN_ARGS[@]}"
"$PYTHON" "$AUDITOR" --output "$OUT" --candidate "$OUT/train/train_config.yaml"
cleanup
trap - EXIT INT TERM
echo "TRAIN_COMPLETE output=$OUT"
