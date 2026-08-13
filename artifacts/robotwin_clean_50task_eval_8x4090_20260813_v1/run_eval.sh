#!/usr/bin/env bash
set -Eeuo pipefail

SUITE_COMMIT=773e7fdc32f96e35dae200f647feb900a0791aaa
RUN_ID=robotwin-ar-bin-qwen3vl2b-h32-final-ema-step10095-demo-clean-50x20-s0-8x4090-20260813-v1
EMA_CHECKPOINT=/mnt/robot-research-exp-0/user/hxs/praxis-vla/outputs/robotwin2_clean/robotwin-bar-bin-qwen3vl2b-h32-10ep-128ppu-gb512-s42-20260812-v1/checkpoints/recent/step_10095/pretrained_model
ROBOTWIN_ROOT=/mnt/robot-research-raw-data-0/user/lqw/code/RoboTwin
SOURCE_ROOT=${PRAXIS_SOURCE_ROOT:?CloudML code checkout path is required}
ENV_ROOT=/mnt/robot-research-exp-0/user/hxs/praxis-vla
POLICY_PYTHON=$ENV_ROOT/.venv/bin/python
ROBOTWIN_PYTHON=$ENV_ROOT/.venv-robotwin/bin/python
OUT=/mnt/robot-research-exp-0/user/hxs/praxis-vla/outputs/robotwin2_clean/$RUN_ID
TASKS=$SOURCE_ROOT/scripts/robotwin/all_tasks.yml
SOURCE_MANIFEST=$OUT/_provenance/robotwin_source_manifest.json

export HOME=/mnt/robot-research-exp-0/user/hxs/home
export TMPDIR=/mnt/robot-research-exp-0/user/hxs/tmp/$RUN_ID
export XDG_CACHE_HOME=/mnt/robot-research-exp-0/user/hxs/cache/xdg/$RUN_ID
export HF_HOME=/mnt/robot-research-exp-0/user/hxs/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export TORCH_HOME=/mnt/robot-research-exp-0/user/hxs/cache/torch/$RUN_ID
export TRITON_CACHE_DIR=/mnt/robot-research-exp-0/user/hxs/cache/triton/$RUN_ID
export TORCHINDUCTOR_CACHE_DIR=/mnt/robot-research-exp-0/user/hxs/cache/torchinductor/$RUN_ID
export PYTHONPYCACHEPREFIX=/mnt/robot-research-exp-0/user/hxs/cache/pycache/$RUN_ID
export PYTHONPATH=$SOURCE_ROOT/src:$SOURCE_ROOT:$ROBOTWIN_ROOT/envs/curobo/src
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export OMP_NUM_THREADS=1

SERVER_PIDS=()
WORKER_PIDS=()
cleanup() {
  local pid
  for pid in "${WORKER_PIDS[@]:-}" "${SERVER_PIDS[@]:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

git -C "$SOURCE_ROOT" merge-base --is-ancestor "$SUITE_COMMIT" HEAD
test -z "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)"
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 8
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c '4090')" -eq 8
test -x "$POLICY_PYTHON"
test -x "$ROBOTWIN_PYTHON"
test -f "$EMA_CHECKPOINT/model.safetensors"
test -f "$EMA_CHECKPOINT/config.json"
test -f "$EMA_CHECKPOINT/policy_io.json"
test -f "$EMA_CHECKPOINT/norm_stats.pt"
test -d "$ROBOTWIN_ROOT/.git"
test "$(git -C "$ROBOTWIN_ROOT" rev-parse HEAD)" = 5fafb0efb7bab9678796642b05a8939ca1068a47
test ! -e "$OUT"

mkdir -p "$HOME" "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" \
  "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$PYTHONPYCACHEPREFIX" \
  "$OUT/_provenance" "$OUT/_logs"

CUDA_VISIBLE_DEVICES=0 "$ROBOTWIN_PYTHON" -c \
  "import sapien; sapien.SapienRenderer(); print('ROBOTWIN_4090_SAPIEN_RENDERER_OK', flush=True)"

"$POLICY_PYTHON" -c \
  "from praxis.serving.model_identity import validate_robotwin_ar_bin_contract; print(validate_robotwin_ar_bin_contract('$EMA_CHECKPOINT'), flush=True)"

cd "$SOURCE_ROOT"
"$ROBOTWIN_PYTHON" scripts/robotwin/run_suite.py prepare-source \
  --robotwin-root "$ROBOTWIN_ROOT" \
  --output "$SOURCE_MANIFEST"
SOURCE_MANIFEST_SHA256=$(tr -d '[:space:]' < "$SOURCE_MANIFEST.sha256")

for gpu in $(seq 0 7); do
  port=$((50051 + gpu))
  CUDA_VISIBLE_DEVICES=$gpu "$POLICY_PYTHON" -m praxis.scripts.serve \
    policy=bar_bin_qwen \
    policy.pretrained_path="$EMA_CHECKPOINT" \
    host=127.0.0.1 \
    port="$port" \
    device=cuda \
    >"$OUT/_logs/server-$gpu.log" 2>&1 &
  SERVER_PIDS+=("$!")
done

"$POLICY_PYTHON" - "$OUT" "$EMA_CHECKPOINT" <<'PY'
import json
import sys
import time
from pathlib import Path

from praxis_remote import PolicyClient

output = Path(sys.argv[1])
expected_checkpoint = Path(sys.argv[2]).resolve()
for gpu in range(8):
    client = PolicyClient(host="127.0.0.1", port=50051 + gpu, timeout=5.0)
    try:
        for _ in range(240):
            try:
                ready, model_info = client.health_check()
                if ready:
                    payload = json.loads(model_info)
                    artifact = Path(
                        payload["serving_artifact"]["checkpoint"]["path"]
                    ).resolve()
                    if artifact != expected_checkpoint:
                        raise RuntimeError(
                            f"Policy server {gpu} loaded {artifact}, expected "
                            f"{expected_checkpoint}."
                        )
                    print(
                        f"ROBOTWIN_POLICY_SERVER_READY gpu={gpu} port={50051 + gpu} "
                        f"checkpoint={artifact}",
                        flush=True,
                    )
                    break
            except Exception:
                time.sleep(5)
        else:
            raise RuntimeError(
                f"Policy server {gpu} did not become ready; see "
                f"{output / '_logs' / f'server-{gpu}.log'}"
            )
    finally:
        client.close()
PY

for worker in $(seq 0 7); do
  port=$((50051 + worker))
  CUDA_VISIBLE_DEVICES=$worker "$ROBOTWIN_PYTHON" scripts/robotwin/run_suite.py worker \
    --robotwin-root "$ROBOTWIN_ROOT" \
    --tasks "$TASKS" \
    --worker-index "$worker" \
    --worker-count 8 \
    --policy-host 127.0.0.1 \
    --policy-port "$port" \
    --output-dir "$OUT" \
    --run-id "$RUN_ID" \
    --expected-checkpoint "$EMA_CHECKPOINT" \
    --robotwin-source-manifest "$SOURCE_MANIFEST" \
    --robotwin-source-manifest-sha256 "$SOURCE_MANIFEST_SHA256" \
    --episodes-per-task 20 \
    --max-seed-attempts 1000 \
    > >(tee "$OUT/_logs/worker-$worker.log" | sed -u "s/^/[worker-$worker] /") 2>&1 &
  WORKER_PIDS+=("$!")
done

worker_failed=0
for index in $(seq 0 7); do
  if ! wait "${WORKER_PIDS[$index]}"; then
    echo "ROBOTWIN_SUITE_WORKER_FAILED worker=$index log=$OUT/_logs/worker-$index.log" >&2
    worker_failed=1
  fi
done
test "$worker_failed" -eq 0

"$ROBOTWIN_PYTHON" scripts/robotwin/run_suite.py aggregate \
  --tasks "$TASKS" \
  --output-dir "$OUT" \
  --run-id "$RUN_ID" \
  --expected-checkpoint "$EMA_CHECKPOINT" \
  --episodes-per-task 20

echo "ROBOTWIN_CLEAN_50X20_FINAL_EMA_EVAL_COMPLETE output_dir=$OUT"
