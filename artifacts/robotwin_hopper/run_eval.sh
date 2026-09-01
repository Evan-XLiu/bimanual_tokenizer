#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
POLICY_PYTHON=${POLICY_PYTHON:-$SOURCE_ROOT/.venv/bin/python}
POLICY_SERVER=${POLICY_SERVER:-$SOURCE_ROOT/.venv/bin/praxis-serve}
TASKS=$SOURCE_ROOT/scripts/robotwin/all_tasks.yml

: "${ROBOTWIN_ROOT:?Set ROBOTWIN_ROOT to the official RoboTwin checkout}"
: "${ROBOTWIN_PYTHON:?Set ROBOTWIN_PYTHON to its Python executable}"
: "${PRAXIS_CHECKPOINT:?Set PRAXIS_CHECKPOINT to an exported pretrained_model directory}"

RUN_ID=${ROBOTWIN_RUN_ID:-robotwin-hopper-${PBS_JOBID:-manual}}
OUT=${ROBOTWIN_OUTPUT_DIR:-$SOURCE_ROOT/outputs/robotwin2_clean/$RUN_ID}
POLICY_PORT=${POLICY_PORT:-50100}
EPISODES_PER_TASK=${EPISODES_PER_TASK:-20}
MAX_SEED_ATTEMPTS=${MAX_SEED_ATTEMPTS:-1000}
ROBOTWIN_TASK=${ROBOTWIN_TASK:-}

test -x "$POLICY_PYTHON"
test -x "$POLICY_SERVER"
test -x "$ROBOTWIN_PYTHON"
if [ ! -f "$ROBOTWIN_ROOT/script/eval_policy.py" ] && \
   [ ! -f "$ROBOTWIN_ROOT/scripts/eval_policy_xpolicylab.py" ]; then
  echo "No supported RoboTwin evaluator found under $ROBOTWIN_ROOT" >&2
  exit 1
fi
test -f "$PRAXIS_CHECKPOINT/config.json"
test -f "$PRAXIS_CHECKPOINT/model.safetensors"
test -f "$TASKS"
"$ROBOTWIN_PYTHON" -c "import praxis_remote"

POLICY_NAME=$(
  "$POLICY_PYTHON" - "$PRAXIS_CHECKPOINT/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)
name = config.get("name")
if not isinstance(name, str) or not name:
    raise SystemExit("Checkpoint config.json has no policy name")
print(name)
PY
)

mkdir -p "$OUT/_logs"
cd "$SOURCE_ROOT"

if [ "${ROBOTWIN_ENABLE_POLICY_TRACE:-0}" = "1" ]; then
  export ROBOTWIN_POLICY_TRACE_PATH="${ROBOTWIN_POLICY_TRACE_PATH:-$OUT/policy_trace.jsonl}"
fi

if [ "${ROLLOUT_PROFILE:-0}" = "1" ]; then
  export ROLLOUT_PROFILE_SERVER_TRACE="${ROLLOUT_PROFILE_SERVER_TRACE:-$OUT/rollout_profile_server.jsonl}"
fi

# Warp/CuRobo kernels are GPU-architecture-specific.  Never reuse the shared
# home-directory caches across heterogeneous Hopper compute nodes.
CACHE_ROOT=${ROBOTWIN_CUROBO_CACHE_ROOT:-$OUT/_cache}
export WARP_CACHE_PATH="$CACHE_ROOT/warp"
export CUDA_CACHE_PATH="$CACHE_ROOT/cuda"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch_extensions"
mkdir -p "$WARP_CACHE_PATH" "$CUDA_CACHE_PATH" "$TORCH_EXTENSIONS_DIR"

# The editable RoboTwin environment contains CuRobo extensions compiled on a
# different GPU architecture.  Shadow only the Python/CUDA sources so CuRobo
# JIT-builds compatible extensions into this run's isolated cache.
CUROBO_SHADOW_ROOT="$OUT/_runtime/curobo_src"
mkdir -p "$CUROBO_SHADOW_ROOT"
tar -C "$ROBOTWIN_ROOT/envs/curobo/src" --exclude='*.so' -cf - curobo \
  | tar -C "$CUROBO_SHADOW_ROOT" -xf -
export PYTHONPATH="$CUROBO_SHADOW_ROOT${PYTHONPATH:+:$PYTHONPATH}"

RUNTIME_OVERLAY="$OUT/_logs/praxis-runtime-overlay.tar"
{
  git diff --name-only -z HEAD -- src scripts tests
  git ls-files --others --exclude-standard -z -- src scripts tests
} | sort -zu | tar -C "$SOURCE_ROOT" --null -T - -cf "$RUNTIME_OVERLAY"
export PRAXIS_RUNTIME_OVERLAY_PATH="$RUNTIME_OVERLAY"
PRAXIS_RUNTIME_OVERLAY_SHA256=$(sha256sum "$RUNTIME_OVERLAY" | awk '{print $1}')
export PRAXIS_RUNTIME_OVERLAY_SHA256

POLICY_PID=
cleanup() {
  if [ -n "$POLICY_PID" ]; then
    kill "$POLICY_PID" 2>/dev/null || true
    wait "$POLICY_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"$POLICY_SERVER" \
  policy="$POLICY_NAME" \
  policy.pretrained_path="$PRAXIS_CHECKPOINT" \
  host=127.0.0.1 \
  port="$POLICY_PORT" \
  device=cuda \
  >"$OUT/_logs/policy-server.log" 2>&1 &
POLICY_PID=$!

"$POLICY_PYTHON" - "$POLICY_PORT" "$PRAXIS_CHECKPOINT" "$POLICY_PID" <<'PY'
import json
import os
import sys
import time
from pathlib import Path

from praxis_remote import PolicyClient

port = int(sys.argv[1])
expected = Path(sys.argv[2]).resolve()
server_pid = int(sys.argv[3])
deadline = time.time() + 1800
while time.time() < deadline:
    try:
        os.kill(server_pid, 0)
    except ProcessLookupError as exc:
        raise RuntimeError("Policy server exited during startup") from exc
    client = PolicyClient(host="127.0.0.1", port=port, timeout=5.0)
    try:
        ready, info = client.health_check()
        loaded = Path(json.loads(info)["serving_artifact"]["checkpoint"]["path"]).resolve()
        if ready and loaded == expected:
            print(f"ROBOTWIN_HOPPER_POLICY_READY port={port}", flush=True)
            break
    except Exception:
        time.sleep(2)
    finally:
        client.close()
else:
    raise RuntimeError(f"Policy server did not become ready on port {port}")
PY

if [ -n "$ROBOTWIN_TASK" ]; then
  "$ROBOTWIN_PYTHON" -u scripts/robotwin/run_eval.py \
    --robotwin-root "$ROBOTWIN_ROOT" \
    --task "$ROBOTWIN_TASK" \
    --task-config demo_clean \
    --episodes-per-task "$EPISODES_PER_TASK" \
    --record-episodes-per-task "$EPISODES_PER_TASK" \
    --seed 0 \
    --policy-host 127.0.0.1 \
    --policy-port "$POLICY_PORT" \
    --output-dir "$OUT" \
    --run-id "$RUN_ID" \
    --expected-checkpoint "$PRAXIS_CHECKPOINT" \
    --max-seed-attempts "$MAX_SEED_ATTEMPTS" \
    --resume \
    2>&1 | tee -a "$OUT/_logs/eval-$ROBOTWIN_TASK.log"
  echo "ROBOTWIN_HOPPER_EVAL_COMPLETE task=$ROBOTWIN_TASK output_dir=$OUT"
  exit 0
fi

"$ROBOTWIN_PYTHON" -u scripts/robotwin/run_suite.py supervise \
  --robotwin-root "$ROBOTWIN_ROOT" \
  --tasks "$TASKS" \
  --policy-host 127.0.0.1 \
  --base-policy-port "$POLICY_PORT" \
  --output-dir "$OUT" \
  --run-id "$RUN_ID" \
  --expected-checkpoint "$PRAXIS_CHECKPOINT" \
  --gpu-count 1 \
  --simulators-per-gpu 1 \
  --server-count 1 \
  --episodes-per-task "$EPISODES_PER_TASK" \
  --max-seed-attempts "$MAX_SEED_ATTEMPTS" \
  --resume \
  2>&1 | tee -a "$OUT/_logs/supervisor.log"

"$ROBOTWIN_PYTHON" -u scripts/robotwin/run_suite.py aggregate \
  --tasks "$TASKS" \
  --output-dir "$OUT" \
  --run-id "$RUN_ID" \
  --expected-checkpoint "$PRAXIS_CHECKPOINT" \
  --episodes-per-task "$EPISODES_PER_TASK" \
  2>&1 | tee -a "$OUT/_logs/aggregate.log"

echo "ROBOTWIN_HOPPER_EVAL_COMPLETE output_dir=$OUT"
