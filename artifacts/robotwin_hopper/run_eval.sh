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

test -x "$POLICY_PYTHON"
test -x "$POLICY_SERVER"
test -x "$ROBOTWIN_PYTHON"
test -f "$ROBOTWIN_ROOT/script/eval_policy.py"
test -f "$PRAXIS_CHECKPOINT/config.json"
test -f "$PRAXIS_CHECKPOINT/model.safetensors"
test -f "$TASKS"
"$ROBOTWIN_PYTHON" -c "import praxis_remote"

mkdir -p "$OUT/_logs"
cd "$SOURCE_ROOT"

POLICY_PID=
cleanup() {
  if [ -n "$POLICY_PID" ]; then
    kill "$POLICY_PID" 2>/dev/null || true
    wait "$POLICY_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"$POLICY_SERVER" \
  policy=bar_bin_qwen \
  policy.pretrained_path="$PRAXIS_CHECKPOINT" \
  policy.n_action_steps=8 \
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
