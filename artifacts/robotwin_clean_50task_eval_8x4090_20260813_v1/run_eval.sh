#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID=robotwin-ar-bin-qwen3vl2b-h32-final-ema-step10095-demo-clean-50x20-s0-8x4090-20260813-simple-v2
SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ROBOTWIN_ROOT=/robot/robot-research-raw-data-0/user/lqw/code/RoboTwin
POLICY_ENV=/robot/robot-research-exp-0/user/hxs/envs/praxis-vla-robotwin-policy-py310-v1
POLICY_PYTHON=$POLICY_ENV/bin/python
POLICY_SERVER=$POLICY_ENV/lib/python3.10/site-packages/praxis/scripts/serve.py
ROBOTWIN_PYTHON=/robot/robot-research-exp-0/user/hxs/envs/praxis-vla-robotwin-eval-py310-v2/bin/python
CHECKPOINT=/robot/robot-research-exp-0/user/hxs/praxis-vla/serving-artifacts/robotwin-bar-bin-qwen3vl2b-h32-final-ema-step10095-robot-mount-v1/pretrained_model
TASKS=$SOURCE_ROOT/scripts/robotwin/all_tasks.yml
OUT=/robot/robot-research-exp-0/user/hxs/praxis-vla/outputs/robotwin2_clean/$RUN_ID

SERVER_PIDS=()
WORKER_PIDS=()
cleanup() {
  local pid
  for pid in "${WORKER_PIDS[@]:-}" "${SERVER_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

test ! -e "$OUT"
mkdir -p "$OUT/_logs"
cd "$SOURCE_ROOT"

for gpu in $(seq 0 7); do
  port=$((50051 + gpu))
  CUDA_VISIBLE_DEVICES=$gpu "$POLICY_PYTHON" -u "$POLICY_SERVER" \
    policy=bar_bin_qwen \
    policy.pretrained_path="$CHECKPOINT" \
    host=127.0.0.1 \
    port="$port" \
    device=cuda \
    >"$OUT/_logs/server-$gpu.log" 2>&1 &
  SERVER_PIDS+=("$!")
  "$POLICY_PYTHON" - "$CHECKPOINT" "$port" "$gpu" <<'PY'
import json
import sys
import time
from pathlib import Path

from praxis_remote import PolicyClient

expected = Path(sys.argv[1]).resolve()
port = int(sys.argv[2])
gpu = int(sys.argv[3])
for _ in range(120):
    client = PolicyClient(host="127.0.0.1", port=port, timeout=5.0)
    try:
        try:
            ready, model_info = client.health_check()
            loaded = Path(
                json.loads(model_info)["serving_artifact"]["checkpoint"]["path"]
            ).resolve()
            if ready and loaded == expected:
                print(f"ROBOTWIN_POLICY_SERVER_READY gpu={gpu}", flush=True)
                break
        except Exception:
            pass
    finally:
        client.close()
    time.sleep(5)
else:
    raise RuntimeError(f"Policy server did not become ready: gpu={gpu}")
PY
done

for worker in $(seq 0 7); do
  port=$((50051 + worker))
  CUDA_VISIBLE_DEVICES=$worker "$ROBOTWIN_PYTHON" -u scripts/robotwin/run_suite.py worker \
    --robotwin-root "$ROBOTWIN_ROOT" \
    --tasks "$TASKS" \
    --worker-index "$worker" \
    --worker-count 8 \
    --policy-host 127.0.0.1 \
    --policy-port "$port" \
    --output-dir "$OUT" \
    --run-id "$RUN_ID" \
    --expected-checkpoint "$CHECKPOINT" \
    --episodes-per-task 20 \
    --max-seed-attempts 1000 \
    >"$OUT/_logs/worker-$worker.log" 2>&1 &
  WORKER_PIDS+=("$!")
done

failed=0
for worker in $(seq 0 7); do
  wait "${WORKER_PIDS[$worker]}" || failed=1
done
test "$failed" -eq 0

"$ROBOTWIN_PYTHON" -u scripts/robotwin/run_suite.py aggregate \
  --tasks "$TASKS" \
  --output-dir "$OUT" \
  --run-id "$RUN_ID" \
  --expected-checkpoint "$CHECKPOINT" \
  --episodes-per-task 20

echo "ROBOTWIN_CLEAN_50X20_FINAL_EMA_EVAL_COMPLETE output_dir=$OUT"
