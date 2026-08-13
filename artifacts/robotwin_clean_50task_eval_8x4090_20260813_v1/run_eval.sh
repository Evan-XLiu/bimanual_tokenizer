#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID=robotwin-ar-bin-qwen3vl2b-h32-final-ema-step10095-demo-clean-50x20-s0-8x4090-20260814-24sim-8server
SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ROBOTWIN_ROOT=/robot/robot-research-raw-data-0/user/lqw/code/RoboTwin
ROBOTWIN_PYTHON=/robot/robot-research-exp-0/user/hxs/envs/praxis-vla-robotwin-eval-py310-v2/bin/python
CHECKPOINT=/robot/robot-research-exp-0/user/hxs/praxis-vla/serving-artifacts/robotwin-bar-bin-qwen3vl2b-h32-final-ema-step10095-robot-mount-v1/pretrained_model
POLICY_HOST=10.92.9.109
POLICY_BASE_PORT=50100
POLICY_REPLICAS=8
SIMULATORS_PER_GPU=3
GPU_COUNT=8
WORKER_COUNT=$((GPU_COUNT * SIMULATORS_PER_GPU))
TASKS=$SOURCE_ROOT/scripts/robotwin/all_tasks.yml
OUT=/robot/robot-research-exp-0/user/hxs/praxis-vla/outputs/robotwin2_clean/$RUN_ID

WORKER_PIDS=()
cleanup() {
  local pid
  for pid in "${WORKER_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

test ! -e "$OUT"
mkdir -p "$OUT/_logs"
cd "$SOURCE_ROOT"

"$ROBOTWIN_PYTHON" - "$CHECKPOINT" "$POLICY_HOST" "$POLICY_BASE_PORT" "$POLICY_REPLICAS" <<'PY'
import json
import sys
import time
from pathlib import Path

from praxis_remote import PolicyClient

expected = Path(sys.argv[1]).resolve()
host = sys.argv[2]
base_port = int(sys.argv[3])
replicas = int(sys.argv[4])
for port in range(base_port, base_port + replicas):
    for _ in range(120):
        client = PolicyClient(host=host, port=port, timeout=5.0)
        try:
            try:
                ready, model_info = client.health_check()
                loaded = Path(
                    json.loads(model_info)["serving_artifact"]["checkpoint"]["path"]
                ).resolve()
                if ready and loaded == expected:
                    print(
                        f"ROBOTWIN_POLICY_SERVER_READY host={host} port={port}",
                        flush=True,
                    )
                    break
            except Exception:
                pass
        finally:
            client.close()
        time.sleep(5)
    else:
        raise RuntimeError(
            f"Policy server did not become ready: host={host} port={port}"
        )
PY

for worker in $(seq 0 $((WORKER_COUNT - 1))); do
  gpu=$((worker % GPU_COUNT))
  policy_port=$((POLICY_BASE_PORT + (worker % POLICY_REPLICAS)))
  CUDA_VISIBLE_DEVICES=$gpu "$ROBOTWIN_PYTHON" -u scripts/robotwin/run_suite.py worker \
    --robotwin-root "$ROBOTWIN_ROOT" \
    --tasks "$TASKS" \
    --worker-index "$worker" \
    --worker-count "$WORKER_COUNT" \
    --policy-host "$POLICY_HOST" \
    --policy-port "$policy_port" \
    --output-dir "$OUT" \
    --run-id "$RUN_ID" \
    --expected-checkpoint "$CHECKPOINT" \
    --episodes-per-task 20 \
    --max-seed-attempts 1000 \
    >"$OUT/_logs/worker-$worker.log" 2>&1 &
  WORKER_PIDS+=("$!")
done

failed=0
for worker in $(seq 0 $((WORKER_COUNT - 1))); do
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
