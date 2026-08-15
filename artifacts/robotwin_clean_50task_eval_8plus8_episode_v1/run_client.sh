#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID=robotwin-ar-bin-qwen3vl2b-h32-final-ema-step10095-demo-clean-50x20-s0-h20e-8plus8-taskmajor-20260815-v4
SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ROBOTWIN_ROOT=/mnt/robot-research-raw-data-0/user/lqw/code/RoboTwin
ROBOTWIN_PYTHON=/robot/robot-research-exp-0/user/hxs/envs/praxis-vla-robotwin-eval-py310-v2/bin/python
CHECKPOINT=/robot/robot-research-exp-0/user/hxs/praxis-vla/serving-artifacts/robotwin-bar-bin-qwen3vl2b-h32-final-ema-step10095-robot-mount-v1/pretrained_model
RDZ_ROOT=/robot/robot-research-exp-0/user/hxs/praxis-vla/outputs/robotwin2_clean/_rendezvous/$RUN_ID
OUT=/robot/robot-research-exp-0/user/hxs/praxis-vla/outputs/robotwin2_clean/$RUN_ID
TASKS=$SOURCE_ROOT/scripts/robotwin/all_tasks.yml
GPU_COUNT=8
SIMULATORS_PER_GPU=3
WORKER_COUNT=$((GPU_COUNT * SIMULATORS_PER_GPU))
SERVER_COUNT=8
EPISODES_PER_TASK=20

test -x "$ROBOTWIN_PYTHON"
test -f "$ROBOTWIN_ROOT/script/eval_policy.py"
test -f "$CHECKPOINT/config.json"
test -f "$CHECKPOINT/model.safetensors"
test -f "$TASKS"
mkdir -p "$RDZ_ROOT" "$OUT/_logs"

RESUME_FLAG=(--resume)
PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_for_workers() {
  local failed=0
  for _ in "${PIDS[@]}"; do
    wait -n || failed=1
  done
  return "$failed"
}

policy_endpoints_healthy() {
  "$ROBOTWIN_PYTHON" - "$1" "$2" "$CHECKPOINT" "$SERVER_COUNT" <<'PY'
import json
import sys
from pathlib import Path

from praxis_remote import PolicyClient

host, base_port = sys.argv[1], int(sys.argv[2])
expected = Path(sys.argv[3]).resolve()
server_count = int(sys.argv[4])
for server_index in range(server_count):
    port = base_port + server_index
    client = PolicyClient(host=host, port=port, timeout=10.0)
    try:
        ready, info = client.health_check()
        loaded = Path(json.loads(info)["serving_artifact"]["checkpoint"]["path"]).resolve()
        if not ready or loaded != expected:
            raise RuntimeError(f"Policy endpoint is not ready: {host}:{port}")
    finally:
        client.close()
    print(
        f"ROBOTWIN_CLIENT_POLICY_HEALTH_OK host={host} port={port}",
        flush=True,
    )
PY
}

deadline=$(($(date +%s) + 86400))
while :; do
  if [ -s "$RDZ_ROOT/server_generation" ] && [ -s "$RDZ_ROOT/server.endpoint" ] && [ -s "$RDZ_ROOT/server.ready" ]; then
    candidate_generation=$(tr -d '[:space:]' <"$RDZ_ROOT/server_generation")
    read -r candidate_endpoint_generation candidate_host candidate_port <"$RDZ_ROOT/server.endpoint"
    candidate_ready_generation=$(tr -d '[:space:]' <"$RDZ_ROOT/server.ready")
    if [ "$candidate_generation" = "$candidate_endpoint_generation" ] && [ "$candidate_generation" = "$candidate_ready_generation" ]; then
      candidate_ip=$(getent ahostsv4 "$candidate_host" | awk '$1 !~ /^127\./ { print $1; exit }' || true)
      if [ -n "$candidate_ip" ] && policy_endpoints_healthy "$candidate_host" "$candidate_port"; then
        GENERATION=$candidate_generation
        POLICY_HOST=$candidate_host
        BASE_PORT=$candidate_port
        policy_ip=$candidate_ip
        break
      fi
    fi
  fi
  [ "$(date +%s)" -lt "$deadline" ] || {
    echo "Timed out waiting for policy server" >&2
    exit 1
  }
  sleep 5
done
echo "ROBOTWIN_CLIENT_POLICY_ENDPOINTS generation=$GENERATION host=$POLICY_HOST ip=$policy_ip ports=$BASE_PORT-$((BASE_PORT + SERVER_COUNT - 1))"

cd "$SOURCE_ROOT"
echo "ROBOTWIN_TASK_MAJOR_PHASE_START workers=$WORKER_COUNT simulators_per_gpu=$SIMULATORS_PER_GPU tasks=50 episodes_per_task=$EPISODES_PER_TASK"
for worker in $(seq 0 $((WORKER_COUNT - 1))); do
  gpu=$((worker / SIMULATORS_PER_GPU))
  server_index=$((worker % SERVER_COUNT))
  policy_port=$((BASE_PORT + server_index))
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
    --episodes-per-task "$EPISODES_PER_TASK" \
    --max-seed-attempts 1000 \
    "${RESUME_FLAG[@]}" \
    2>&1 | sed -u $'s/\033\\[[0-9;]*m//g' | sed -u "s/^/[task-worker-$worker] /" | tee -a "$OUT/_logs/task-worker-$worker.log" &
  PIDS+=("$!")
done
if ! wait_for_workers; then
  echo "ROBOTWIN_TASK_MAJOR_PHASE_FAILED" >&2
  exit 1
fi
PIDS=()

"$ROBOTWIN_PYTHON" -u scripts/robotwin/run_suite.py aggregate \
  --tasks "$TASKS" \
  --output-dir "$OUT" \
  --run-id "$RUN_ID" \
  --expected-checkpoint "$CHECKPOINT" \
  --episodes-per-task "$EPISODES_PER_TASK" \
  2>&1 | tee -a "$OUT/_logs/aggregate.log"

printf '%s\n' "$GENERATION" >"$RDZ_ROOT/client_complete.tmp"
mv "$RDZ_ROOT/client_complete.tmp" "$RDZ_ROOT/client_complete"
echo "ROBOTWIN_CLEAN_50X20_FINAL_EMA_EVAL_COMPLETE output_dir=$OUT"
