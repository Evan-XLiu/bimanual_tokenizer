#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID=robotwin-ar-bin-qwen3vl2b-h32-final-ema-step10095-demo-clean-50x20-s0-h20-8plus8-24sim-episode-20260814-v1
SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ROBOTWIN_ROOT=/mnt/robot-research-raw-data-0/user/lqw/code/RoboTwin
ROBOTWIN_PYTHON=/robot/robot-research-exp-0/user/hxs/envs/praxis-vla-robotwin-eval-py310-v2/bin/python
CHECKPOINT=/robot/robot-research-exp-0/user/hxs/praxis-vla/serving-artifacts/robotwin-bar-bin-qwen3vl2b-h32-final-ema-step10095-robot-mount-v1/pretrained_model
RDZ_ROOT=/robot/robot-research-exp-0/user/hxs/praxis-vla/outputs/robotwin2_clean/_rendezvous/$RUN_ID
OUT=/robot/robot-research-exp-0/user/hxs/praxis-vla/outputs/robotwin2_clean/$RUN_ID
TASKS=$SOURCE_ROOT/scripts/robotwin/all_tasks.yml
PLUGIN=$SOURCE_ROOT/scripts/robotwin/policy/praxis_ar/deploy_policy.yml
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
test -f "$PLUGIN"
mkdir -p "$RDZ_ROOT" "$OUT/_logs"

RESUME_FLAG=()
if [ -e "$OUT/progress.jsonl" ] || [ -e "$OUT/adjust_bottle/episode_plan.json" ]; then
  RESUME_FLAG+=(--resume)
fi

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd "$SOURCE_ROOT"
echo "ROBOTWIN_PLANNING_PHASE_START workers=$WORKER_COUNT tasks=50 episodes_per_task=$EPISODES_PER_TASK"
for worker in $(seq 0 $((WORKER_COUNT - 1))); do
  gpu=$((worker / SIMULATORS_PER_GPU))
  CUDA_VISIBLE_DEVICES=$gpu "$ROBOTWIN_PYTHON" -u scripts/robotwin/run_episode_suite.py plan-worker \
    --robotwin-root "$ROBOTWIN_ROOT" \
    --plugin-config "$PLUGIN" \
    --tasks "$TASKS" \
    --worker-index "$worker" \
    --worker-count "$WORKER_COUNT" \
    --output-dir "$OUT" \
    --run-id "$RUN_ID" \
    --episodes-per-task "$EPISODES_PER_TASK" \
    --max-seed-attempts 1000 \
    "${RESUME_FLAG[@]}" \
    2>&1 | sed -u "s/^/[planner-$worker] /" | tee -a "$OUT/_logs/planner-$worker.log" &
  PIDS+=("$!")
done
failed=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || failed=1
done
test "$failed" -eq 0
PIDS=()
echo "ROBOTWIN_PLANNING_PHASE_COMPLETE tasks=50 plans=1000"

deadline=$(($(date +%s) + 86400))
while [ ! -s "$RDZ_ROOT/server_generation" ] || [ ! -s "$RDZ_ROOT/server.endpoint" ] || [ ! -s "$RDZ_ROOT/server.ready" ]; do
  [ "$(date +%s)" -lt "$deadline" ] || { echo "Timed out waiting for policy server" >&2; exit 1; }
  sleep 5
done
GENERATION=$(tr -d '[:space:]' <"$RDZ_ROOT/server_generation")
read -r endpoint_generation POLICY_HOST BASE_PORT <"$RDZ_ROOT/server.endpoint"
ready_generation=$(tr -d '[:space:]' <"$RDZ_ROOT/server.ready")
[ "$endpoint_generation" = "$GENERATION" ]
[ "$ready_generation" = "$GENERATION" ]
policy_ip=$(getent ahostsv4 "$POLICY_HOST" | awk '$1 !~ /^127\./ { print $1; exit }')
test -n "$policy_ip"
echo "ROBOTWIN_CLIENT_POLICY_ENDPOINTS generation=$GENERATION host=$POLICY_HOST ip=$policy_ip ports=$BASE_PORT-$((BASE_PORT + SERVER_COUNT - 1))"

for server_index in $(seq 0 $((SERVER_COUNT - 1))); do
  port=$((BASE_PORT + server_index))
  "$ROBOTWIN_PYTHON" - "$POLICY_HOST" "$port" "$CHECKPOINT" <<'PY'
import json
import sys
from pathlib import Path

from praxis_remote import PolicyClient

client = PolicyClient(host=sys.argv[1], port=int(sys.argv[2]), timeout=30.0)
try:
    ready, info = client.health_check()
    loaded = Path(json.loads(info)["serving_artifact"]["checkpoint"]["path"]).resolve()
    assert ready and loaded == Path(sys.argv[3]).resolve()
    print(f"ROBOTWIN_CLIENT_POLICY_HEALTH_OK host={sys.argv[1]} port={sys.argv[2]}", flush=True)
finally:
    client.close()
PY
done

echo "ROBOTWIN_ROLLOUT_PHASE_START workers=$WORKER_COUNT simulators_per_gpu=$SIMULATORS_PER_GPU"
for worker in $(seq 0 $((WORKER_COUNT - 1))); do
  gpu=$((worker / SIMULATORS_PER_GPU))
  server_index=$((worker % SERVER_COUNT))
  policy_port=$((BASE_PORT + server_index))
  CUDA_VISIBLE_DEVICES=$gpu "$ROBOTWIN_PYTHON" -u scripts/robotwin/run_episode_suite.py episode-worker \
    --robotwin-root "$ROBOTWIN_ROOT" \
    --plugin-config "$PLUGIN" \
    --tasks "$TASKS" \
    --worker-index "$worker" \
    --worker-count "$WORKER_COUNT" \
    --policy-host "$POLICY_HOST" \
    --policy-port "$policy_port" \
    --output-dir "$OUT" \
    --run-id "$RUN_ID" \
    --expected-checkpoint "$CHECKPOINT" \
    --episodes-per-task "$EPISODES_PER_TASK" \
    --heartbeat-steps 100 \
    "${RESUME_FLAG[@]}" \
    2>&1 | sed -u "s/^/[episode-$worker] /" | tee -a "$OUT/_logs/episode-$worker.log" &
  PIDS+=("$!")
done
failed=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || failed=1
done
test "$failed" -eq 0
PIDS=()
echo "ROBOTWIN_ROLLOUT_PHASE_COMPLETE episodes=1000"

"$ROBOTWIN_PYTHON" -u scripts/robotwin/run_episode_suite.py aggregate \
  --tasks "$TASKS" \
  --output-dir "$OUT" \
  --run-id "$RUN_ID" \
  --expected-checkpoint "$CHECKPOINT" \
  --episodes-per-task "$EPISODES_PER_TASK" \
  2>&1 | tee -a "$OUT/_logs/aggregate.log"

printf '%s\n' "$GENERATION" >"$RDZ_ROOT/client_complete.tmp"
mv "$RDZ_ROOT/client_complete.tmp" "$RDZ_ROOT/client_complete"
echo "ROBOTWIN_CLEAN_50X20_FINAL_EMA_EVAL_COMPLETE output_dir=$OUT"
