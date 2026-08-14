#!/usr/bin/env bash
set -Eeuo pipefail
RUN_ID=robotwin-ar-bin-qwen3vl2b-h32-final-ema-step10095-demo-clean-50x20-s0-h20-16policy-20260814-v4
SOURCE_ROOT=/robot/robot-research-exp-0/user/hxs/praxis-vla
ROBOTWIN_ROOT=/robot/robot-research-raw-data-0/user/lqw/code/RoboTwin
ROBOTWIN_PYTHON=/robot/robot-research-exp-0/user/hxs/envs/praxis-vla-robotwin-eval-py310-v2/bin/python
CHECKPOINT=/robot/robot-research-exp-0/user/hxs/praxis-vla/serving-artifacts/robotwin-bar-bin-qwen3vl2b-h32-final-ema-step10095-robot-mount-v1/pretrained_model
RDZ_ROOT=/robot/robot-research-exp-0/user/hxs/praxis-vla/outputs/robotwin2_clean/_rendezvous/$RUN_ID
OUT=/robot/robot-research-exp-0/user/hxs/praxis-vla/outputs/robotwin2_clean/$RUN_ID
TASKS=$SOURCE_ROOT/scripts/robotwin/all_tasks.yml
NODE_COUNT=3; GPU_COUNT=8; SIMULATORS_PER_GPU=2; WORKERS_PER_NODE=$((GPU_COUNT * SIMULATORS_PER_GPU)); WORKER_COUNT=$((NODE_COUNT * WORKERS_PER_NODE)); SERVER_COUNT=16
NODE_RANK=${MLP_ROLE_INDEX:-${ARNOLD_ID:-${NODE_RANK:-${RANK:-0}}}}; NODE_RANK=$((10#$NODE_RANK)); test "$NODE_RANK" -lt "$NODE_COUNT"
mkdir -p "$RDZ_ROOT" "$OUT/_logs"
deadline=$(($(date +%s) + 86400))
while [ ! -s "$RDZ_ROOT/server_generation" ] || [ "$(find "$RDZ_ROOT" -maxdepth 1 -name 'server-*.ready' | wc -l)" -lt 2 ]; do
  [ "$(date +%s)" -lt "$deadline" ] || { echo "Timed out waiting for policy endpoints"; exit 1; }
  sleep 5
done
GENERATION=$(tr -d '[:space:]' <"$RDZ_ROOT/server_generation")
ENDPOINT_HOSTS=(); ENDPOINT_PORTS=()
for endpoint in "$RDZ_ROOT"/server-*.endpoint; do
  read -r generation host port <"$endpoint"
  [ "$generation" = "$GENERATION" ] || continue
  for offset in $(seq 0 7); do
    ENDPOINT_HOSTS+=("$host"); ENDPOINT_PORTS+=("$((port + offset))")
  done
done
[ "${#ENDPOINT_HOSTS[@]}" -eq "$SERVER_COUNT" ] || { echo "Expected $SERVER_COUNT policy endpoints, found ${#ENDPOINT_HOSTS[@]}"; exit 1; }
RESUME_FLAG=()
if [ -e "$OUT/comparison.json" ] || [ -d "$OUT/adjust_bottle" ]; then RESUME_FLAG+=(--resume); fi
PIDS=()
cleanup() { for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done; wait 2>/dev/null || true; }
trap cleanup EXIT INT TERM
cd "$SOURCE_ROOT"
for local_worker in $(seq 0 $((WORKERS_PER_NODE - 1))); do
  gpu=$((local_worker / SIMULATORS_PER_GPU)); global_worker=$((NODE_RANK * WORKERS_PER_NODE + local_worker)); endpoint_index=$((global_worker % SERVER_COUNT))
  CUDA_VISIBLE_DEVICES=$gpu "$ROBOTWIN_PYTHON" -u scripts/robotwin/run_suite.py worker \
    --robotwin-root "$ROBOTWIN_ROOT" --tasks "$TASKS" --worker-index "$global_worker" --worker-count "$WORKER_COUNT" \
    --policy-host "${ENDPOINT_HOSTS[$endpoint_index]}" --policy-port "${ENDPOINT_PORTS[$endpoint_index]}" \
    --output-dir "$OUT" --run-id "$RUN_ID" --expected-checkpoint "$CHECKPOINT" --episodes-per-task 20 --max-seed-attempts 1000 \
    "${RESUME_FLAG[@]}" >"$OUT/_logs/worker-$global_worker.log" 2>&1 &
  PIDS+=("$!")
done
failed=0
for pid in "${PIDS[@]}"; do wait "$pid" || failed=1; done
test "$failed" -eq 0
printf '%s\n' "$GENERATION" >"$RDZ_ROOT/client-node-$NODE_RANK.complete"
if [ "$NODE_RANK" = 0 ]; then
  while [ "$(find "$RDZ_ROOT" -maxdepth 1 -name 'client-node-*.complete' | wc -l)" -lt "$NODE_COUNT" ]; do
    [ "$(date +%s)" -lt "$deadline" ] || { echo "Timed out waiting for client nodes"; exit 1; }
    sleep 5
  done
  "$ROBOTWIN_PYTHON" -u scripts/robotwin/run_suite.py aggregate --tasks "$TASKS" --output-dir "$OUT" --run-id "$RUN_ID" --expected-checkpoint "$CHECKPOINT" --episodes-per-task 20
  printf '%s\n' "$GENERATION" >"$RDZ_ROOT/client_complete.tmp"; mv "$RDZ_ROOT/client_complete.tmp" "$RDZ_ROOT/client_complete"
fi
echo "ROBOTWIN_CLEAN_50X20_FINAL_EMA_EVAL_COMPLETE output_dir=$OUT"
