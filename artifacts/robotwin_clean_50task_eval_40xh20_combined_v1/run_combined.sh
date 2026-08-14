#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID=robotwin-ar-bin-qwen3vl2b-h32-final-ema-step10095-demo-clean-50x20-s0-h20-16policy-20260814-v20
SOURCE_ROOT=/robot/robot-research-exp-0/user/hxs/praxis-vla
ROBOTWIN_ROOT=/robot/robot-research-raw-data-0/user/lqw/code/RoboTwin
ROBOTWIN_PYTHON=/robot/robot-research-exp-0/user/hxs/envs/praxis-vla-robotwin-eval-py310-v2/bin/python
POLICY_ENV=/robot/robot-research-exp-0/user/hxs/envs/praxis-vla-robotwin-policy-py310-v1
POLICY_PYTHON=$POLICY_ENV/bin/python
POLICY_SERVER=$POLICY_ENV/lib/python3.10/site-packages/praxis/scripts/serve.py
CHECKPOINT=/robot/robot-research-exp-0/user/hxs/praxis-vla/serving-artifacts/robotwin-bar-bin-qwen3vl2b-h32-final-ema-step10095-robot-mount-v1/pretrained_model
RDZ_ROOT=/robot/robot-research-exp-0/user/hxs/praxis-vla/outputs/robotwin2_clean/_rendezvous/$RUN_ID
OUT=/robot/robot-research-exp-0/user/hxs/praxis-vla/outputs/robotwin2_clean/$RUN_ID
TASKS=$SOURCE_ROOT/scripts/robotwin/all_tasks.yml
NODE_COUNT=5
POLICY_NODE_COUNT=2
CLIENT_NODE_COUNT=3
GPU_COUNT=8
SIMULATORS_PER_GPU=2
WORKERS_PER_CLIENT_NODE=$((GPU_COUNT * SIMULATORS_PER_GPU))
WORKER_COUNT=$((CLIENT_NODE_COUNT * WORKERS_PER_CLIENT_NODE))
SERVER_COUNT=$((POLICY_NODE_COUNT * GPU_COUNT))
BASE_PORT=50100
DEADLINE=$(($(date +%s) + 86400))
HEALTH_TIMEOUT_SEC=1800

case "${HOSTNAME:-}" in
  *-master-*) NODE_RANK=0 ;;
  *-worker-*) NODE_RANK=$((10#${HOSTNAME##*-worker-} + 1)) ;;
  *)
    NODE_RANK=${MLP_ROLE_INDEX:-${ARNOLD_ID:-${NODE_RANK:-${RANK:-0}}}}
    NODE_RANK=$((10#$NODE_RANK))
    ;;
esac
test "$NODE_RANK" -ge 0
test "$NODE_RANK" -lt "$NODE_COUNT"
test -x "$POLICY_PYTHON"
test -x "$ROBOTWIN_PYTHON"
test -f "$POLICY_SERVER"
test -f "$CHECKPOINT/config.json"
test -f "$CHECKPOINT/model.safetensors"
mkdir -p "$RDZ_ROOT" "$OUT/_logs"

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [ "$NODE_RANK" -lt "$POLICY_NODE_COUNT" ]; then
  namespace=$(cat /var/run/secrets/kubernetes.io/serviceaccount/namespace)
  host="${HOSTNAME}.${namespace}.svc.cluster.local"
  host_ip=$(getent ahostsv4 "$host" | awk '$1 !~ /^127\\./ { print $1; exit }')
  test -n "$host_ip"
  if [ "$NODE_RANK" = 0 ]; then
    generation="${CLOUDML_JOB_ID:-robotwin-combined}-${RUN_ID}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
    printf '%s\n' "$generation" >"$RDZ_ROOT/server_generation.tmp"
    mv "$RDZ_ROOT/server_generation.tmp" "$RDZ_ROOT/server_generation"
  fi
  while [ ! -s "$RDZ_ROOT/server_generation" ]; do sleep 2; done
  generation=$(tr -d '[:space:]' <"$RDZ_ROOT/server_generation")
  endpoint_file="$RDZ_ROOT/server-${NODE_RANK}.endpoint"
  ready_file="$RDZ_ROOT/server-${NODE_RANK}.ready"
  rm -f "$endpoint_file" "$ready_file"
  printf '%s %s %s\n' "$generation" "$host" "$((BASE_PORT + NODE_RANK * GPU_COUNT))" >"$endpoint_file"
  echo "ROBOTWIN_COMBINED_POLICY_ENDPOINT node=$NODE_RANK host=$host ip=$host_ip"
  for gpu in $(seq 0 $((GPU_COUNT - 1))); do
    port=$((BASE_PORT + NODE_RANK * GPU_COUNT + gpu))
    CUDA_VISIBLE_DEVICES=$gpu "$POLICY_PYTHON" -u "$POLICY_SERVER" \
      policy=bar_bin_qwen policy.pretrained_path="$CHECKPOINT" \
      host=0.0.0.0 port="$port" device=cuda \
      >"$RDZ_ROOT/server-${NODE_RANK}-${gpu}.log" 2>&1 &
    PIDS+=("$!")
  done
  health_deadline=$(($(date +%s) + HEALTH_TIMEOUT_SEC))
  for gpu in $(seq 0 $((GPU_COUNT - 1))); do
    port=$((BASE_PORT + NODE_RANK * GPU_COUNT + gpu))
    "$POLICY_PYTHON" - 127.0.0.1 "$port" "$CHECKPOINT" "$health_deadline" <<'PY'
import json, sys, time
from pathlib import Path
from praxis_remote import PolicyClient

host, port, expected, deadline = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]).resolve(), int(sys.argv[4])
while time.time() < deadline:
    client = PolicyClient(host=host, port=port, timeout=5.0)
    try:
        try:
            ready, info = client.health_check()
            if ready and Path(json.loads(info)["serving_artifact"]["checkpoint"]["path"]).resolve() == expected:
                print(f"ROBOTWIN_COMBINED_POLICY_READY host={host} port={port}", flush=True)
                break
        except Exception:
            pass
    finally:
        client.close()
    time.sleep(2)
else:
    raise RuntimeError(f"Policy server did not become ready: {host}:{port}")
PY
  done
  printf '%s\n' "$generation" >"$ready_file"
  echo "ROBOTWIN_COMBINED_POLICY_NODE_READY node=$NODE_RANK"
  while :; do
    if [ -s "$RDZ_ROOT/client_complete" ] && [ "$(tr -d '[:space:]' <"$RDZ_ROOT/client_complete")" = "$generation" ]; then break; fi
    [ "$(date +%s)" -lt "$DEADLINE" ] || { echo "Timed out waiting for client completion"; exit 1; }
    sleep 10
  done
else
  CLIENT_NODE_RANK=$((NODE_RANK - POLICY_NODE_COUNT))
  echo "ROBOTWIN_COMBINED_CLIENT_NODE node=$NODE_RANK client_node=$CLIENT_NODE_RANK"
  while [ ! -s "$RDZ_ROOT/server_generation" ] || [ "$(find "$RDZ_ROOT" -maxdepth 1 -name 'server-*.ready' | wc -l)" -lt "$POLICY_NODE_COUNT" ]; do
    [ "$(date +%s)" -lt "$DEADLINE" ] || { echo "Timed out waiting for policy endpoints"; exit 1; }
    sleep 5
  done
  generation=$(tr -d '[:space:]' <"$RDZ_ROOT/server_generation")
  ENDPOINT_HOSTS=()
  ENDPOINT_PORTS=()
  for endpoint in "$RDZ_ROOT"/server-*.endpoint; do
    read -r endpoint_generation endpoint_host endpoint_port <"$endpoint"
    [ "$endpoint_generation" = "$generation" ] || continue
    for offset in $(seq 0 $((GPU_COUNT - 1))); do
      ENDPOINT_HOSTS+=("$endpoint_host")
      ENDPOINT_PORTS+=("$((endpoint_port + offset))")
    done
  done
  test "${#ENDPOINT_HOSTS[@]}" -eq "$SERVER_COUNT"
  "$ROBOTWIN_PYTHON" - "$CHECKPOINT" "${ENDPOINT_HOSTS[@]}" -- "${ENDPOINT_PORTS[@]}" <<'PY'
import json, sys
from pathlib import Path
from praxis_remote import PolicyClient

expected = Path(sys.argv[1]).resolve()
separator = sys.argv.index("--")
hosts = sys.argv[2:separator]
ports = [int(value) for value in sys.argv[separator + 1:]]
assert len(hosts) == len(ports) == 16
for index, (host, port) in enumerate(zip(hosts, ports, strict=True)):
    client = PolicyClient(host=host, port=port, timeout=15.0)
    try:
        ready, info = client.health_check()
    finally:
        client.close()
    assert ready
    loaded = Path(json.loads(info)["serving_artifact"]["checkpoint"]["path"]).resolve()
    assert loaded == expected, (loaded, expected)
    print(f"ROBOTWIN_COMBINED_CLIENT_ENDPOINT_READY index={index} host={host} port={port}", flush=True)
PY
  cd "$SOURCE_ROOT"
  for local_worker in $(seq 0 $((WORKERS_PER_CLIENT_NODE - 1))); do
    gpu=$((local_worker / SIMULATORS_PER_GPU))
    global_worker=$((CLIENT_NODE_RANK * WORKERS_PER_CLIENT_NODE + local_worker))
    endpoint_index=$((global_worker % SERVER_COUNT))
    CUDA_VISIBLE_DEVICES=$gpu "$ROBOTWIN_PYTHON" -u scripts/robotwin/run_suite.py worker \
      --robotwin-root "$ROBOTWIN_ROOT" --tasks "$TASKS" \
      --worker-index "$global_worker" --worker-count "$WORKER_COUNT" \
      --policy-host "${ENDPOINT_HOSTS[$endpoint_index]}" --policy-port "${ENDPOINT_PORTS[$endpoint_index]}" \
      --output-dir "$OUT" --run-id "$RUN_ID" --expected-checkpoint "$CHECKPOINT" \
      --episodes-per-task 20 --max-seed-attempts 1000 --resume \
      >"$OUT/_logs/worker-$global_worker.log" 2>&1 &
    PIDS+=("$!")
  done
  failed=0
  for pid in "${PIDS[@]}"; do wait "$pid" || failed=1; done
  test "$failed" -eq 0
  printf '%s\n' "$generation" >"$RDZ_ROOT/client-node-$CLIENT_NODE_RANK.complete"
  if [ "$CLIENT_NODE_RANK" = 0 ]; then
    while [ "$(find "$RDZ_ROOT" -maxdepth 1 -name 'client-node-*.complete' | wc -l)" -lt "$CLIENT_NODE_COUNT" ]; do
      [ "$(date +%s)" -lt "$DEADLINE" ] || { echo "Timed out waiting for client nodes"; exit 1; }
      sleep 5
    done
    "$ROBOTWIN_PYTHON" -u scripts/robotwin/run_suite.py aggregate \
      --tasks "$TASKS" --output-dir "$OUT" --run-id "$RUN_ID" \
      --expected-checkpoint "$CHECKPOINT" --episodes-per-task 20
    printf '%s\n' "$generation" >"$RDZ_ROOT/client_complete.tmp"
    mv "$RDZ_ROOT/client_complete.tmp" "$RDZ_ROOT/client_complete"
  fi
fi

echo "ROBOTWIN_COMBINED_NODE_COMPLETE node=$NODE_RANK output_dir=$OUT"
