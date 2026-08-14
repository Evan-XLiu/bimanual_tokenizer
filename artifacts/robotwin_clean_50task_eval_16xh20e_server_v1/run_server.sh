#!/usr/bin/env bash
set -Eeuo pipefail
RUN_ID=robotwin-ar-bin-qwen3vl2b-h32-final-ema-step10095-demo-clean-50x20-s0-h20-16policy-20260814-v7
POLICY_ENV=/robot/robot-research-exp-0/user/hxs/envs/praxis-vla-robotwin-policy-py310-v1
POLICY_PYTHON=$POLICY_ENV/bin/python
POLICY_SERVER=$POLICY_ENV/lib/python3.10/site-packages/praxis/scripts/serve.py
CHECKPOINT=/robot/robot-research-exp-0/user/hxs/praxis-vla/serving-artifacts/robotwin-bar-bin-qwen3vl2b-h32-final-ema-step10095-robot-mount-v1/pretrained_model
RDZ_ROOT=/robot/robot-research-exp-0/user/hxs/praxis-vla/outputs/robotwin2_clean/_rendezvous/$RUN_ID
BASE_PORT=50100; NODE_COUNT=2; GPU_COUNT=8; SERVER_TIMEOUT_SEC=86400; HEALTH_TIMEOUT_SEC=1800
NODE_RANK=${MLP_ROLE_INDEX:-${ARNOLD_ID:-${NODE_RANK:-${RANK:-0}}}}
NODE_RANK=$((10#$NODE_RANK)); test "$NODE_RANK" -lt "$NODE_COUNT"
test -x "$POLICY_PYTHON"; test -f "$POLICY_SERVER"; test -f "$CHECKPOINT/config.json"; test -f "$CHECKPOINT/model.safetensors"
mkdir -p "$RDZ_ROOT"
host=$(hostname -I | awk '{print $1}')
case "$host" in ""|127.*) echo "Cannot determine routable server host: $host"; exit 2 ;; esac
if [ "$NODE_RANK" = 0 ]; then
  generation="${CLOUDML_JOB_ID:-robotwin-server}-${RUN_ID}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
  printf '%s\n' "$generation" >"$RDZ_ROOT/server_generation.tmp"; mv "$RDZ_ROOT/server_generation.tmp" "$RDZ_ROOT/server_generation"
fi
deadline=$(($(date +%s) + SERVER_TIMEOUT_SEC))
while [ ! -s "$RDZ_ROOT/server_generation" ]; do [ "$(date +%s)" -lt "$deadline" ] || exit 1; sleep 2; done
generation=$(tr -d '[:space:]' <"$RDZ_ROOT/server_generation")
endpoint_file="$RDZ_ROOT/server-${NODE_RANK}.endpoint"
ready_file="$RDZ_ROOT/server-${NODE_RANK}.ready"
rm -f "$endpoint_file" "$ready_file"
printf '%s %s %s\n' "$generation" "$host" "$((BASE_PORT + NODE_RANK * GPU_COUNT))" >"$endpoint_file"
PIDS=()
cleanup() { for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done; wait 2>/dev/null || true; }
trap cleanup EXIT INT TERM
for gpu in $(seq 0 $((GPU_COUNT - 1))); do
  port=$((BASE_PORT + NODE_RANK * GPU_COUNT + gpu))
  CUDA_VISIBLE_DEVICES=$gpu "$POLICY_PYTHON" -u "$POLICY_SERVER" policy=bar_bin_qwen policy.pretrained_path="$CHECKPOINT" host=0.0.0.0 port="$port" device=cuda >"$RDZ_ROOT/server-${NODE_RANK}-${gpu}.log" 2>&1 &
  PIDS+=("$!")
done
health_deadline=$(($(date +%s) + HEALTH_TIMEOUT_SEC))
for gpu in $(seq 0 $((GPU_COUNT - 1))); do
  port=$((BASE_PORT + NODE_RANK * GPU_COUNT + gpu))
  "$POLICY_PYTHON" - "$host" "$port" "$CHECKPOINT" "$health_deadline" <<'PY'
import json, sys, time
from pathlib import Path
from praxis_remote import PolicyClient
host, port, expected = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]).resolve()
for _ in range(max(1, int((int(sys.argv[4]) - int(time.time())) / 2))):
    client = PolicyClient(host=host, port=port, timeout=5.0)
    try:
        try:
            ready, info = client.health_check()
            if ready and Path(json.loads(info)["serving_artifact"]["checkpoint"]["path"]).resolve() == expected:
                print(f"ROBOTWIN_POLICY_SERVER_READY host={host} port={port}", flush=True); break
        except Exception: pass
    finally: client.close()
    time.sleep(2)
else: raise RuntimeError(f"Policy server did not become ready: {host}:{port}")
PY
  if [ "$(date +%s)" -ge "$health_deadline" ]; then
    echo "Policy server health timeout node=$NODE_RANK gpu=$gpu host=$host port=$port" >&2
    exit 1
  fi
done
printf '%s\n' "$generation" >"$ready_file"
echo "ROBOTWIN_POLICY_SERVER_NODE_READY node=$NODE_RANK generation=$generation host=$host"
while :; do
  if [ -s "$RDZ_ROOT/client_complete" ] && [ "$(tr -d '[:space:]' <"$RDZ_ROOT/client_complete")" = "$generation" ]; then exit 0; fi
  [ "$(date +%s)" -lt "$deadline" ] || { echo "Timed out waiting for client completion"; exit 1; }
  sleep 10
done
