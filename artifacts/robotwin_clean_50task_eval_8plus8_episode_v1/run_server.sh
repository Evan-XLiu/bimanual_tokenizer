#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID=robotwin-ar-bin-qwen3vl2b-h32-final-ema-step10095-demo-clean-50x20-s0-h20-8plus8-24sim-episode-20260814-v1
SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
POLICY_ENV=/robot/robot-research-exp-0/user/hxs/envs/praxis-vla-robotwin-policy-py310-v1
POLICY_PYTHON=$POLICY_ENV/bin/python
POLICY_SERVER=$POLICY_ENV/lib/python3.10/site-packages/praxis/scripts/serve.py
CHECKPOINT=/robot/robot-research-exp-0/user/hxs/praxis-vla/serving-artifacts/robotwin-bar-bin-qwen3vl2b-h32-final-ema-step10095-robot-mount-v1/pretrained_model
RDZ_ROOT=/robot/robot-research-exp-0/user/hxs/praxis-vla/outputs/robotwin2_clean/_rendezvous/$RUN_ID
BASE_PORT=50100
GPU_COUNT=8
TIMEOUT_SEC=86400

test -x "$POLICY_PYTHON"
test -f "$POLICY_SERVER"
test -f "$CHECKPOINT/config.json"
test -f "$CHECKPOINT/model.safetensors"
mkdir -p "$RDZ_ROOT/logs"

namespace=$(cat /var/run/secrets/kubernetes.io/serviceaccount/namespace)
host="${HOSTNAME}.${namespace}.svc.cluster.local"
host_ip=$(getent ahostsv4 "$host" | awk '$1 !~ /^127\./ { print $1; exit }')
test -n "$host_ip"
generation="${CLOUDML_JOB_ID:-robotwin-server}-${RUN_ID}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
rm -f "$RDZ_ROOT/server.endpoint" "$RDZ_ROOT/server.ready"
printf '%s\n' "$generation" >"$RDZ_ROOT/server_generation.tmp"
mv "$RDZ_ROOT/server_generation.tmp" "$RDZ_ROOT/server_generation"
printf '%s %s %s\n' "$generation" "$host" "$BASE_PORT" >"$RDZ_ROOT/server.endpoint.tmp"
mv "$RDZ_ROOT/server.endpoint.tmp" "$RDZ_ROOT/server.endpoint"
echo "ROBOTWIN_POLICY_SERVER_ENDPOINT generation=$generation host=$host ip=$host_ip base_port=$BASE_PORT count=$GPU_COUNT"

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for gpu in $(seq 0 $((GPU_COUNT - 1))); do
  port=$((BASE_PORT + gpu))
  CUDA_VISIBLE_DEVICES=$gpu "$POLICY_PYTHON" -u "$POLICY_SERVER" \
    policy=bar_bin_qwen \
    policy.pretrained_path="$CHECKPOINT" \
    host=0.0.0.0 \
    port="$port" \
    device=cuda \
    2>&1 | sed -u "s/^/[policy-$gpu] /" | tee -a "$RDZ_ROOT/logs/policy-$gpu.log" &
  PIDS+=("$!")
done

health_deadline=$(($(date +%s) + 1800))
for gpu in $(seq 0 $((GPU_COUNT - 1))); do
  port=$((BASE_PORT + gpu))
  "$POLICY_PYTHON" - "$host" "$port" "$CHECKPOINT" "$health_deadline" <<'PY'
import json
import sys
import time
from pathlib import Path

from praxis_remote import PolicyClient

host, port = sys.argv[1], int(sys.argv[2])
expected = Path(sys.argv[3]).resolve()
deadline = int(sys.argv[4])
while time.time() < deadline:
    client = PolicyClient(host=host, port=port, timeout=5.0)
    try:
        ready, info = client.health_check()
        if ready:
            loaded = Path(json.loads(info)["serving_artifact"]["checkpoint"]["path"]).resolve()
            if loaded == expected:
                print(f"ROBOTWIN_POLICY_SERVER_READY host={host} port={port}", flush=True)
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

printf '%s\n' "$generation" >"$RDZ_ROOT/server.ready.tmp"
mv "$RDZ_ROOT/server.ready.tmp" "$RDZ_ROOT/server.ready"
echo "ROBOTWIN_POLICY_SERVER_ALL_READY generation=$generation endpoints=$GPU_COUNT source=$SOURCE_ROOT"

deadline=$(($(date +%s) + TIMEOUT_SEC))
while :; do
  if [ -s "$RDZ_ROOT/client_complete" ] && [ "$(tr -d '[:space:]' <"$RDZ_ROOT/client_complete")" = "$generation" ]; then
    echo "ROBOTWIN_POLICY_SERVER_CLIENT_COMPLETE generation=$generation"
    exit 0
  fi
  for pid in "${PIDS[@]}"; do
    kill -0 "$pid" 2>/dev/null || { echo "A policy server exited before client completion" >&2; exit 1; }
  done
  [ "$(date +%s)" -lt "$deadline" ] || { echo "Timed out waiting for client completion" >&2; exit 1; }
  sleep 10
done
