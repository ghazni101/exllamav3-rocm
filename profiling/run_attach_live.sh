#!/usr/bin/env bash
# Definitive live-serve attach profile. Recipe (established 2026-09-16):
#   1. serve must run with ROCP_TOOL_ATTACH=1 + LD_PRELOAD=librocprofiler-register.so (+SYS_PTRACE, nofile 65536)
#   2. target must be QUIESCENT when the attach lands (attach under load hangs)
#   3. use rocprof-attach DIRECTLY (rocprofv3 --attach wrapper hangs even when idle)
#   4. tool config travels via the client's env (ROCPROF_KERNEL_TRACE etc.)
#   5. drive HTTP traffic only AFTER the attach has been established
set -uo pipefail
BASE=/home/ghazni/github/exllamav3-rocm/profiling
OUT=$BASE/out_attach_live
rm -rf "$OUT"; mkdir -p "$OUT"

echo "=== waiting for serve to go fully idle (post-recreate health + settle) ==="
for i in $(seq 1 40); do
  H=$(curl -s -o /dev/null -w '%{http_code}' http://192.168.1.200:9001/v1/model/list 2>/dev/null)
  [ "$H" = "200" ] && break; sleep 15
done
sleep 10

echo "=== attach while idle (60 s window, kernel trace) ==="
docker exec exllamav3-rocm-serve sh -c "rm -rf /tmp/prof; mkdir -p /tmp/prof"
docker exec exllamav3-rocm-serve sh -c "
timeout 240 env ROCPROFILER_LOG_LEVEL=info ROCPROF_KERNEL_TRACE=1 \
  ROCPROF_OUTPUT_PATH=/tmp/prof ROCPROF_OUTPUT_FORMAT=csv \
  python3 -u /opt/rocm-venv/lib/python3.12/site-packages/_rocm_sdk_devel/bin/rocprof-attach \
  -p 1 --attach-children=false \
  -t /opt/rocm-venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so \
  -d 60000 > /tmp/attach_live.log 2>&1; echo attach-rc=\$?" &
ATT=$!
# wait until the attach is established (client prints 'Attaching for 60000 msec')
for i in $(seq 1 30); do
  if docker exec exllamav3-rocm-serve sh -c 'grep -q "Attaching for 60000" /tmp/attach_live.log 2>/dev/null'; then
    echo "attach established after ~$((i*3))s"; break
  fi; sleep 3
done

echo "=== driving HTTP load inside the window ==="
python3 "$BASE/ttft_probe.py" attached > /tmp/ttft_attached.log 2>&1 &
PROBE=$!
wait $PROBE
wait $ATT; RC=$?
echo "attach rc=$RC"
docker exec exllamav3-rocm-serve sh -c 'grep -E "success|ERROR|Detaching" /tmp/attach_live.log | head -6; ls -la /tmp/prof'
docker cp exllamav3-rocm-serve:/tmp/prof/. "$OUT"/ 2>/dev/null
sudo chmod -R a+rX "$OUT" 2>/dev/null
find "$OUT" -type f -name '*.csv' -exec ls -la {} \;
cat /tmp/ttft_*.json 2>/dev/null | head -5
