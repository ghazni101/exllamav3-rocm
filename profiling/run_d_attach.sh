#!/usr/bin/env bash
# Experiment D: attach-profile the LIVE serve (TabbyAPI layer included).
# Assumes the serve container was recreated with the attach override
# (ROCP_TOOL_ATTACH=1 + LD_PRELOAD=librocprofiler-register.so + SYS_PTRACE).
# Phase 1: HTTP TTFT probe with NO profiler attached (clean request-path numbers).
# Phase 2: quiescent-target attach test (8 s window while idle) — the open question
#          from docs/rocprofv3-findings-log.md §2.
# Phase 3: if phase 2 produced output, attach 45 s while driving HTTP traffic.
set -uo pipefail
BASE=/home/ghazni/github/exllamav3-rocm/profiling
OUT=$BASE/out_attach
rm -rf "$OUT"; mkdir -p "$OUT"

echo "=== [phase1] TTFT probe, no profiler ==="
python3 "$BASE/ttft_probe.py" clean 2>&1 | grep -E '^\[' | head -12

echo "=== [phase2] quiescent attach test ==="
docker exec exllamav3-rocm-serve sh -c 'rm -rf /tmp/prof; mkdir -p /tmp/prof' 2>/dev/null
docker exec exllamav3-rocm-serve sh -c \
  "timeout 240 rocprofv3 --attach 1 --attach-children=false --attach-duration-msec 8000 \
     --attach-sync-output --kernel-trace -f csv -d /tmp/prof > /tmp/attach_q.log 2>&1; echo rc=\$?" &
ATT=$!
sleep 90   # well past the 8 s window; detach can take 1-2 min
wait $ATT; RC=$?
echo "attach rc=$RC"
docker exec exllamav3-rocm-serve sh -c 'cat /tmp/attach_q.log; echo ---; ls -la /tmp/prof' | tail -15
QOK=$(docker exec exllamav3-rocm-serve sh -c 'ls /tmp/prof/*kernel_trace* 2>/dev/null | wc -l')
echo "quiescent kernel trace files: $QOK"

if [ "$QOK" != "0" ]; then
  echo "=== [phase3] attach under HTTP load (45 s) ==="
  docker exec exllamav3-rocm-serve sh -c 'rm -rf /tmp/prof; mkdir -p /tmp/prof'
  docker exec exllamav3-rocm-serve sh -c \
    "timeout 300 rocprofv3 --attach 1 --attach-children=false --attach-duration-msec 45000 \
       --kernel-trace -f csv -d /tmp/prof > /tmp/attach_l.log 2>&1; echo rc=\$?" &
  ATT=$!
  sleep 12
  python3 "$BASE/ttft_probe.py" attached 2>&1 | grep -E '^\[' | head -12
  wait $ATT; RC=$?
  echo "attach rc=$RC"
  docker exec exllamav3-rocm-serve sh -c 'tail -12 /tmp/attach_l.log; echo ---; ls -la /tmp/prof'
  docker cp exllamav3-rocm-serve:/tmp/prof/. "$OUT"/ 2>/dev/null
  sudo chmod -R a+rX "$OUT" 2>/dev/null
fi
echo "=== done ==="
