#!/usr/bin/env bash
# Experiment B: (1) full-model CLI kernel trace of bench_rocm.py with the known-good
# flag set (--kernel-trace only); (2) flag isolation on short profile_rocm.py runs;
# (3) HIP graph-launch coverage via --hip-graph-trace. One lock hold, serial containers.
set -uo pipefail
SHIMS=/home/ghazni/github/exllamav3-rocm/profiling/shims
BASE=/home/ghazni/github/exllamav3-rocm/profiling
OUT=$BASE/out_b
rm -rf "$OUT"; mkdir -p "$OUT"

run() {  # run <name> <extra flags...> -- app args via env
  local NAME="$1"; shift
  echo "=== [$NAME] $* ==="
  docker run --rm --name exl3-b-$NAME \
    --device /dev/kfd --device /dev/dri --group-add 44 --group-add 993 \
    --ipc host --shm-size 4g --ulimit nofile=65536:65536 --cap-add=PERFMON \
    -e EXL3_MODEL=/models/Qwen3.8-27B-SC_4.00bpw_H5_V6 \
    -e EXL3_CACHE_TOKENS=${CACHE:-32768} -e EXL3_GEN_TOKENS=${GEN:-128} \
    -e EXL3_WARMUP_TOKENS=${WARMUP:-8} -e EXL3_DECODE_TOKENS=${DECODE:-64} \
    -e EXL3_PREFILL_TOKENS=${PREFILL:-0} \
    -e HIPFIRE_KERNEL_CACHE=/var/cache/hipfire -e HIPFIRE_DIR=/root/.hipfire \
    -e LD_PRELOAD=/shims/ld_scope_shim.so \
    -v "$SHIMS":/shims:ro -v /home/ghazni/models/exl3/turboderp:/models:ro \
    -v exllamav3-kcache:/var/cache/hipfire -v "$OUT/$NAME":/out \
    --entrypoint rocprofv3 tabbyapi-rocm:serve \
    "$@" -f csv -d /out \
    -- python3 "$APP" 2>&1 | grep -Ei '^\[profile\]|^\[load\]|decode|prefill|error|signal' | head -8
  sudo chmod -R a+rX "$OUT/$NAME" 2>/dev/null
}

# 0) SQ/derived block mapping on the cheap probe
echo "=== [sqmap] SQ block census on probe ==="
docker run --rm --name exl3-b-sqmap \
  --device /dev/kfd --device /dev/dri --group-add 44 --group-add 993 \
  --ipc host --shm-size 4g --ulimit nofile=65536:65536 --cap-add=PERFMON \
  -e LD_PRELOAD=/shims/ld_scope_shim.so \
  -v "$SHIMS":/shims:ro -v "$OUT/sqmap":/out \
  -v "$BASE/counter_probe.py":/probe/counter_probe.py:ro \
  --entrypoint rocprofv3 tabbyapi-rocm:serve \
  --kernel-trace --pmc "SQ_WAVES SQ_BUSY_CYCLES SQ_WAVE_CYCLES SQ_INSTS_VALU SQ_INSTS_SMEM SFetchInsts Wavefronts MemUnitBusy ALUStalledByLDS" \
  -f csv -d /out -- python3 /probe/counter_probe.py 2>&1 | grep -Ei 'probe-done|error code|signal' | head -4
sudo chmod -R a+rX "$OUT/sqmap" 2>/dev/null

# 1) full model, kernel trace only (known good), the primary analysis artifact
APP=/opt/exllamav3/bench_rocm.py WARMUP=8 GEN=128 \
run trace_main --kernel-trace

# 2..4) flag isolation, short runs
APP=/opt/exllamav3/profile_rocm.py DECODE=64 PREFILL=0 \
run iso_memcopy   --kernel-trace --memory-copy-trace
APP=/opt/exllamav3/profile_rocm.py DECODE=64 PREFILL=0 \
run iso_hiprt     --kernel-trace --hip-runtime-trace
APP=/opt/exllamav3/profile_rocm.py DECODE=64 PREFILL=0 \
run iso_hiprt_only --hip-runtime-trace

# 5) graph-launch coverage
APP=/opt/exllamav3/profile_rocm.py DECODE=64 PREFILL=0 \
run graph_cov     --kernel-trace --hip-graph-trace

echo "=== outputs ==="
find "$OUT" -name '*.csv' -o -name '*.json' | sort
