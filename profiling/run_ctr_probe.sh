#!/usr/bin/env bash
# Counter discovery on gfx1100 (SDK rocprofiler-sdk 1.3.5):
# which memory-system counters actually return nonzero through the official CLI.
# GL2C_EA_*, FETCH_SIZE, GRBM_* are known dead (docs/rocprofv3-findings-log.md §3);
# this probe tries the untried GL2C_MC_*, WRITE_SIZE, SQ_INST_CYCLES_VMEM,
# TA_BUFFER_LOAD_WAVEFRONTS and the GRBM-derived GPUBusy/GPU_UTIL/L2CacheHit.
set -uo pipefail
SHIMS=/home/ghazni/github/exllamav3-rocm/profiling/shims
OUT=${1:-/home/ghazni/github/exllamav3-rocm/profiling/out_ctr_probe}
rm -rf "$OUT"; mkdir -p "$OUT"

run_set () {
  local NAME="$1"; shift
  echo "=== $NAME: $* ==="
  docker run --rm --name exl3-ctr-$NAME \
    --device /dev/kfd --device /dev/dri --group-add 44 --group-add 993 \
    --ipc host --shm-size 4g --ulimit nofile=65536:65536 --cap-add=PERFMON \
    -e LD_PRELOAD=/shims/ld_scope_shim.so \
    -v "$SHIMS":/shims:ro -v "$OUT":/out \
    -v /home/ghazni/github/exllamav3-rocm/profiling/counter_probe.py:/probe/counter_probe.py:ro \
    --entrypoint rocprofv3 tabbyapi-rocm:serve \
    --kernel-trace --pmc "$@" -f csv -d /out \
    -- python3 /probe/counter_probe.py 2>&1 | grep -Ei 'probe-done|error|reject|fall|cannot|unable' | head -5
  for f in "$OUT"/*/*counter_collection*.csv; do
    [ -f "$f" ] || continue
    echo "--- $(basename $(dirname $f))/$(basename $f) nonzero by counter:"
    awk -F, 'NR>1{for(i=1;i<=NF;i++) if($i+0!=0) nz[i]++} END{for(i in nz) print "  col",i":",nz[i],"nonzero rows"}' "$f" | head -8
    rm -f "$f"
  done
}

run_set mc    "GL2C_MC_RDREQ_sum GL2C_MC_WRREQ_sum GL2C_EA_RDREQ_128B_sum GL2C_HIT_sum GL2C_MISS_sum"
run_set deriv "FETCH_SIZE WRITE_SIZE GPUBusy GPU_UTIL L2CacheHit"
run_set sqta  "SQ_WAVES SQ_INST_CYCLES_VMEM TA_BUFFER_LOAD_WAVEFRONTS SQ_INSTS_FLAT"
run_set all1  "GL2C_MC_RDREQ_sum GL2C_MC_WRREQ_sum SQ_WAVES SQ_INST_CYCLES_VMEM TA_BUFFER_LOAD_WAVEFRONTS"
echo "done"
