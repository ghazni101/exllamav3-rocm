#!/usr/bin/env bash
# Phase A round 2: cold-protocol re-baseline + candidate confirmations, one config per process,
# pool >= 6x the 96 MB Infinity Cache (AB_ROT=20, clamped to the instances available per K).
#   run_a3_round2.sh <image>
set -uo pipefail
BASE=/home/ghazni/github/exllamav3-rocm/profiling
OUT=$BASE/out_a
IMG=${1:-exllamav3-rocm:perf-a}

one() {
    local tag=$1; shift
    local envs=()
    for kv in "$@"; do envs+=(-e "$kv"); done
    docker run --rm --name exl3-s1 --device /dev/kfd --device /dev/dri \
      --group-add 44 --group-add 993 --ipc host --shm-size 4g --ulimit nofile=65536:65536 \
      -e EXL3_MODEL=/models/Qwen3.8-27B-SC_4.00bpw_H5_V6 -e EXL3_CACHE_TOKENS=32768 \
      -e AB_DIR=/out -e AB_ROT=20 -e EXL3_INT8_GEMV_MAX_K=6 -e AB_TAG="$tag" "${envs[@]}" \
      -v /home/ghazni/models/exl3/turboderp:/models:ro -v $BASE:/prof:ro -v $OUT:/out \
      --entrypoint python3 "$IMG" /prof/sq_sweep1.py 2>&1 | grep -E "^JSON|Traceback|Error" || true
}

rm -f $OUT/a3_procs2.json
export OUT_JSON=a3_procs2   # documented in the probe (currently fixed name a3_procs.json)

# baseline, twice per (K,m) to expose run-to-run spread under the new protocol
for rep in 1 2; do
  for K in 3 4 5; do
    one r${rep}_base_K${K}_m1 AB_K=$K AB_M=1
    one r${rep}_base_K${K}_m4 AB_K=$K AB_M=4
  done
done

# candidate 1: slice height 48 (default is 64 on ROCm)
for rep in 1 2; do
  for K in 3 4 5; do
    one r${rep}_rp48_K${K}_m1 AB_K=$K AB_M=1 EXL3_SQ_ROWS_PER=48
  done
done

# candidate 2: narrow instead of smem-staged for the staged K (3/5) - the RDNA routing question
for rep in 1 2; do
  for K in 3 5; do
    one r${rep}_stage0_K${K}_m1 AB_K=$K AB_M=1 EXL3_SQ_STAGE_SMEM=0 EXL3_SQ_ROWS_PER=48
  done
done

# candidate 1+2 combined
for rep in 1 2; do
  for K in 3 5; do
    one r${rep}_rp48stage0_K${K}_m1 AB_K=$K AB_M=1 EXL3_SQ_ROWS_PER=48 EXL3_SQ_STAGE_SMEM=0
  done
done

echo "=== round 2 complete ==="
python3 - "$OUT/a3_procs.json" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
print(f"{'tag':24s} {'K':>2s} {'m':>2s} {'poolMB':>7s} {'xIC':>5s} {'cold_med':>9s} {'warm_med':>9s} {'GB/s':>7s} {'GB/s_w':>7s} ret")
for r in rows:
    print(f"{r['tag']:24s} {r['K']:2d} {r['m']:2d} {r['pool_mb']:7.0f} {r['pool_over_ic']:5.1f} "
          f"{r['cold_med_ms']:9.4f} {r['warm_med_ms']:9.4f} {r['gbps']:7.1f} {r['gbps_warm']:7.1f} {r['ret']}")
PY
