#!/usr/bin/env bash
# A3/B1.2 sweep driver: one configuration per process (see profiling/sq_sweep1.py for why).
#   run_a3_matrix.sh <image> [out_json_name]
# Result rows append to profiling/out_a/a3_procs.json.
set -uo pipefail
BASE=/home/ghazni/github/exllamav3-rocm/profiling
OUT=$BASE/out_a
IMG=${1:-exllamav3-rocm:perf-a}

one() {   # one <tag> NAME=VAL ...
    local tag=$1; shift
    local envs=()
    for kv in "$@"; do envs+=(-e "$kv"); done
    docker run --rm --name exl3-s1 --device /dev/kfd --device /dev/dri \
      --group-add 44 --group-add 993 --ipc host --shm-size 4g --ulimit nofile=65536:65536 \
      -e EXL3_MODEL=/models/Qwen3.8-27B-SC_4.00bpw_H5_V6 -e EXL3_CACHE_TOKENS=32768 \
      -e AB_DIR=/out -e AB_ROT=6 -e AB_REPS=6 -e EXL3_INT8_GEMV_MAX_K=6 \
      -e AB_TAG="$tag" "${envs[@]}" \
      -v /home/ghazni/models/exl3/turboderp:/models:ro -v $BASE:/prof:ro -v $OUT:/out \
      --entrypoint python3 "$IMG" /prof/sq_sweep1.py 2>&1 | grep -E "^JSON|Traceback|Error" || true
}

rm -f $OUT/a3_procs.json

# 1. default geometry: does one-config-per-process reproduce the default arm?
one base_K3 AB_K=3 AB_M=1
one base_K4 AB_K=4 AB_M=1
one base_K5 AB_K=5 AB_M=1
one base_K3_m4 AB_K=3 AB_M=4
one base_K4_m4 AB_K=4 AB_M=4
one base_K5_m4 AB_K=5 AB_M=4

# 2. grid multiplier (same slice height)
for sms in 24 48 64 96 128 192; do
  one g${sms}_K3_rp64 AB_K=3 AB_M=1 AB_SMS=$sms EXL3_SQ_ROWS_PER=64
  one g${sms}_K4_rp64 AB_K=4 AB_M=1 AB_SMS=$sms EXL3_SQ_ROWS_PER=64
done

# 3. slice height at the best grid candidates
for rp in 32 48 96 128; do
  one rp${rp}_K3 AB_K=3 AB_M=1 AB_SMS=48 EXL3_SQ_ROWS_PER=$rp
  one rp${rp}_K4 AB_K=4 AB_M=1 AB_SMS=48 EXL3_SQ_ROWS_PER=$rp
done

# 4. B1.2 staging routing A/B (K=3 and K=5 are the smem-routed ones)
for st in 0 1; do
  one stage${st}_K3 AB_K=3 AB_M=1 AB_SMS=48 EXL3_SQ_ROWS_PER=64 EXL3_SQ_STAGE_SMEM=$st
  one stage${st}_K5 AB_K=5 AB_M=1 AB_SMS=48 EXL3_SQ_ROWS_PER=64 EXL3_SQ_STAGE_SMEM=$st
done

echo "=== matrix complete: $OUT/a3_procs.json ==="
