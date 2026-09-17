#!/usr/bin/env bash
# Phase A GPU session: A2 (per-K/m cold rates), A3 (rows_per x grid-multiplier sweep) and the
# B1.2 staging A/B, all on ONE image with the standing serve stopped.
#
#   run_a_probe.sh <image>          (default exllamav3-rocm:perf-a)
#
# Outputs into profiling/out_a/ (mounted as /out).
set -uo pipefail
BASE=/home/ghazni/github/exllamav3-rocm/profiling
OUT=$BASE/out_a
IMG=${1:-exllamav3-rocm:perf-a}

run_py() {   # run_py <script> [NAME=VAL ...]
    local script=$1; shift
    local envs=()
    for kv in "$@"; do envs+=(-e "$kv"); done
    docker run --rm --name exl3-a --device /dev/kfd --device /dev/dri \
      --group-add 44 --group-add 993 --ipc host --shm-size 4g --ulimit nofile=65536:65536 \
      -e EXL3_MODEL=/models/Qwen3.8-27B-SC_4.00bpw_H5_V6 -e EXL3_CACHE_TOKENS=32768 \
      -e AB_DIR=/out -e AB_ROT=6 -e AB_REPS=6 -e EXL3_INT8_GEMV_MAX_K=6 "${envs[@]}" \
      -v /home/ghazni/models/exl3/turboderp:/models:ro -v $BASE:/prof:ro -v $OUT:/out \
      --entrypoint python3 "$IMG" "$script"
}

echo "=== image guard ==="
docker run --rm --entrypoint bash "$IMG" -lc \
  'SO=$(find /opt/rocm-venv -name "exllamav3_ext*.so" | head -1); echo "ext: $SO"; strings "$SO" | grep -c "msq-launch"' \
  || { echo "image missing the patched route - void"; exit 1; }

echo "=== A2: per-K/m cold rates (K=3,4,5,6 x m=1,2,4 + K=4 IC control) ==="
: > $OUT/a2_probe3.txt
run_py /prof/gemv_probe3.py 2>&1 | tee -a $OUT/a2_probe3.txt \
  | grep -E "^==|^   m=|^K=|wrote|Traceback|Error" || true
grep -q "wrote a2_probe3.json" $OUT/a2_probe3.txt || { echo "A2 FAILED"; exit 1; }

echo "=== A3: rows_per x grid-multiplier sweep + B1.2 staging A/B ==="
: > $OUT/a3_sweep.txt
sweep() {   # sweep <tag> <env...>
  local tag=$1; shift
  local envs=()
  for kv in "$@"; do envs+=(-e "$kv"); done
  echo "--- $tag" | tee -a $OUT/a3_sweep.txt
  run_py /prof/gemv_sweep.py SWEEP_TAG="$tag" "${envs[@]}" 2>&1 | tee -a $OUT/a3_sweep.txt \
    | grep -E "^==|^   m=|wrote|Traceback|Error" || true
}

# grid multiplier sweep at the pinned slice heights (one process per rows_per value)
for rp in 32 64 96; do
  sweep "a3r${rp}" SWEEP_K=3,4,5 SWEEP_M=1,4 SWEEP_SMS=0,48,96,192,384 EXL3_SQ_ROWS_PER=$rp
done

# B1.2: the smem-vs-narrow routing decision on RDNA (K=3 and K=5 are the smem-routed ones)
for stage in 0 1; do
  sweep "a3stage${stage}" SWEEP_K=3,5 SWEEP_M=1,4 SWEEP_SMS=0,96 EXL3_SQ_STAGE_SMEM=$stage \
    EXL3_SQ_ROWS_PER=64
done

echo "=== phase A session complete ==="
