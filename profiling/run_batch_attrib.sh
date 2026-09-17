#!/usr/bin/env bash
# Batch-gate attribution: is the concurrent-vs-sequential token mismatch caused by the shipped
# slice-height default (48) or pre-existing on the pre-change geometry (64)?
#   run_batch_attrib.sh
# Runs correctness_gate.py batch three times:
#   1. perf-a, default (pre-change: 64 in both sq and msq launchers)
#   2. perf-a, EXL3_SQ_ROWS_PER=48 (the knob, which both launchers read)
#   3. perf-c, default (shipping candidate: 48 in both)
set -uo pipefail
BASE=/home/ghazni/github/exllamav3-rocm/profiling
OUT=$BASE/out_a
C="--rm --device /dev/kfd --device /dev/dri --group-add 44 --group-add 993 --ipc host \
  --shm-size 4g --ulimit nofile=65536:65536 \
  -e EXL3_MODEL=/models/Qwen3.8-27B-SC_4.00bpw_H5_V6 -e EXL3_CACHE_TOKENS=32768 \
  -v /home/ghazni/models/exl3/turboderp:/models:ro -v $BASE:/prof:ro -v $OUT:/out"

: > $OUT/s5_batch_attrib.txt
run() {   # run <tag> <image> [ENV=VAL ...]
    local tag=$1 img=$2; shift 2
    local envs=(); for kv in "$@"; do envs+=(-e "$kv"); done
    echo "--- $tag ($img ${*:-default})" | tee -a $OUT/s5_batch_attrib.txt
    docker run $C --name exl3-ba "${envs[@]}" --entrypoint python3 "$img" \
      /prof/correctness_gate.py batch /out/batch_${tag}.json 2>&1 | tail -4 \
      | tee -a $OUT/s5_batch_attrib.txt
}

run perf_a_default   exllamav3-rocm:perf-a
run perf_a_rp48      exllamav3-rocm:perf-a EXL3_SQ_ROWS_PER=48
run perf_c_default   exllamav3-rocm:perf-c
echo "=== batch attribution complete ==="
