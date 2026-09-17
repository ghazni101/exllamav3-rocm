#!/usr/bin/env bash
# End-to-end A/B of the sq-path config candidates on ONE image (env-controlled, so no rebuild and
# no cross-image codegen drift). bench_lean is the gate: decode b1/b4/b8 + cold prefill 2k.
#   run_b1_config_ab.sh <image>
set -uo pipefail
BASE=/home/ghazni/github/exllamav3-rocm/profiling
OUT=$BASE/out_a
IMG=${1:-exllamav3-rocm:perf-a}

one() {   # one <tag> ENV=VAL ...
    local tag=$1; shift
    local envs=()
    for kv in "$@"; do envs+=(-e "$kv"); done
    echo "--- bench_lean [$tag] ${*:-default}"
    docker run --rm --name exl3-bl --device /dev/kfd --device /dev/dri \
      --group-add 44 --group-add 993 --ipc host --shm-size 4g --ulimit nofile=65536:65536 \
      -e EXL3_MODEL=/models/Qwen3.8-27B-SC_4.00bpw_H5_V6 -e EXL3_CACHE_TOKENS=32768 \
      "${envs[@]}" \
      -v /home/ghazni/models/exl3/turboderp:/models:ro -v $BASE:/prof:ro -v $OUT:/out \
      --entrypoint python3 "$IMG" /prof/bench_lean.py 2>&1 | tail -2
}

: > $OUT/b1_config_ab.txt
for tag_envs in "default" "rp48 EXL3_SQ_ROWS_PER=48" "stage0 EXL3_SQ_STAGE_SMEM=0" \
                "rp48_stage0 EXL3_SQ_ROWS_PER=48 EXL3_SQ_STAGE_SMEM=0"; do
    set -- $tag_envs
    tag=$1; shift
    one "$tag" "$@" | tee -a $OUT/b1_config_ab.txt
done
echo "=== b1 config A/B complete ==="
