#!/usr/bin/env bash
# Phase B session: B2 (wide-unit prefetch depth) A/B - cold kernel bench + end-to-end gate -
# plus the B4 fp16-GEMV re-test. One image, every arm env-selected.
#   run_b2_session.sh <image>
set -uo pipefail
BASE=/home/ghazni/github/exllamav3-rocm/profiling
OUT=$BASE/out_a
IMG=${1:-exllamav3-rocm:perf-b}

drun() {   # drun <script> NAME=VAL ...
    local script=$1; shift
    local envs=()
    for kv in "$@"; do envs+=(-e "$kv"); done
    docker run --rm --name exl3-b2 --device /dev/kfd --device /dev/dri \
      --group-add 44 --group-add 993 --ipc host --shm-size 4g --ulimit nofile=65536:65536 \
      -e EXL3_MODEL=/models/Qwen3.8-27B-SC_4.00bpw_H5_V6 -e EXL3_CACHE_TOKENS=32768 \
      -e AB_DIR=/out -e AB_ROT=20 -e EXL3_INT8_GEMV_MAX_K=6 "${envs[@]}" \
      -v /home/ghazni/models/exl3/turboderp:/models:ro -v $BASE:/prof:ro -v $OUT:/out \
      --entrypoint python3 "$IMG" "$script"
}

echo "=== image guard ==="
docker run --rm --entrypoint bash "$IMG" -lc \
  'SO=$(find /opt/rocm-venv -name "exllamav3_ext*.so" | head -1); strings "$SO" | grep -c "msq-launch"; strings "$SO" | grep -c "EXL3_SQ_PF"' \
  || { echo "void"; exit 1; }

echo "=== B2 cold kernel A/B (K=4 m=1, (5120,17408), pool 891 MB, one config per process) ==="
: > $OUT/b2_pf.txt
for rep in 1 2; do
  for pf in 3 5; do
    drun /prof/sq_sweep1.py AB_TAG=pf${pf}_r${rep} AB_K=4 AB_M=1 AB_SMS=0 EXL3_SQ_PF=$pf 2>&1 \
      | tee -a $OUT/b2_pf.txt | grep -E "^JSON|Traceback" || true
  done
done

echo "=== end-to-end: default (rows48+PF3) x2, PF=5 x2 ==="
: > $OUT/b2_e2e.txt
for arm in "base1" "base2" "pf5_a EXL3_SQ_PF=5" "pf5_b EXL3_SQ_PF=5"; do
  set -- $arm; tag=$1; shift
  envs=(); for kv in "$@"; do envs+=(-e "$kv"); done
  echo "--- bench_lean [$tag]" | tee -a $OUT/b2_e2e.txt
  docker run --rm --name exl3-bl --device /dev/kfd --device /dev/dri \
    --group-add 44 --group-add 993 --ipc host --shm-size 4g --ulimit nofile=65536:65536 \
    -e EXL3_MODEL=/models/Qwen3.8-27B-SC_4.00bpw_H5_V6 -e EXL3_CACHE_TOKENS=32768 "${envs[@]}" \
    -v /home/ghazni/models/exl3/turboderp:/models:ro -v $BASE:/prof:ro -v $OUT:/out \
    --entrypoint python3 "$IMG" /prof/bench_lean.py 2>&1 | tail -1 | tee -a $OUT/b2_e2e.txt
done

echo "=== B4: fp16 QTIP GEMV re-test (EXL3_INT8_GEMV=0) ==="
docker run --rm --name exl3-b4 --device /dev/kfd --device /dev/dri \
  --group-add 44 --group-add 993 --ipc host --shm-size 4g --ulimit nofile=65536:65536 \
  -e EXL3_MODEL=/models/Qwen3.8-27B-SC_4.00bpw_H5_V6 -e EXL3_CACHE_TOKENS=32768 \
  -e EXL3_INT8_GEMV=0 EXL3_INT8_GEMV_MAX_K=0 \
  -v /home/ghazni/models/exl3/turboderp:/models:ro -v $BASE:/prof:ro -v $OUT:/out \
  --entrypoint python3 "$IMG" /prof/bench_lean.py 2>&1 | tail -1 | tee $OUT/b4_fp16gemv.txt

echo "=== B2 session complete ==="
