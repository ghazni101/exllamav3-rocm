#!/usr/bin/env bash
# Verify the image that will be promoted: the shipped default (knob unset) must reproduce the
# measured arm=1 numbers - bench prefill ~510-560 tok/s, the same KLD profile against the saved
# arm=0 reference, and the same (pre-existing) golden divergences. Serve stays down; the promote
# step brings it back with the new image.
#   run_perf_i_verify.sh <image>
set -uo pipefail
BASE=/home/ghazni/github/exllamav3-rocm/profiling
OUT=$BASE/out_a
IMG=${1:-exllamav3-rocm:perf-i}
C="--rm --device /dev/kfd --device /dev/dri --group-add 44 --group-add 993 --ipc host \
   --shm-size 4g --ulimit nofile=65536:65536 \
   -e EXL3_MODEL=/models/Qwen3.8-27B-SC_4.00bpw_H5_V6 -e EXL3_CACHE_TOKENS=32768 \
   -v /home/ghazni/models/exl3/turboderp:/models:ro -v $BASE:/prof:ro -v $OUT:/out"

echo "=== guard: shipped default present, no env override below ==="
docker run --rm --entrypoint bash "$IMG" -lc \
  'SO=$(find /opt/rocm-venv -name "exllamav3_ext*.so" | head -1); \
   echo "ext=$SO f16out=$(strings "$SO" | grep -c EXL3_HGEMM_F16OUT) msq=$(strings "$SO" | grep -c msq-launch)"'

echo "=== bench_lean, shipped default, 2 runs ==="
for i in 1 2; do
  docker run $C --name exl3-vi-$i --entrypoint python3 "$IMG" /prof/bench_lean.py 2>&1 | tail -1 \
    | tee -a $OUT/perf_i_bench.txt
done

echo "=== numcheck (NUMCHECK_LONG=1) vs the arm-0 reference: expect the arm-1 KLD profile ==="
docker run $C --name exl3-vi-nc -e NUMCHECK_LONG=1 --entrypoint python3 "$IMG" \
  /prof/correctness_gate.py numcheck compare /out/f16out_off.pt 2>&1 | tee $OUT/perf_i_numcheck.txt | tail -12

echo "=== golden tokens, shipped default ==="
docker run $C --name exl3-vi-g2 --entrypoint python3 "$IMG" \
  /prof/correctness_gate.py compare /prof/out_exec3/golden_tabbyapi.json 2>&1 \
  | tee $OUT/perf_i_golden.txt | tail -5

echo "=== verify complete ==="
