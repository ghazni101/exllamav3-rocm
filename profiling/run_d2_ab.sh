#!/usr/bin/env bash
# D2 implementation A/B: does routing hgemm through ATen/hipBLASLt move prefill end-to-end?
#   run_d2_ab.sh <image>
# Arms (env-selected on one image, so no cross-image codegen drift):
#   1. default (incumbent cublasGemmEx path)          bench_lean
#   2. EXL3_HGEMM_ATEN=1                              bench_lean x2
#   3. numcheck save on both arms -> cross-arm KLD (<= 4e-3 is the project's accepted delta)
set -uo pipefail
BASE=/home/ghazni/github/exllamav3-rocm/profiling
OUT=$BASE/out_a
IMG=${1:-exllamav3-rocm:perf-d}
C="--rm --device /dev/kfd --device /dev/dri --group-add 44 --group-add 993 --ipc host \
  --shm-size 4g --ulimit nofile=65536:65536 \
  -e EXL3_MODEL=/models/Qwen3.8-27B-SC_4.00bpw_H5_V6 -e EXL3_CACHE_TOKENS=32768 \
  -v /home/ghazni/models/exl3/turboderp:/models:ro -v $BASE:/prof:ro -v $OUT:/out"

echo "=== image guard ==="
docker run --rm --entrypoint bash "$IMG" -lc \
  'SO=$(find /opt/rocm-venv -name "exllamav3_ext*.so" | head -1); echo "msq=$(strings "$SO" | grep -c msq-launch) ATEN=$(strings "$SO" | grep -c EXL3_HGEMM_ATEN)"' || exit 1

echo "=== numcheck: incumbent arm ==="
docker run $C --name exl3-d2n0 --entrypoint python3 "$IMG" \
  /prof/correctness_gate.py numcheck save /out/d2_numcheck_off.pt 2>&1 | tail -2
echo "=== numcheck: ATen arm ==="
docker run $C --name exl3-d2n1 -e EXL3_HGEMM_ATEN=1 --entrypoint python3 "$IMG" \
  /prof/correctness_gate.py numcheck save /out/d2_numcheck_on.pt 2>&1 | tail -2
echo "=== cross-arm KLD ==="
docker run $C --name exl3-d2nd --entrypoint python3 "$IMG" \
  /prof/numcheck_diff.py /out/d2_numcheck_off.pt /out/d2_numcheck_on.pt 2>&1 | tail -10 \
  | tee $OUT/d2_numcheck_diff.txt

echo "=== bench_lean: incumbent (default) ==="
docker run $C --name exl3-d2b0 --entrypoint python3 "$IMG" /prof/bench_lean.py 2>&1 | tail -1 \
  | tee $OUT/d2_bench_off.txt
echo "=== bench_lean: ATen arm x2 ==="
for i in 1 2; do
  docker run $C --name exl3-d2b1 -e EXL3_HGEMM_ATEN=1 --entrypoint python3 "$IMG" \
    /prof/bench_lean.py 2>&1 | tail -1 | tee -a $OUT/d2_bench_on.txt
done
echo "=== D2 implementation A/B complete ==="
