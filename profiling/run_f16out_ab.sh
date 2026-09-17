#!/usr/bin/env bash
# A/B and gates for EXL3_HGEMM_F16OUT (fp32-output reconstruct GEMM: fp16 slab + widening).
# Both arms come from the same image; the knob is read once per process, so each arm is its own
# container. Follows run_s5_gates.sh's invocations.
#   run_f16out_ab.sh <image> probe|bench|gates <arm 0|1>
set -uo pipefail
BASE=/home/ghazni/github/exllamav3-rocm/profiling
OUT=$BASE/out_a
IMG=${1:-exllamav3-rocm:perf-g}
STAGE=${2:-probe}
ARM=${3:-0}
C="--rm --device /dev/kfd --device /dev/dri --group-add 44 --group-add 993 --ipc host \
   --shm-size 4g --ulimit nofile=65536:65536 \
   -e EXL3_MODEL=/models/Qwen3.8-27B-SC_4.00bpw_H5_V6 -e EXL3_CACHE_TOKENS=32768 \
   -v /home/ghazni/models/exl3/turboderp:/models:ro -v $BASE:/prof:ro -v $OUT:/out"

case "$STAGE" in
probe)
  echo "=== knob present in the binary? ==="
  docker run --rm --entrypoint bash "$IMG" -lc \
    'SO=$(find /opt/rocm-venv -name "exllamav3_ext*.so" | head -1); echo "EXL3_HGEMM_F16OUT=$(strings "$SO" | grep -c EXL3_HGEMM_F16OUT)"'
  for arm in 0 1; do
    echo "=== f32out probe: arm=$arm ==="
    docker run $C --name exl3-f32-$arm -e EXL3_HGEMM_F16OUT=$arm --entrypoint python3 "$IMG" \
      /prof/f32out_gemm_probe.py 2>&1 | tail -12
  done
  ;;
bench)
  for arm in 0 1; do
    for i in 1 2; do
      echo "=== bench_lean arm=$arm run$i ==="
      docker run $C --name exl3-bench-$arm-e$i -e EXL3_HGEMM_F16OUT=$arm --entrypoint python3 "$IMG" \
        /prof/bench_lean.py 2>&1 | tail -1 | tee -a $OUT/f16out_bench_arm$arm.txt
    done
  done
  ;;
gates)
  echo "=== numcheck (long variant, NUMCHECK_LONG=1): save ref with arm=0, then compare ==="
  if [ "$ARM" = "0" ]; then
    docker run $C --name exl3-nc0 -e EXL3_HGEMM_F16OUT=0 -e NUMCHECK_LONG=1 --entrypoint python3 "$IMG" \
      /prof/correctness_gate.py numcheck save /out/f16out_off.pt 2>&1 | tail -3
  else
    echo "--- KLD/token divergence of arm=1 vs arm=0 on the reconstruct path"
    docker run $C --name exl3-nc1 -e EXL3_HGEMM_F16OUT=1 -e NUMCHECK_LONG=1 --entrypoint python3 "$IMG" \
      /prof/correctness_gate.py numcheck compare /out/f16out_off.pt 2>&1 | tee $OUT/f16out_numcheck.txt | tail -14
    echo "--- same-arm determinism (save then compare, both arm=1)"
    docker run $C --name exl3-nc1b -e EXL3_HGEMM_F16OUT=1 -e NUMCHECK_LONG=1 --entrypoint python3 "$IMG" \
      /prof/correctness_gate.py numcheck save /out/f16out_on_a.pt 2>&1 | tail -2
    docker run $C --name exl3-nc1c -e EXL3_HGEMM_F16OUT=1 -e NUMCHECK_LONG=1 --entrypoint python3 "$IMG" \
      /prof/correctness_gate.py numcheck compare /out/f16out_on_a.pt 2>&1 | tail -4
    echo "--- golden tokens (expected to diverge on reconstruct-path prompts for a numeric change)"
    docker run $C --name exl3-g2 -e EXL3_HGEMM_F16OUT=1 --entrypoint python3 "$IMG" \
      /prof/correctness_gate.py compare /prof/out_exec3/golden_tabbyapi.json 2>&1 \
      | tee $OUT/f16out_golden.txt | tail -6
    echo "--- batch-vs-sequential at m=8"
    docker run $C --name exl3-g3 -e EXL3_HGEMM_F16OUT=1 --entrypoint python3 "$IMG" \
      /prof/correctness_gate.py batch /out/f16out_batch.json 2>&1 | tee $OUT/f16out_batch.txt | tail -4
  fi
  ;;
*) echo "unknown stage: $STAGE" >&2; exit 64 ;;
esac
