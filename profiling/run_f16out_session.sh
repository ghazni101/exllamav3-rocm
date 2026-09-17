#!/usr/bin/env bash
# Full session for EXL3_HGEMM_F16OUT (fp32-output reconstruct GEMM via an fp16 slab + widening):
# probe -> serve down -> bench A/B -> numeric-change gates -> soak -> serve up.
#   run_f16out_session.sh <image>
# The serve is stopped for the bench (full-VRAM work) and always restored, including on failure.
set -uo pipefail
BASE=/home/ghazni/github/exllamav3-rocm/profiling
OUT=$BASE/out_a
IMG=${1:-exllamav3-rocm:perf-h}
SERVE=exllamav3-rocm-serve
C="--rm --device /dev/kfd --device /dev/dri --group-add 44 --group-add 993 --ipc host \
   --shm-size 4g --ulimit nofile=65536:65536 \
   -e EXL3_MODEL=/models/Qwen3.8-27B-SC_4.00bpw_H5_V6 -e EXL3_CACHE_TOKENS=32768 \
   -v /home/ghazni/models/exl3/turboderp:/models:ro -v $BASE:/prof:ro -v $OUT:/out"

restore_serve() {
  echo "=== restoring the standing serve ==="
  docker start $SERVE >/dev/null 2>&1
  for _ in $(seq 1 40); do
    sleep 15
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://192.168.1.200:9001/v1/model || true)
    [ "$code" = "200" ] && { echo "serve healthy"; break; }
  done
  docker ps --filter name=$SERVE --format '{{.Names}} {{.Status}}'
  curl -s -m 10 http://192.168.1.200:9001/v1/model | head -c 80; echo
}
trap restore_serve EXIT

echo "=== image guard ==="
docker run --rm --entrypoint bash "$IMG" -lc \
  'SO=$(find /opt/rocm-venv -name "exllamav3_ext*.so" | head -1); \
   echo "ext=$SO f16out=$(strings "$SO" | grep -c EXL3_HGEMM_F16OUT) msq=$(strings "$SO" | grep -c msq-launch)"'

echo "=== P1: per-shape probe, both arms (shape alternation exercises the slab cache) ==="
for arm in 0 1; do
  docker run $C --name exl3-pr-$arm -e EXL3_HGEMM_F16OUT=$arm --entrypoint python3 "$IMG" \
    /prof/f32out_gemm_probe.py 2>&1 | grep -E "fp32_ref|convert path|Error" || true
done

echo "=== stopping the serve for full-VRAM work ==="
docker stop -t 30 $SERVE >/dev/null 2>&1
sleep 5
docker ps --filter name=$SERVE --format '{{.Names}} {{.Status}}' || true

echo "=== P2: bench_lean A/B (2 runs per arm) ==="
for arm in 0 1; do
  for i in 1 2; do
    echo "--- arm=$arm run$i"
    docker run $C --name exl3-bench-$arm-$i -e EXL3_HGEMM_F16OUT=$arm --entrypoint python3 "$IMG" \
      /prof/bench_lean.py 2>&1 | tail -1 | tee -a $OUT/f16out_bench_arm$arm.txt
  done
done

echo "=== G1: numeric-change KLD (NUMCHECK_LONG=1), arm=0 reference then arm=1 compare ==="
docker run $C --name exl3-nc0 -e EXL3_HGEMM_F16OUT=0 -e NUMCHECK_LONG=1 --entrypoint python3 "$IMG" \
  /prof/correctness_gate.py numcheck save /out/f16out_off.pt 2>&1 | tail -2
docker run $C --name exl3-nc1 -e EXL3_HGEMM_F16OUT=1 -e NUMCHECK_LONG=1 --entrypoint python3 "$IMG" \
  /prof/correctness_gate.py numcheck compare /out/f16out_off.pt 2>&1 | tee $OUT/f16out_numcheck.txt | tail -14

echo "=== G1b: same-arm determinism at arm=1 ==="
docker run $C --name exl3-nc1b -e EXL3_HGEMM_F16OUT=1 -e NUMCHECK_LONG=1 --entrypoint python3 "$IMG" \
  /prof/correctness_gate.py numcheck save /out/f16out_on_a.pt 2>&1 | tail -2
docker run $C --name exl3-nc1c -e EXL3_HGEMM_F16OUT=1 -e NUMCHECK_LONG=1 --entrypoint python3 "$IMG" \
  /prof/correctness_gate.py numcheck compare /out/f16out_on_a.pt 2>&1 | tail -5

echo "=== G2: golden tokens, arm=1 ==="
docker run $C --name exl3-g2 -e EXL3_HGEMM_F16OUT=1 --entrypoint python3 "$IMG" \
  /prof/correctness_gate.py compare /prof/out_exec3/golden_tabbyapi.json 2>&1 \
  | tee $OUT/f16out_golden.txt | tail -6

echo "=== G3: batch-vs-sequential m=8, arm=1 ==="
docker run $C --name exl3-g3 -e EXL3_HGEMM_F16OUT=1 --entrypoint python3 "$IMG" \
  /prof/correctness_gate.py batch /out/f16out_batch.json 2>&1 | tee $OUT/f16out_batch.txt | tail -4

echo "=== G4: 10-minute soak, arm=1 ==="
SOAK_MINUTES=10 docker run $C --name exl3-g4 -e EXL3_HGEMM_F16OUT=1 -e SOAK_MINUTES=10 --entrypoint python3 "$IMG" \
  /prof/soak.py 2>&1 | tail -3 | tee $OUT/f16out_soak.txt

echo "=== session complete ==="
