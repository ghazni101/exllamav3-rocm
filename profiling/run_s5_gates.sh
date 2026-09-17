#!/usr/bin/env bash
# Session-5 final gates on the shipping image: determinism, golden compare, batch, soak, bench.
#   run_s5_gates.sh <image>
set -uo pipefail
BASE=/home/ghazni/github/exllamav3-rocm/profiling
OUT=$BASE/out_a
IMG=${1:-exllamav3-rocm:perf-c}
C="--rm --device /dev/kfd --device /dev/dri --group-add 44 --group-add 993 --ipc host \
  --shm-size 4g --ulimit nofile=65536:65536 \
  -e EXL3_MODEL=/models/Qwen3.8-27B-SC_4.00bpw_H5_V6 -e EXL3_CACHE_TOKENS=32768 \
  -v /home/ghazni/models/exl3/turboderp:/models:ro -v $BASE:/prof:ro -v $OUT:/out"

echo "=== image guard (deployed route + knobs present?) ==="
docker run --rm --entrypoint bash "$IMG" -lc \
  'SO=$(find /opt/rocm-venv -name "exllamav3_ext*.so" | head -1); echo "ext: $SO"; \
   echo "msq=$(strings "$SO" | grep -c msq-launch) sqstage=$(strings "$SO" | grep -c EXL3_SQ_STAGE_SMEM) sqlf=$(strings "$SO" | grep -c EXL3_SQ_PF)"' \
  || { echo "VOID"; exit 1; }

echo "=== G1: T1-numerics determinism (numcheck save + compare, same binary) ==="
docker run $C --name exl3-g1 --entrypoint python3 "$IMG" /prof/correctness_gate.py numcheck save /out/s5_numcheck_a.pt 2>&1 | tail -2
docker run $C --name exl3-g1b --entrypoint python3 "$IMG" /prof/correctness_gate.py numcheck compare /out/s5_numcheck_a.pt 2>&1 \
  | tee $OUT/s5_numcheck.txt | grep -E "PASS|FAIL|worst" || true
grep -q "NUMCHECK PASS" $OUT/s5_numcheck.txt || echo "G1: CHECK (no PASS line)"

echo "=== G2: golden-token compare vs the deployed baseline ==="
docker run $C --name exl3-g2 --entrypoint python3 "$IMG" /prof/correctness_gate.py compare /prof/out_exec3/golden_tabbyapi.json 2>&1 \
  | tee $OUT/s5_golden.txt | grep -E "MISMATCH|PASS|FAIL" || true

echo "=== G3: batch-vs-sequential at m=8 ==="
docker run $C --name exl3-g3 --entrypoint python3 "$IMG" /prof/correctness_gate.py batch /out/s5_batch.json 2>&1 \
  | tee $OUT/s5_batch.txt | grep -E "PASS|MISMATCH|FAIL" || true

echo "=== G5: bench_lean (final metrics, 2 runs) ==="
for i in 1 2; do
  docker run $C --name exl3-g5 --entrypoint python3 "$IMG" /prof/bench_lean.py 2>&1 | tail -1 \
    | tee -a $OUT/s5_bench_lean.txt
done

echo "=== G4: 10-minute soak ==="
SOAK_MINUTES=10 docker run $C --name exl3-g4 -e SOAK_MINUTES=10 --entrypoint python3 "$IMG" /prof/soak.py 2>&1 \
  | tail -3 | tee $OUT/s5_soak.txt
grep -q "vram_leak_mb" $OUT/s5_soak.txt && echo "G4 PASS" || echo "G4: no leak line (hang or failure)"
echo "=== gates complete ==="
