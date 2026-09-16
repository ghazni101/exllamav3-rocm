#!/usr/bin/env bash
# Baseline session on the CURRENT deployed image (exllamav3-rocm:serve):
#  1. golden-token baseline + batch-vs-sequential gate
#  2. pp-0 probe (matmul vs hgemm_recon on the model's exact shapes)
#  3. bench_lean perf baseline (unprofiled)
set -euo pipefail
BASE=/home/ghazni/github/exllamav3-rocm/profiling
OUT=$BASE/out_exec1
mkdir -p "$OUT"
IMG=${IMG:-exllamav3-rocm:serve}
DCKR="docker run --rm --name exl3-exec --device /dev/kfd --device /dev/dri \
  --group-add 44 --group-add 993 --ipc host --shm-size 4g --ulimit nofile=65536:65536 \
  -e EXL3_MODEL=/models/Qwen3.8-27B-SC_4.00bpw_H5_V6 -e EXL3_CACHE_TOKENS=32768 \
  -e GATE_GEN_TOKENS=256 \
  -e HIPFIRE_KERNEL_CACHE=/var/cache/hipfire -e HIPFIRE_DIR=/root/.hipfire \
  -v /home/ghazni/models/exl3/turboderp:/models:ro -v exllamav3-kcache:/var/cache/hipfire \
  -v $BASE:/prof:ro -v $OUT:/out -v /home/ghazni/github/exllamav3-rocm/profiling/gate:/gate"

echo "=== [1] correctness baseline (current image) ==="
$DCKR --entrypoint python3 "$IMG" /prof/correctness_gate.py baseline /out/golden_baseline.json

echo "=== [2] pp-0 gemm probe (current image) ==="
$DCKR --entrypoint python3 "$IMG" /prof/gemm_probe2.py 2>&1 | tee "$OUT/gemm_probe2_current.txt" | grep -E '===|WEIGHTED|shapes'

echo "=== [3] bench_lean baseline (current image) ==="
$DCKR --entrypoint python3 "$IMG" /prof/bench_lean.py 2>&1 | tee "$OUT/bench_lean_current.txt" | tail -8
echo "=== done ==="
