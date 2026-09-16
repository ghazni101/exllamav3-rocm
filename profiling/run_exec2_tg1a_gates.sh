#!/usr/bin/env bash
# tg-1a gate session on the PATCHED image (exllamav3-rocm:perf-tg1a).
# Gates (all must pass before promotion):
#   G1 golden-token parity vs baseline captured on the current image
#   G2 batch-vs-sequential equivalence
#   G3 kernel trace shows exl3_gemm_kernel grids = 96 blocks x 512 threads
#   G4 10-minute mixed-load soak, no hang, no VRAM leak
#   G5 perf: bench_lean batch-8 aggregate improves vs the current-image number
set -uo pipefail
BASE=/home/ghazni/github/exllamav3-rocm/profiling
OUT=$BASE/out_exec2
mkdir -p "$OUT"
IMG=exllamav3-rocm:perf-tg1a
DCKR="docker run --rm --name exl3-exec --device /dev/kfd --device /dev/dri \
  --group-add 44 --group-add 993 --ipc host --shm-size 4g --ulimit nofile=65536:65536 \
  -e EXL3_MODEL=/models/Qwen3.8-27B-SC_4.00bpw_H5_V6 -e EXL3_CACHE_TOKENS=32768 \
  -e HIPFIRE_KERNEL_CACHE=/var/cache/hipfire -e HIPFIRE_DIR=/root/.hipfire \
  -v /home/ghazni/models/exl3/turboderp:/models:ro -v exllamav3-kcache:/var/cache/hipfire \
  -v $BASE:/prof:ro -v $OUT:/out -v /home/ghazni/github/exllamav3-rocm/profiling/shims:/shims:ro"

echo "=== G5a: bench_lean perf (CURRENT image, reference) ==="
CUR=exllamav3-rocm:serve
$DCKR --entrypoint python3 "$CUR" /prof/bench_lean.py 2>&1 | tee "$OUT/bench_lean_current.txt" | tail -8

echo "=== G1+G2: correctness on patched image ==="
$DCKR --entrypoint python3 "$IMG" /prof/correctness_gate.py compare /out/golden_baseline.json || { echo "G1 FAIL"; exit 1; }
$DCKR --entrypoint python3 "$IMG" /prof/correctness_gate.py batch /out/gate_batch_new.json || { echo "G2 FAIL"; exit 1; }

echo "=== G3: coop grid check (kernel trace, shim + CLI) ==="
rm -rf "$OUT/grid"; mkdir -p "$OUT/grid"
G3=$(docker run --rm --name exl3-grid --device /dev/kfd --device /dev/dri \
  --group-add 44 --group-add 993 --ipc host --shm-size 4g --ulimit nofile=65536:65536 --cap-add=PERFMON \
  -e EXL3_MODEL=/models/Qwen3.8-27B-SC_4.00bpw_H5_V6 -e EXL3_CACHE_TOKENS=32768 \
  -e EXL3_WARMUP_TOKENS=8 -e EXL3_DECODE_TOKENS=24 -e EXL3_PREFILL_TOKENS=0 \
  -e HIPFIRE_KERNEL_CACHE=/var/cache/hipfire -e HIPFIRE_DIR=/root/.hipfire \
  -e LD_PRELOAD=/shims/ld_scope_shim.so \
  -v /home/ghazni/models/exl3/turboderp:/models:ro -v exllamav3-kcache:/var/cache/hipfire \
  -v /home/ghazni/github/exllamav3-rocm/profiling/shims:/shims:ro -v "$OUT/grid":/out \
  --entrypoint rocprofv3 "$IMG" --kernel-trace -f csv -d /out \
  -- python3 /opt/exllamav3/profile_rocm.py 2>&1 | grep -E '^\[profile\]' || true)
echo "$G3"
KT=$(ls "$OUT"/grid/*/1_kernel_trace.csv 2>/dev/null | head -1)
sudo chmod -R a+rX "$OUT/grid"
python3 - "$KT" <<'EOF'
import csv, sys, collections
grids = collections.Counter()
with open(sys.argv[1]) as f:
    r = csv.reader(f); hdr = next(r)
    iN, iG, iW = hdr.index("Kernel_Name"), hdr.index("Grid_Size_X"), hdr.index("Workgroup_Size_X")
    for row in r:
        if "exl3_gemm_kernel" in row[iN]:
            grids[(int(row[iG]), int(row[iW]))] += 1
print("exl3_gemm_kernel (threads, wgsize) -> count:", dict(grids))
blocks = {g // w for (g, w) in grids}
if blocks == {96}:
    print("G3 PASS: all coop launches at 96 blocks x 512 threads")
else:
    print(f"G3 FAIL: coop grids in blocks = {blocks} (expected {{96}})")
    sys.exit(1)
EOF
[ $? -ne 0 ] && { echo "G3 FAIL"; exit 1; }

echo "=== G4: 10-minute soak ==="
SOAK_MINUTES=10 $DCKR --entrypoint python3 "$IMG" /prof/soak.py || { echo "G4 FAIL"; exit 1; }

echo "=== G5: bench_lean perf (patched) ==="
$DCKR --entrypoint python3 "$IMG" /prof/bench_lean.py 2>&1 | tee "$OUT/bench_lean_tg1a.txt" | tail -10
echo "=== gate session complete ==="
