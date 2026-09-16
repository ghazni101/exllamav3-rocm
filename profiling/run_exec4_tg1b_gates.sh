#!/usr/bin/env bash
# tg-1b gate session on exllamav3-rocm:perf-tg1b (msq for single-matrix m>4).
# Reference: out_exec3/golden_tabbyapi.json + bench_lean_tabbyapi.txt (deployed image).
set -uo pipefail
BASE=/home/ghazni/github/exllamav3-rocm/profiling
OUT=$BASE/out_exec4
mkdir -p "$OUT"
IMG=exllamav3-rocm:perf-tg1b
DCKR="docker run --rm --name exl3-exec --device /dev/kfd --device /dev/dri \
  --group-add 44 --group-add 993 --ipc host --shm-size 4g --ulimit nofile=65536:65536 \
  -e EXL3_MODEL=/models/Qwen3.8-27B-SC_4.00bpw_H5_V6 -e EXL3_CACHE_TOKENS=32768 \
  -e HIPFIRE_KERNEL_CACHE=/var/cache/hipfire -e HIPFIRE_DIR=/root/.hipfire \
  -v /home/ghazni/models/exl3/turboderp:/models:ro -v exllamav3-kcache:/var/cache/hipfire \
  -v $BASE:/prof:ro -v $OUT:/out -v /home/ghazni/github/exllamav3-rocm/profiling/shims:/shims:ro"

echo "=== G1: golden-token parity vs deployed reference ==="
$DCKR --entrypoint python3 "$IMG" /prof/correctness_gate.py compare /out/golden_tabbyapi.json || { echo "G1 FAIL"; exit 1; }

echo "=== G2: batch-vs-sequential at m=8 (exercises the new msq path) ==="
$DCKR --entrypoint python3 "$IMG" /prof/correctness_gate.py batch /out/gate_batch_tg1b.json || { echo "G2 FAIL"; exit 1; }

echo "=== G3: batch-8 routing check (kernel trace: coop gone, msq present) ==="
rm -rf "$OUT/grid"; mkdir -p "$OUT/grid"
docker run --rm --name exl3-grid --device /dev/kfd --device /dev/dri \
  --group-add 44 --group-add 993 --ipc host --shm-size 4g --ulimit nofile=65536:65536 \
  -e EXL3_MODEL=/models/Qwen3.8-27B-SC_4.00bpw_H5_V6 -e EXL3_CACHE_TOKENS=32768 \
  -e EXL3_GEN_TOKENS=48 -e HIPFIRE_KERNEL_CACHE=/var/cache/hipfire -e HIPFIRE_DIR=/root/.hipfire \
  -e LD_PRELOAD=/shims/ld_scope_shim.so \
  -v /home/ghazni/models/exl3/turboderp:/models:ro -v exllamav3-kcache:/var/cache/hipfire \
  -v /home/ghazni/github/exllamav3-rocm/profiling/shims:/shims:ro -v "$OUT/grid":/out \
  -v $BASE:/prof:ro \
  --entrypoint rocprofv3 "$IMG" --kernel-trace -f csv -d /out \
  -- python3 /prof/bench_lean.py > "$OUT/grid/bench_stdout.txt" 2>&1 || true
KT=$(ls "$OUT"/grid/*/1_kernel_trace.csv 2>/dev/null | head -1)
sudo chmod -R a+rX "$OUT/grid"
python3 - "$KT" <<'EOF'
import csv, sys, collections
# batch-8 window is the tail of bench_lean; count coop vs msq kernels in the last 40% of the span
rows = []
with open(sys.argv[1]) as f:
    r = csv.reader(f); hdr = next(r)
    iN, iS, iG, iW = hdr.index("Kernel_Name"), hdr.index("Start_Timestamp"), hdr.index("Grid_Size_X"), hdr.index("Workgroup_Size_X")
    for row in r:
        try: rows.append((int(row[iS]), row[iN], int(row[iG]), int(row[iW])))
        except: continue
rows.sort()
t0, t1 = rows[0][0], rows[-1][0]
tail = [x for x in rows if x[0] - t0 > (t1 - t0) * 0.6]
c = collections.Counter()
for _, n, g, w in tail:
    if "exl3_gemm_kernel" in n: c["coop"] += 1
    elif "exl3_gemv_int8_msq" in n: c["msq"] += 1
    elif "exl3_gemv_int8_sq" in n: c["sq"] += 1
print("batch-8 tail window kernel counts:", dict(c))
if c.get("coop", 0) == 0 and c.get("msq", 0) > 100:
    print("G3 PASS: batch projections routed to msq, coop eliminated")
else:
    print("G3 FAIL: coop still present or msq not used")
    sys.exit(1)
EOF
[ $? -ne 0 ] && { echo "G3 FAIL"; exit 1; }

echo "=== G4: 10-minute soak ==="
SOAK_MINUTES=10 $DCKR --entrypoint python3 "$IMG" /prof/soak.py || { echo "G4 FAIL"; exit 1; }

echo "=== G5: bench_lean perf (patched) ==="
$DCKR --entrypoint python3 "$IMG" /prof/bench_lean.py 2>&1 | tee "$OUT/bench_lean_tg1b.txt" | tail -6
echo "=== gate session complete ==="
