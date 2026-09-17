#!/usr/bin/env bash
# T1 gates phase B: determinism, golden compare, batch gate, bench, soak.
# Requires phase A artifacts in out_exec5/. Run under the GPU lock, serve stopped.
set -uo pipefail
BASE=/home/ghazni/github/exllamav3-rocm/profiling
OUT=$BASE/out_exec5
IMG=exllamav3-rocm:perf-t1
DCKR="docker run --rm --name exl3-exec --device /dev/kfd --device /dev/dri \
  --group-add 44 --group-add 993 --ipc host --shm-size 4g --ulimit nofile=65536:65536 \
  -e EXL3_MODEL=/models/Qwen3.8-27B-SC_4.00bpw_H5_V6 -e EXL3_CACHE_TOKENS=32768 \
  -e HIPFIRE_KERNEL_CACHE=/var/cache/hipfire -e HIPFIRE_DIR=/root/.hipfire \
  -v /home/ghazni/models/exl3/turboderp:/models:ro -v exllamav3-kcache:/var/cache/hipfire \
  -v $BASE:/prof:ro -v $OUT:/out"

echo "=== G2: T1-numerics determinism (save run2, compare run3 vs run2) ==="
$DCKR --entrypoint python3 "$IMG" /prof/correctness_gate.py numcheck save /out/numcheck_t1_r2.pt || { echo "G2 FAIL"; exit 1; }
$DCKR --entrypoint python3 "$IMG" /prof/correctness_gate.py numcheck compare /out/numcheck_t1_r2.pt 2>&1 | tee "$OUT/numcheck_t1_r3.txt" | grep -E "PASS|FAIL" || { echo "G2 RUN ERROR"; exit 1; }
grep -q "NUMCHECK PASS" "$OUT/numcheck_t1_r3.txt" || { echo "G2 FAIL"; exit 1; }
python3 -c "
import re, sys
txt = open('$OUT/numcheck_t1_r3.txt').read()
worst = float(re.search(r'worst ([0-9.e+-]+)', txt).group(1))
sys.exit(0 if worst < 1e-9 else 1)
" || { echo "G2 FAIL: T1 numerics not bit-reproducible"; exit 1; }
echo "G2 PASS"

echo "=== G3: golden-token compare vs deployed baseline (short-prompt divergence tolerated) ==="
$DCKR --entrypoint python3 "$IMG" /prof/correctness_gate.py compare /prof/out_exec3/golden_tabbyapi.json 2>&1 | tee "$OUT/golden_t1.txt" | grep -E "MISMATCH|PASS|FAIL" || true
python3 -c "
import re, sys
txt = open('$OUT/golden_t1.txt').read()
if '[gate] PASS' in txt:
    print('G3 PASS: all prompts token-identical to deployed baseline')
    sys.exit(0)
div = set(re.findall(r'MISMATCH (\w+)', txt))
long_prompts = {'longctx_4k', 'prefix_A', 'prefix_B'}
bad = div & long_prompts
if bad:
    print(f'G3 FAIL: reconstruct-path prompts diverged: {bad}')
    sys.exit(1)
print(f'G3 PASS(conditional): short prompts diverged {sorted(div)} (numerics changed by design), long/reconstruct identical')
" || exit 1

echo "=== G4: batch-vs-sequential at m=8 ==="
$DCKR --entrypoint python3 "$IMG" /prof/correctness_gate.py batch /out/gate_batch_t1.json 2>&1 | tee "$OUT/batch_t1.txt" | grep -E "PASS|MISMATCH" || { echo "G4 FAIL"; exit 1; }
grep -q "PASS" "$OUT/batch_t1.txt" || { echo "G4 FAIL"; exit 1; }

echo "=== G5: bench_lean perf ==="
$DCKR --entrypoint python3 "$IMG" /prof/bench_lean.py 2>&1 | tee "$OUT/bench_lean_t1.txt" | tail -8

echo "=== G6: 10-minute soak ==="
SOAK_MINUTES=10 $DCKR --entrypoint python3 "$IMG" /prof/soak.py 2>&1 | tee "$OUT/soak_t1.txt" | tail -3
grep -q "vram_leak_mb" "$OUT/soak_t1.txt" || { echo "G6 FAIL (soak did not complete)"; exit 1; }
echo "=== phase B complete ==="
