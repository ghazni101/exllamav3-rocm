#!/usr/bin/env bash
# T1 gates phase A: standalone kernel A/B (msq vs sq-ref vs autotuned-coop, cold-state
# timing) + P1 bisect probe + numcheck KLD vs incumbent numerics.
# Run under the GPU lock with the standing serve stopped (full-VRAM work).
set -uo pipefail
BASE=/home/ghazni/github/exllamav3-rocm/profiling
OUT=$BASE/out_exec5
mkdir -p "$OUT"
IMG=exllamav3-rocm:perf-t1
DCKR="docker run --rm --name exl3-exec --device /dev/kfd --device /dev/dri \
  --group-add 44 --group-add 993 --ipc host --shm-size 4g --ulimit nofile=65536:65536 \
  -e EXL3_MODEL=/models/Qwen3.8-27B-SC_4.00bpw_H5_V6 -e PYTHONUNBUFFERED=1 \
  -e HIPFIRE_KERNEL_CACHE=/var/cache/hipfire -e HIPFIRE_DIR=/root/.hipfire \
  -v /home/ghazni/models/exl3/turboderp:/models:ro -v exllamav3-kcache:/var/cache/hipfire \
  -v $BASE:/prof:ro -v $OUT:/out"

echo "=== P1 probe: prefill GEMM gap bisect (informs pp-1.2) ==="
timeout -k 30 300 $DCKR --entrypoint python3 "$IMG" /prof/pp_bisect.py 2>&1 | tee "$OUT/pp_bisect.txt" | grep -E "TF/s|\{" || true

echo "=== G0a: standalone A/B, msq mode (cold rotation) ==="
timeout -k 30 420 $DCKR --entrypoint python3 "$IMG" /prof/msq_ab2.py 2>&1 | tee "$OUT/ab_msq.txt" | grep -E "==|\[msq\]|saved" || { echo "G0a RUN ERROR"; exit 1; }

echo "=== G0b: standalone A/B, autotuned coop reference (cold rotation) ==="
timeout -k 30 420 $DCKR -e EXL3_INT8_MSQ=0 -e AB_MODE=coopref --entrypoint python3 "$IMG" /prof/msq_ab2.py 2>&1 | tee "$OUT/ab_coop.txt" | grep -E "==|\[coop\]|saved" || { echo "G0b RUN ERROR"; exit 1; }

echo "=== G0c: compare (same-input tolerances) ==="
timeout -k 30 300 $DCKR -e AB_MODE=compare --entrypoint python3 "$IMG" /prof/msq_ab2.py 2>&1 | tee "$OUT/ab_compare.txt" | grep -E "K=|PASS|FAIL" || { echo "G0c RUN ERROR"; exit 1; }
grep -q "^\[PASS\]" "$OUT/ab_compare.txt" || { echo "G0 FAIL"; exit 1; }

echo "=== G1a: numcheck reference on incumbent numerics (EXL3_INT8_MSQ=0) ==="
timeout -k 30 420 $DCKR -e EXL3_INT8_MSQ=0 --entrypoint python3 "$IMG" /prof/correctness_gate.py numcheck save /out/numcheck_coop.pt | tail -3 || { echo "G1a FAIL"; exit 1; }

echo "=== G1b: KLD compare, T1 numerics (5e-3: per-slice vs global scheme delta; greedy tokens must not diverge) ==="
timeout -k 30 420 $DCKR -e NUMCHECK_KLD=5e-3 --entrypoint python3 "$IMG" /prof/correctness_gate.py numcheck compare /out/numcheck_coop.pt 2>&1 | tee "$OUT/numcheck_t1.txt" | grep -E "numcheck|PASS|FAIL" || { echo "G1b RUN ERROR"; exit 1; }
grep -q "NUMCHECK PASS" "$OUT/numcheck_t1.txt" || { echo "G1b FAIL"; exit 1; }

echo "=== phase A complete ==="
