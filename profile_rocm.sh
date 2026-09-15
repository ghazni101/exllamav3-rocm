#!/usr/bin/env bash
# rocprofv3 profiling wrapper for exllamav3 on RDNA3.
#
# Usage (inside the container):
#   ./profile_rocm.sh kernel   # per-kernel dispatch trace + stats (hotspots)
#   ./profile_rocm.sh api      # HIP runtime API trace (host overhead / gaps)
#   ./profile_rocm.sh pmc      # HW counters on the hot GEMV/GEMM kernels (bandwidth)
#
# Outputs land in $OUT (default /tmp/rocmprof). The steady-state decode region
# is sliced offline by timestamp, after the warmup generation.
set -euo pipefail

OUT=${OUT:-/tmp/rocmprof}
MODE=${1:-kernel}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT"

case "$MODE" in
  kernel)
    rocprofv3 --kernel-trace --stats \
      -f csv -d "$OUT" -o kernel_trace \
      --summary-output-file "$OUT/kernel_summary.txt" -u usec \
      -- python3 "$SCRIPT_DIR/profile_rocm.py"
    ;;
  api)
    rocprofv3 --hip-runtime-trace --stats \
      -f csv -d "$OUT" -o api_trace \
      --summary-output-file "$OUT/api_summary.txt" -u usec \
      --summary-groups "HIP_API|KERNEL_DISPATCH|MEMORY_COPY" \
      -- python3 "$SCRIPT_DIR/profile_rocm.py"
    ;;
  pmc)
    # Achieved DRAM traffic of the quant GEMV kernels: sector-level read counts.
    # TCC = L2 cache channels; total bytes read ~= sum of TCC_*RQ counts * 32B sectors.
    ROCPROF_COUNTERS=${ROCPROF_COUNTERS:-"SQ_WAVES TCC_RD_REQ_32B TCC_WR_REQ_32B TCC_HIT_32B TCC_MISS_32B"}
    REGEX=${KERNEL_REGEX:-"exl3_gemv|exl3_gemv_int8|hgemm"}
    rocprofv3 -i <(echo "$ROCPROF_COUNTERS") \
      --kernel-trace --kernel-include-regex "$REGEX" \
      -f csv -d "$OUT" -o pmc_trace \
      -- python3 "$SCRIPT_DIR/profile_rocm.py"
    ;;
  *)
    echo "unknown mode: $MODE (use kernel|api|pmc)" >&2
    exit 2
    ;;
esac

echo "outputs in $OUT:"
ls -la "$OUT"
