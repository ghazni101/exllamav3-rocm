#!/usr/bin/env bash
# rocprofv3 profiling wrapper for exllamav3 on RDNA3.
#
# IMPORTANT: do NOT invoke the `rocprofv3` frontend — it LD_PRELOADs
# librocprofiler-sdk.so, whose bundled libLLVM interposes llvm:: symbols into
# triton's libtriton.so static initializers and corrupts the heap (SIGSEGV in
# cfree during `import exllamav3`). Instead we load the same tool library via
# the register mechanism (ROCP_TOOL_LIBRARIES), which dlopens it lazily after
# all imports are initialized.
#
# Usage (inside the container):
#   ./profile_rocm.sh kernel   # per-kernel dispatch trace (hotspots)
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

TOOL=/opt/rocm/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so
export ROCP_TOOL_LIBRARIES="$TOOL"
export ROCPROFILER_LIBRARY_CTOR=1
export ROCPROF_OUTPUT_FORMAT=csv
export ROCPROF_OUTPUT_PATH="$OUT"
export ROCPROF_TIME_FORMAT=nsec

case "$MODE" in
  kernel)
    export ROCPROF_OUTPUT_FILE_NAME=kernel_trace
    export ROCPROF_KERNEL_TRACE=1
    export ROCPROF_STATS=1 ROCPROF_STATS_SUMMARY=1
    export ROCPROF_STATS_SUMMARY_OUTPUT="$OUT/kernel_summary.txt"
    ;;
  api)
    export ROCPROF_OUTPUT_FILE_NAME=api_trace
    export ROCPROF_HIP_RUNTIME_API_TRACE=1
    export ROCPROF_KERNEL_TRACE=1
    export ROCPROF_STATS=1 ROCPROF_STATS_SUMMARY=1
    export ROCPROF_STATS_SUMMARY_OUTPUT="$OUT/api_summary.txt"
    ;;
  pmc)
    # Achieved DRAM traffic of the quant GEMV/GEMM kernels.
    # gfx1101 counters: FETCH_SIZE = KB fetched from VRAM; SQ_WAVES = waves
    # launched (blocks*warps). TCC_* does not exist on gfx1101 (use GL2C_*).
    # Single pass only: >~4 counters fails rocprofiler_create_counter_config.
    ROCPROF_COUNTERS=${ROCPROF_COUNTERS:-"pmc: FETCH_SIZE SQ_WAVES"}
    REGEX=${KERNEL_REGEX:-"exl3_gemv|exl3_gemv_int8|exl3_mgemm|exl3_gemm"}
    export ROCPROF_OUTPUT_FILE_NAME=pmc_trace
    export ROCPROF_COUNTER_COLLECTION=1
    export ROCPROF_COUNTERS
    export ROCPROF_KERNEL_FILTER_INCLUDE_REGEX="$REGEX"
    ;;
  *)
    echo "unknown mode: $MODE (use kernel|api|pmc)" >&2
    exit 2
    ;;
esac

python3 "$SCRIPT_DIR/profile_rocm.py"

echo "outputs in $OUT:"
ls -la "$OUT"
