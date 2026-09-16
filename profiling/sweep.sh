#!/usr/bin/env bash
# Sequential knob sweep under a single GPU lock hold.
set -uo pipefail
P=/home/ghazni/github/exllamav3-rocm/profiling
COMMON=(--rm --device /dev/kfd --device /dev/dri --group-add 44 --group-add 993
  --ipc host --shm-size 4g --ulimit nofile=65536:65536
  -v /home/ghazni/models/exl3/turboderp:/models:ro
  -v exllamav3-kcache:/var/cache/hipfire
  -v "$P":/prof:ro
  -e EXL3_MODEL=/models/Qwen3.8-27B-SC_4.00bpw_H5_V6
  -e EXL3_CACHE_TOKENS=32768)
run_cfg () {
  local label="$1"; shift
  echo "=== $label ==="
  docker run "${COMMON[@]}" "$@" --entrypoint python3 tabbyapi-rocm:serve /prof/bench_lean.py 2>&1 | grep -E '^\{|Error|error|Traceback' | tail -3
}
run_cfg "baseline"
run_cfg "sq_rows_per=32"   -e EXL3_SQ_ROWS_PER=32
run_cfg "sq_rows_per=128"  -e EXL3_SQ_ROWS_PER=128
run_cfg "sq_rows_per=256"  -e EXL3_SQ_ROWS_PER=256
run_cfg "int8_gemv=1"      -e EXL3_INT8_GEMV=1
run_cfg "msq=0"            -e EXL3_INT8_MSQ=0
