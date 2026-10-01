#!/usr/bin/env bash
# CPU-only: confirm -mcumode flipped .workgroup_processor_mode to 0x00.
#   verify_cumode.sh <image> <out_dir>
set -euo pipefail
IMG=${1:?image}
OUT=${2:?out_dir}
mkdir -p "$OUT/co"
docker run --rm --entrypoint bash \
  -v /home/ghazni/github/exllamav3-rocm/profiling:/prof:ro \
  -v "$OUT":/out \
  "$IMG" -lc '
set -euo pipefail
SO=/opt/rocm-venv/lib/python3.12/site-packages/exllamav3_ext.cpython-312-x86_64-linux-gnu.so
OBJCOPY=$(find /opt/rocm-venv/lib/python3.12/site-packages/_rocm_sdk_core -name llvm-objcopy | head -1)
export LLVM_OBJCOPY=$OBJCOPY
python3 /prof/extract_isa.py "$SO" /out/co | tail -1
python3 /prof/wgp_mode_census.py /out/co
'
