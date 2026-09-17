#!/usr/bin/env bash
# Dump the AMDGPU device ISA of a HIP fat binary for instruction-mix analysis (plan step A4).
# Runs INSIDE a ROCm container (hipcc's LLVM tools + the ROCm SDK are on the container's PATH).
#
#   isa_dump.sh <fat_binary> <out_dir>
#
# Writes <out_dir>/co/*.o (gfx1100 code objects) and <out_dir>/isa.txt (their combined
# disassembly, one "## <object>" header per object). `roc-obj-*` in this SDK is broken, so the
# bundler container is parsed by profiling/extract_isa.py; the SDK's llvm-objdump does not accept
# `--mcpu=gfx1100` (unrecognized processor), but the code objects are already gfx1100 ELF so the
# default decoding is correct (verified: v_dot/v_mul/ds_* decode, see out_a/isa/dp4a_peak_w32.txt).
set -euo pipefail
SRC=$1
OUT=$2
OBJDUMP=${OBJDUMP:-$(find /opt/rocm-venv/lib/python3.12/site-packages/_rocm_sdk_core -name llvm-objdump | head -1)}
export LLVM_OBJCOPY=${LLVM_OBJCOPY:-$(find /opt/rocm-venv/lib/python3.12/site-packages/_rocm_sdk_core -name llvm-objcopy | head -1)}
mkdir -p "$OUT/co"
python3 /prof/extract_isa.py "$SRC" "$OUT/co" | tail -1
: > "$OUT/isa.txt"
for f in "$OUT"/co/*.o; do
    echo "## ${f##*/}" >> "$OUT/isa.txt"
    "$OBJDUMP" -d "$f" >> "$OUT/isa.txt" 2>&1 || echo "(disassembly failed)" >> "$OUT/isa.txt"
done
wc -l "$OUT/isa.txt"
