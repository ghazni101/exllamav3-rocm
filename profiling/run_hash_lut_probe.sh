#!/usr/bin/env bash
# Isolated 2x256 LDS hash-LUT vs v_mul_lo probe. Tiny VRAM; serve may stay.
set -euo pipefail
ROOT=/home/ghazni/github/exllamav3-rocm
OUT=$ROOT/profiling/out_guided
IMG=${1:-exllamav3-rocm:goal-isa-had}
mkdir -p "$OUT"
~/gpu-coord/gpu-ctl reserve "[svcon] hash LUT LDS vs v_mul_lo probe" 8
rc=0
~/gpu-coord/gpu-ctl run 480 "[svcon] hash LUT probe" -- \
  docker run --rm --name exl3-hash-lut-probe \
    --device /dev/kfd --device /dev/dri --group-add 44 --group-add 993 \
    --ipc host --ulimit nofile=65536:65536 \
    -v "$ROOT/profiling:/prof:ro" \
    --entrypoint bash "$IMG" -lc '
      set -e
      HIPCC=$(command -v hipcc || true)
      if [ -z "$HIPCC" ]; then
        HIPCC=$(find /opt /usr -name hipcc -type f 2>/dev/null | head -1)
      fi
      echo "hipcc=$HIPCC"
      "$HIPCC" --offload-arch=gfx1100 -O3 -mcumode -o /tmp/hash_lut_probe /prof/hash_lut_probe.hip
      /tmp/hash_lut_probe 1024 20 384
    ' | tee "$OUT/hash-lut-probe.log" || rc=$?
~/gpu-coord/gpu-ctl done || true
echo "rc=$rc"
exit $rc
