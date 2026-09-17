#!/usr/bin/env bash
# Direct hgemm probe, both arms, under [svcon] (the probe's footprint is a few hundred MB, so the
# standing serve stays up). One process per arm because the extension reads EXL3_HGEMM_LT once.
#   run_hgemm_lt_probe.sh <image>
set -uo pipefail
BASE=/home/ghazni/github/exllamav3-rocm/profiling
OUT=$BASE/out_a
IMG=${1:-exllamav3-rocm:perf-f}
C="--rm --device /dev/kfd --device /dev/dri --group-add 44 --group-add 993 --ipc host \
  --shm-size 4g --ulimit nofile=65536:65536 \
  -e EXL3_MODEL=/models/Qwen3.8-27B-SC_4.00bpw_H5_V6 -e OUT_DIR=/out \
  -v /home/ghazni/models/exl3/turboderp:/models:ro -v $BASE:/prof:ro -v $OUT:/out"

echo "=== knob compiled in? ==="
docker run --rm --entrypoint bash "$IMG" -lc \
  'SO=$(find /opt/rocm-venv -name "exllamav3_ext*.so" | head -1); \
   echo "EXL3_HGEMM_LT=$(strings "$SO" | grep -c EXL3_HGEMM_LT) msq=$(strings "$SO" | grep -c msq-launch) \
hipblaslt_linked=$(ldd "$SO" | grep -c hipblaslt)"'

for arm in 0 1; do
  name=$([ "$arm" = "1" ] && echo lt || echo incumbent)
  echo "=== probe: $name (EXL3_HGEMM_LT=$arm) ==="
  docker run $C --name exl3-lt$arm -e EXL3_HGEMM_LT=$arm --entrypoint python3 "$IMG" \
    /prof/hgemm_lt_probe.py 2>&1 | tail -12 | tee $OUT/hgemm_lt_probe_$name.txt
done
echo "=== probe complete ==="
