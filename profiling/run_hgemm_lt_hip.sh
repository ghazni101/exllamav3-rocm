#!/usr/bin/env bash
# Compile and run the standalone hipBLASLt probe inside the image (no extension rebuild needed).
#   run_hgemm_lt_hip.sh <image> <M> <K> <N> <reps>
set -uo pipefail
BASE=/home/ghazni/github/exllamav3-rocm/profiling
IMG=${1:-exllamav3-rocm:perf-f}
M=${2:-2048}; K=${3:-5120}; N=${4:-17408}; REPS=${5:-20}
DEV=/opt/rocm-venv/lib/python3.12/site-packages/_rocm_sdk_devel

docker run --rm --device /dev/kfd --device /dev/dri --group-add 44 --group-add 993 \
  --ipc host --shm-size 4g --ulimit nofile=65536:65536 \
  -v "$BASE":/prof:ro -v "$BASE/out_a":/out \
  --entrypoint bash "$IMG" -lc "
set -e
INC=/opt/rocm-venv/lib/python3.12/site-packages/_rocm_sdk_devel/include
LIB=/opt/rocm-venv/lib/python3.12/site-packages/_rocm_sdk_libraries/lib
echo \"include=\$INC lib=\$LIB\"
hipcc --offload-arch=gfx1100 -O3 -I\$INC -o /out/hgemm_lt_hiptest /prof/hgemm_lt_probe.hip -L\$LIB -lhipblaslt -Wl,-rpath,\$LIB
echo '=== small correctness shapes ==='
/out/hgemm_lt_hiptest 32 256 128 3
echo '=== model shape ==='
/out/hgemm_lt_hiptest $M $K $N $REPS
"
