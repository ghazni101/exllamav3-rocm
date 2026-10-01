#!/usr/bin/env bash
# Recapture b1 decode mix at CTX=4096 on the CU-mode keep (rows_per=40, 41.30).
# torch.profiler; numbers are shares, not tok/s (profiler overhead).
set -uo pipefail
IMG=${1:-exllamav3-rocm:goal-isa-had}
TAG=${2:-had-r40}
ROOT=/home/ghazni/github/exllamav3-rocm
OUT=$ROOT/profiling/out_guided
~/gpu-coord/gpu-ctl reserve "decode composition CTX=4096 $TAG" 25
rc=0
docker run --rm --name "exl3-goal-decode-$TAG" \
  --device /dev/kfd --device /dev/dri \
  --group-add 44 --group-add 993 --ipc host --shm-size 4g \
  --ulimit nofile=65536:65536 \
  -e EXL3_MODEL=/models/qwen38-27b \
  -e PYTHONUNBUFFERED=1 \
  -e EXL3_SQ_GRID_MULT=2 \
  -e EXL3_SQ_ROWS_PER=40 \
  -e CTX=4096 \
  -e DECODE_N=64 \
  -e TAG="$TAG" \
  -e OUT_DIR=/out_guided \
  -e EXL3_CACHE_TOKENS=8192 \
  -v /home/ghazni/models/exl3/Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw:/models/qwen38-27b:ro \
  -v "$ROOT/profiling":/prof:ro \
  -v "$OUT":/out_guided \
  --entrypoint python3 "$IMG" /prof/decode_profile.py \
  2>&1 | tee "$OUT/${TAG}_profile.log" || rc=$?
~/gpu-coord/gpu-ctl done || true
exit $rc
