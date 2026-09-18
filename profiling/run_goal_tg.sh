#!/usr/bin/env bash
# Goal TG benchmark: 4096/256 autoregressive, batch-1, greedy, no speculation.
#   run_goal_tg.sh <image> <label> [ENV=VAL ...]
# Output: profiling/out_guided/<label>.json (+ token parity vs final-post-revert.json
# or $TG_REF_FILE when set)
set -uo pipefail
BASE=/home/ghazni/github/exllamav3-rocm/profiling
OUT=$BASE/out_guided
IMG=${1:?image}; LABEL=${2:?label}; shift 2
envs=()
for kv in "$@"; do envs+=(-e "$kv"); done
REF=${TG_REF_FILE:-/out_guided/final-post-revert.json}
refargs=()
[ -f "$OUT/$(basename "$REF")" ] && refargs+=(-e "TG_REFERENCE=$REF")
mounts=()
for m in ${EXTRA_MOUNTS:-}; do mounts+=(-v "$m"); done

docker run --rm --name exl3-goal --device /dev/kfd --device /dev/dri \
  --group-add 44 --group-add 993 --ipc host --shm-size 4g --ulimit nofile=65536:65536 \
  -e EXL3_MODEL=/models/qwen38-27b \
  -e TG_OUTPUT=/out_guided/$LABEL.json \
  "${refargs[@]}" \
  "${mounts[@]}" \
  "${envs[@]}" \
  -v /home/ghazni/models/exl3/Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw:/models/qwen38-27b:ro \
  -v $BASE:/prof:ro -v $OUT:/out_guided \
  --entrypoint python3 "$IMG" /prof/guided_tg.py 2>&1 | tail -8
