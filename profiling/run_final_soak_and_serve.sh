#!/usr/bin/env bash
# Final session: re-run the soak with the (fixed) post-load leak baseline on the shipping image,
# then restore and verify the standing serve.
#   run_final_soak_and_serve.sh <image>
set -uo pipefail
BASE=/home/ghazni/github/exllamav3-rocm/profiling
OUT=$BASE/out_a
IMG=${1:-exllamav3-rocm:perf-c}

echo "=== G4 (re-run): soak with the post-load leak baseline ==="
docker run --rm --name exl3-soak2 --device /dev/kfd --device /dev/dri \
  --group-add 44 --group-add 993 --ipc host --shm-size 4g --ulimit nofile=65536:65536 \
  -e EXL3_MODEL=/models/Qwen3.8-27B-SC_4.00bpw_H5_V6 -e EXL3_CACHE_TOKENS=32768 \
  -e SOAK_MINUTES=10 \
  -v /home/ghazni/models/exl3/turboderp:/models:ro -v $BASE:/prof:ro -v $OUT:/out \
  --entrypoint python3 "$IMG" /prof/soak.py 2>&1 | tail -3 | tee $OUT/s5_soak2.txt

echo "=== restore the standing serve ==="
docker start exllamav3-rocm-serve
for i in $(seq 1 40); do
  if curl -s -m 3 http://192.168.1.200:9001/v1/model > /dev/null 2>&1; then echo "serve responding after ${i} polls"; break; fi
  sleep 5
done
curl -s -m 5 http://192.168.1.200:9001/v1/model | head -c 300; echo
docker ps --filter name=exllamav3-rocm-serve --format '{{.Names}} {{.Status}}'
echo "=== done ==="
