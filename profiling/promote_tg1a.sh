#!/usr/bin/env bash
# Promote tg-1a image to the standing serve, with rollback point.
#   1. tag rollback point of the current base
#   2. retag perf-tg1a as exllamav3-rocm:serve
#   3. rebuild the tabbyapi overlay FROM the new base (cheap: deps + COPY)
#   4. recreate the serve (picks up warmup entrypoint + safe_defaults + new ext)
set -euo pipefail
docker tag exllamav3-rocm:serve exllamav3-rocm:serve-pre-tg1a
docker tag exllamav3-rocm:perf-tg1a exllamav3-rocm:serve
docker build -f ~/github/tabbyAPI/Dockerfile.rocm -t tabbyapi-rocm:serve ~/github/tabbyAPI 2>&1 | tail -3
sg docker -c "docker compose -f ~/docker-containers/exllamav3-rocm/docker-compose.yml up -d --force-recreate serve"
for i in $(seq 1 40); do
  H=$(curl -s -o /dev/null -w '%{http_code}' http://192.168.1.200:9001/v1/model/list 2>/dev/null)
  [ "$H" = "200" ] && echo "serve healthy after ~$((i*15))s" && exit 0
  sleep 15
done
echo "serve did not become healthy in 10 min"; exit 1
