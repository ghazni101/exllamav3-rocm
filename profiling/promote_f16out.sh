#!/usr/bin/env bash
# Promote the f16out image to the standing serve, with a rollback point.
# Follows promote_tg1a.sh: rollback tag of the current base, retag the perf image as the base,
# rebuild the TabbyAPI overlay FROM it, recreate the serve, then poll for health.
#   promote_f16out.sh [image]
set -euo pipefail
IMG=${1:-exllamav3-rocm:perf-i}
COMPOSE=~/docker-containers/exllamav3-rocm/docker-compose.yml

docker image inspect "$IMG" >/dev/null || { echo "missing image $IMG"; exit 1; }
docker tag exllamav3-rocm:serve exllamav3-rocm:serve-pre-f16out
docker tag "$IMG" exllamav3-rocm:serve
docker build -f ~/github/tabbyAPI/Dockerfile.rocm -t tabbyapi-rocm:serve ~/github/tabbyAPI 2>&1 | tail -3
docker compose -f "$COMPOSE" up -d --force-recreate serve
for i in $(seq 1 40); do
  H=$(curl -s -o /dev/null -w '%{http_code}' http://192.168.1.200:9001/v1/model/list 2>/dev/null)
  [ "$H" = "200" ] && echo "serve healthy after ~$((i * 15))s" && exit 0
  sleep 15
done
echo "serve did not become healthy in 10 min"
exit 1
