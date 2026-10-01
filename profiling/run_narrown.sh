#!/usr/bin/env bash
# CU-mode NARROWN A/B vs the xor-16+rows40 keep (40.79). Env only.
# Usage: run_narrown.sh <rows> [image] [label]
set -uo pipefail
NROWS=${1:?narrown rows}
IMG=${2:-exllamav3-rocm:goal-isa-swizzle}
LABEL=${3:-isa-swizzle-r40-nn$NROWS}
ROOT=/home/ghazni/github/exllamav3-rocm
OUT=$ROOT/profiling/out_guided
export TG_REF_FILE=/out_guided/isa-swizzle-r40.json
export TG_FULL_LOG=1
~/gpu-coord/gpu-ctl reserve "CU NARROWN=$NROWS vs swizzle-r40 40.79" 20
rc=0
bash "$ROOT/profiling/run_goal_tg.sh" "$IMG" "$LABEL" \
  EXL3_SQ_GRID_MULT=2 EXL3_SQ_ROWS_PER=40 EXL3_SQ_ROWS_PER_NARROWN="$NROWS" \
  EXL3_SQ_LAUNCH_LOG=8 || rc=$?
~/gpu-coord/gpu-ctl done || true
python3 - "$OUT/$LABEL.json" << 'PY2'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
if not p.exists():
    print(p.name, "MISSING"); raise SystemExit(1)
d = json.loads(p.read_text())
tps = [round(r["engine_tps"], 3) for r in d["runs"]]
print(p.stem, "median", round(d["median_engine_tps"], 3),
      "runs", tps, "parity", d.get("token_parity"))
print("swizzle-r40 ref median 40.788")
PY2
exit $rc
