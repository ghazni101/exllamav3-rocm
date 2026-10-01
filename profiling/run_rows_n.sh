#!/usr/bin/env bash
# Env-only rows_per A/B on the Hadamard keep. Usage: run_rows_n.sh <rows> [image] [label]
set -uo pipefail
ROWS=${1:?rows}
IMG=${2:-exllamav3-rocm:goal-isa-had}
LABEL=${3:-isa-had-r$ROWS}
ROOT=/home/ghazni/github/exllamav3-rocm
OUT=$ROOT/profiling/out_guided
export TG_REF_FILE=/out_guided/isa-had-r40.json
export TG_FULL_LOG=1
~/gpu-coord/gpu-ctl reserve "had MULT=2 rows_per=$ROWS vs had-r40 41.30" 20
rc=0
bash "$ROOT/profiling/run_goal_tg.sh" "$IMG" "$LABEL" \
  EXL3_SQ_GRID_MULT=2 EXL3_SQ_ROWS_PER="$ROWS" EXL3_SQ_LAUNCH_LOG=8 || rc=$?
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
print("had-r40 ref median 41.296")
PY2
exit $rc
