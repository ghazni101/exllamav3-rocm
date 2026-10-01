#!/usr/bin/env bash
# CU-mode MULT=2 rows_per=48 vs the 32 default keep (38.36). Env only.
set -uo pipefail
IMG=${1:-exllamav3-rocm:goal-isa-cumode}
LABEL=${2:-isa-cumode-r48}
ROOT=/home/ghazni/github/exllamav3-rocm
OUT=$ROOT/profiling/out_guided
export TG_REF_FILE=/out_guided/isa-cumode-g2.json
export TG_FULL_LOG=1
~/gpu-coord/gpu-ctl reserve "CU MULT=2 rows_per=48 vs g2 38.36" 20
rc=0
bash "$ROOT/profiling/run_goal_tg.sh" "$IMG" "$LABEL" \
  EXL3_SQ_GRID_MULT=2 EXL3_SQ_ROWS_PER=48 EXL3_SQ_LAUNCH_LOG=8 || rc=$?
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
print("cumode-g2 ref median 38.361")
PY2
exit $rc
