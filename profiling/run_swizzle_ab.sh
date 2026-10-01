#!/usr/bin/env bash
# xor-16 DS_SWIZZLE A/B on CU-mode+MULT=2+rows_per=48 vs the 40.08 keep.
set -uo pipefail
IMG=${1:-exllamav3-rocm:goal-isa-swizzle}
LABEL=${2:-isa-swizzle-r48}
ROOT=/home/ghazni/github/exllamav3-rocm
OUT=$ROOT/profiling/out_guided
export TG_REF_FILE=/out_guided/isa-cumode-r48.json
export TG_FULL_LOG=1
~/gpu-coord/gpu-ctl reserve "xor-16 swizzle vs r48 40.08" 20
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
print("cumode-r48 ref median 40.082")
PY2
exit $rc
