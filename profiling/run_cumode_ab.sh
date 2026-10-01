#!/usr/bin/env bash
# CU-mode 4096/256 A/B vs xmask (35.67). One compile variable (-mcumode).
# MULT=1 first (pure flag). MULT=2 only if launch log still shows sms=48 maxb=1,
# i.e. occupancy API did not already double the grid.
set -uo pipefail
IMG=${1:-exllamav3-rocm:goal-isa-cumode}
ROOT=/home/ghazni/github/exllamav3-rocm
OUT=$ROOT/profiling/out_guided
export TG_REF_FILE=/out_guided/isa-xmask.json
export TG_FULL_LOG=1

summarize() {
  local label=$1
  python3 - "$OUT/$label.json" << 'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
if not p.exists():
    print(p.name, "MISSING"); raise SystemExit(1)
d = json.loads(p.read_text())
tps = [round(r["engine_tps"], 3) for r in d["runs"]]
print(p.stem, "median", round(d["median_engine_tps"], 3),
      "runs", tps, "parity", d.get("token_parity"))
PY
}

~/gpu-coord/gpu-ctl reserve "CU-mode vs xmask 35.67 MULT=1 then 2" 30
rc=0
bash "$ROOT/profiling/run_goal_tg.sh" "$IMG" isa-cumode \
  EXL3_SQ_LAUNCH_LOG=16 || rc=$?
summarize isa-cumode || rc=$?

# Second arm only if the first produced a JSON (kernel did not fail to launch).
if [ -f "$OUT/isa-cumode.json" ]; then
  bash "$ROOT/profiling/run_goal_tg.sh" "$IMG" isa-cumode-g2 \
    EXL3_SQ_GRID_MULT=2 EXL3_SQ_LAUNCH_LOG=16 || rc=$?
  summarize isa-cumode-g2 || rc=$?
fi
~/gpu-coord/gpu-ctl done || true
echo "xmask ref median 35.667 runs [35.777, 35.667, 35.615]"
exit $rc
