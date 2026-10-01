#!/usr/bin/env bash
# Patch K=3 extract, rebuild goal-isa-k3ext, A/B vs had-r40 41.30.
set -euo pipefail
ROOT=/home/ghazni/github/exllamav3-rocm
cd "$ROOT"
python3 << 'PY'
from pathlib import Path

p = Path('exllamav3/exllamav3_ext/quant/exl3_gemv_int8_kernel.cuh')
s = p.read_text()
old = '''        w7 = fshift(b, a, s2);
        w6 = w7 >> bits;
        w5 = w6 >> bits;
        w4 = w5 >> bits;
        w3 = fshift(b, a, s2 + bits * 4);
        w2 = w3 >> bits;
        w1 = w2 >> bits;
        w0 = w1 >> bits;
        w7 &= 0xffff; w6 &= 0xffff; w5 &= 0xffff; w4 &= 0xffff;
        w3 &= 0xffff; w2 &= 0xffff; w1 &= 0xffff; w0 &= 0xffff;
'''
new = '''        // Independent 16-bit extracts. Serial `w >>= 3` then mask is bit-identical
        // but 4-deep VALU-dependent; BFE16 from the unmasked 32-bit windows is not.
        // K=3 is 7.76 ms of the 18.30 ms GEMV block on this model.
        uint32_t u = fshift(b, a, s2);
        uint32_t v = fshift(b, a, s2 + bits * 4);
        w7 = u & 0xffff;
        BFE16_IMM(w6, u, 3);
        BFE16_IMM(w5, u, 6);
        BFE16_IMM(w4, u, 9);
        w3 = v & 0xffff;
        BFE16_IMM(w2, v, 3);
        BFE16_IMM(w1, v, 6);
        BFE16_IMM(w0, v, 9);
'''
n = s.count(old)
if n != 1:
    raise SystemExit(f'kernel old count {n}')
p.write_text(s.replace(old, new, 1))
print('kernel patched')

p = Path('docs/rocprofv3-findings-log.md')
s = p.read_text()
a = 'Open: rows_per=36 (hole between 32 and 40). Do not call'
b = 'rows_per=36 rounds to 40; 41.18 ≈ 41.30. Open: K=3 extract ILP. Do not call'
if a in s:
    s = s.replace(a, b, 1)
    print('s21 updated')
else:
    print('s21 skip')
a = 'Next: rows_per=36 (only hole between 32 and 40).\n'
b = '''Next was rows_per=36 — invalid, rounds to 40.

## 38. Session ISA (2026-09-23): rows_per=36 is a no-op (rounds to 40)

`EXL3_SQ_ROWS_PER=36` on `goal-isa-had`. Host `(n+7)&~7` → 40. Launch log
`rows_per=40`, grid=384. Token parity true.

| arm | median tok/s | runs | parity |
|---|---|---|---|
| **rows=40** (`isa-had-r40.json`) | **41.30** | 41.39 / 41.30 / 41.19 | (ref) |
| ROWS=36 (`isa-had-r36.json`) | 41.18 | 41.28 / 41.18 / 41.13 | **true** |

−0.3%, overlapping. No hole between 32 and 40: legal values are multiples of 8,
`>= SQ_MINROWS=16`. Occupancy-shape closed. Next: K=3 `ext8w` independent BFE
(7.76 ms/token of GEMV; serial `>>3` chain vs K=4's independent `v_bfe`).
'''
if a in s:
    s = s.replace(a, b, 1)
    print('s38 appended')
else:
    print('s38 skip')
p.write_text(s)

p = Path('doc/env_vars.md')
s = p.read_text()
a = 'Occupancy peak is 40. Compile default 40'
b = 'Occupancy peak is 40. 36 rounds to 40 and matches 41.30 within noise. Compile default 40'
if a in s:
    p.write_text(s.replace(a, b, 1))
    print('env_vars updated')
else:
    print('env_vars skip')
PY

echo '=== docker build goal-isa-k3ext ==='
docker build -f Dockerfile.goal --build-arg EXL3_CUMODE=1 -t exllamav3-rocm:goal-isa-k3ext .
echo '=== A/B ==='
OUT=$ROOT/profiling/out_guided
export TG_REF_FILE=/out_guided/isa-had-r40.json
export TG_FULL_LOG=1
~/gpu-coord/gpu-ctl reserve "K=3 independent BFE extract vs had-r40 41.30" 25
rc=0
bash "$ROOT/profiling/run_goal_tg.sh" exllamav3-rocm:goal-isa-k3ext isa-had-k3ext \
  EXL3_SQ_GRID_MULT=2 EXL3_SQ_ROWS_PER=40 EXL3_SQ_LAUNCH_LOG=8 || rc=$?
~/gpu-coord/gpu-ctl done || true
python3 - "$OUT/isa-had-k3ext.json" << 'PY2'
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
