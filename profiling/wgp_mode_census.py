#!/usr/bin/env python3
"""Count .workgroup_processor_mode bytes in extracted gfx1100 code objects.

msgpack uint: 0x00 = CU mode, 0x01 = WGP mode (gfx11 default).
Usage: wgp_mode_census.py <co_dir>
"""
import sys
from collections import Counter
from pathlib import Path

key = b".workgroup_processor_mode"
root = Path(sys.argv[1])
c = Counter()
missing = 0
n = 0
for f in sorted(root.glob("*.o")):
    n += 1
    data = f.read_bytes()
    i = 0
    found = False
    while True:
        j = data.find(key, i)
        if j < 0:
            break
        found = True
        c[data[j + len(key)]] += 1
        i = j + len(key)
    if not found:
        missing += 1
print("objects", n, "missing_key", missing)
print("mode_bytes", {hex(k): v for k, v in sorted(c.items())})
cu = c.get(0x00, 0)
wgp = c.get(0x01, 0)
if wgp and not cu:
    print("verdict WGP")
elif cu and not wgp:
    print("verdict CU")
else:
    print("verdict MIXED_OR_EMPTY")
