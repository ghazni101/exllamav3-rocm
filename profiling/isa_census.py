#!/usr/bin/env python3
"""Instruction-mix census over a dumped AMDGPU ISA listing (see profiling/isa_dump.sh).

Finds the loops of the requested kernel instantiations and reports, per loop, the instruction
histogram (and optionally the ordered operand-producing instructions). Loop detection is
structural: a backward branch to `<same symbol>+<offset below the branch's own offset>` closes a
loop whose body is every instruction in [target, branch] - enough for the fully unrolled per-
k-row bodies of the EXL3 GEMV units, without reconstructing control flow.

Usage:
  isa_census.py <isa.txt> <symbol-regex> [--all-loops] [--dump-inner]
"""
import bisect
import re
import sys
from collections import Counter

FUNC_RE = re.compile(r"^([0-9a-f]+) <(.+)>:$")
INSN_RE = re.compile(r"^\s+([a-z][a-z0-9_.]*)\s")
ADDR_RE = re.compile(r"//\s*([0-9A-Fa-f]{8,})")
BR_RE = re.compile(r"<(\S+)\+0x([0-9a-f]+)>")
MEM = ("ds_read", "global_load", "buffer_load", "flat_load", "ds_write")


def parse(path):
    """-> ordered {symbol: (base_addr, [(addr, mnemonic, line), ...])}"""
    funcs = {}
    cur = None
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("##"):
                cur = None
                continue
            m = FUNC_RE.match(line)
            if m:
                cur = funcs.setdefault(m.group(2), (int(m.group(1), 16), []))[1]
                continue
            if cur is None:
                continue
            im = INSN_RE.match(line)
            if not im:
                continue
            am = ADDR_RE.search(line)
            cur.append((int(am.group(1), 16) if am else None, im.group(1), line.rstrip()))
    return funcs


def loops(insns, base):
    """-> [(start, end, [insns])] over backward branches (branch targets are symbol-relative)."""
    addrs = [a for a, _m, _l in insns]
    out = []
    for addr, mn, line in insns:
        if not mn.startswith("s_cbranch") and mn != "s_branch":
            continue
        for _sym, off in BR_RE.findall(line):
            off = base + int(off, 16)
            if addr is None or off >= addr:
                continue
            i0 = bisect.bisect_left(addrs, off)
            i1 = bisect.bisect_right(addrs, addr)
            if i1 > i0:
                out.append((off, addr, insns[i0:i1]))
    return out


def summarize(body):
    h = Counter(m for _a, m, _l in body)
    nd = h.get("v_dot4", 0) + h.get("v_dot2", 0)
    mem = sum(v for k, v in h.items() if k.startswith(MEM))
    other = len(body) - nd - mem
    print(f"    body: {len(body)} insns  v_dot={nd}  mem={mem}  other={other}")
    print("      " + "  ".join(f"{k}:{v}" for k, v in h.most_common()))


def main():
    path, pat = sys.argv[1], sys.argv[2]
    all_loops = "--all-loops" in sys.argv
    dump = "--dump-inner" in sys.argv
    rx = re.compile(pat)
    funcs = parse(path)
    hits = 0
    for sym, (base, insns) in funcs.items():
        if not rx.search(sym):
            continue
        hits += 1
        print(f"\n=== {sym}  ({len(insns)} instructions) ===")
        for (s, e, body) in loops(insns, base):
            nd = sum(1 for _a, m, _l in body if m.startswith("v_dot"))
            if not all_loops and not nd:
                continue
            print(f"  loop 0x{s:x}-0x{e:x}")
            summarize(body)
            if dump and len(body) <= 60:
                for a, m, l in body:
                    print(f"      {a:x}  {m}")
    print(f"\nmatched {hits} function(s)")


main()
