#!/usr/bin/env python3
"""Extract gfx1100 device code objects from a HIP fat binary (extension .so or hipcc output).

The device ISA is what the perf work reasons about (issue mix, inner-loop bodies, register
pressure); `llvm-objdump -d` on a fat .so shows only host stubs, and this SDK's roc-obj-* helpers
are broken (import error), so the clang-offload-bundler container format is parsed directly.
Bundles inside `.hip_fatbin` are concatenated with 4 KB alignment padding, so bundles are located
by scanning for the magic rather than by computing sizes.

Usage:
  extract_isa.py <fat_binary> <out_dir>              # .so / fat executable: dumps .hip_fatbin
  extract_isa.py --bundle <bundle_file> <out_dir>    # an already-dumped bundler container
Writes <out_dir>/co_<n>_<idx>.o per gfx1100 member and prints a manifest line per object.
"""
import os
import struct
import subprocess
import sys

MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"


def bundles(data):
    """Yield (bundle_start, [(entry_id, offset, size), ...]) for every bundle in `data`."""
    pos = data.find(MAGIC)
    while pos >= 0:
        (n,) = struct.unpack_from("<Q", data, pos + len(MAGIC))
        p = pos + len(MAGIC) + 8
        entries = []
        for _ in range(n):
            off, size, idlen = struct.unpack_from("<QQQ", data, p)
            p += 24
            eid = data[p:p + idlen].decode()
            p += idlen
            entries.append((eid, off, size))
        yield pos, entries
        pos = data.find(MAGIC, p)


def main():
    args = sys.argv[1:]
    section = ".hip_fatbin"
    if "--section" in args:
        i = args.index("--section")
        section = args[i + 1]
        del args[i:i + 2]
    standalone = "--bundle" in args
    if standalone:
        args.remove("--bundle")
    src, outdir = args[0], args[1]
    os.makedirs(outdir, exist_ok=True)

    if standalone:
        data = open(src, "rb").read()
    else:
        tmp = os.path.join(outdir, "_fatbin.bin")
        subprocess.run([os.environ.get("LLVM_OBJCOPY", "llvm-objcopy"),
                        "--dump-section", f"{section}={tmp}", src], check=True)
        data = open(tmp, "rb").read()

    n = 0
    for bpos, entries in bundles(data):
        for idx, (eid, off, size) in enumerate(entries):
            if "gfx1100" not in eid or size == 0:
                continue
            blob = data[bpos + off: bpos + off + size]
            if blob[:4] != b"\x7fELF":
                print(f"skip {eid}: not ELF ({len(blob)} bytes)")
                continue
            name = f"co_{n:03d}_{idx}.o"
            with open(os.path.join(outdir, name), "wb") as f:
                f.write(blob)
            print(f"{name}: {size:8d} bytes  {eid}")
            n += 1
    print(f"extracted {n} gfx1100 code object(s) from {src}")


main()
