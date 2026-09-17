"""Shared cold-rotation GEMV bench machinery for the sq/int8 kernels (plan steps A2/A3).

Timing discipline (from msq_ab2.py / findings-log 11.3): in-model decode re-reads a weight tensor
after ~13 GB of other traffic, so a loop over ONE tensor measures an Infinity-Cache-resident
kernel (96 MB IC), not production. Every timed call here streams a *different* same-shape layer
instance, and the pool is at least ROT instances deep (default 6); a pool smaller than the cache is
flagged IC_resident so its GB/s is never quoted as a DRAM rate.

K discovery: `Linear.inner` only exists after load, so K comes from the checkpoint's safetensors
header (trellis shape[2] / 16) joined by tensor key - the same source the loader uses. That lets a
pool be selected per K without loading (and paying for) instances of other K.

dp4a / instruction-rate conversion (measured on the installed extension's ISA - every sq unit
issues 16 v_dot4 per (16 k-rows x 32 columns) unit row, i.e. one dp4a per 32 weights, K-independent
- see out_a/isa_ext/isa.txt and profiling/isa_census.py):

    dp4a/byte   = 0.25 / K     (K=3 0.0833, K=4 0.0625, K=5 0.05, K=6 0.0417)
    dp4a/weight = 0.03125

so a measured GB/s converts to G warp-dp4a/s as GB/s * 0.25/K and G weights/s as GB/s * 8/K.
Instruction mixes per unit row, also from the ISA (v_dot4 / v_mul_lo_u32 / other / bytes):

    K=4 wide   16 / 16 / 52 / 256 B      K=3 smem  16 / 16 / 50 / 192 B
    K=5 smem   16 / 16 / 72 / 320 B      K=6 narrow 16 / 16 / 66 / 384 B
"""
import collections
import json
import os
import struct

import torch
from exllamav3 import Config, Model

IC_BYTES = 96 * 2 ** 20


def trellis_bytes(k, n, K):
    return k * n * K // 8


def read_header(model_dir):
    """-> {key_without_suffix: (K, k, n)} for every EXL3 trellis tensor in the checkpoint."""
    out = {}
    for fn in os.listdir(model_dir):
        if not fn.endswith(".safetensors"):
            continue
        with open(os.path.join(model_dir, fn), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        for key, ent in hdr.items():
            if key == "__metadata__" or not key.endswith(".trellis"):
                continue
            sh = ent["shape"]
            if len(sh) != 3 or sh[2] % 16:
                continue
            out[key[:-len(".trellis")]] = (sh[2] // 16, sh[0] * 16, sh[1] * 16)
    return out


def k_of(key, header, index=None):
    """Resolve a module key to its K through the header, tolerating front-prefix differences.
    Longest suffix with at least two dotted components wins; an ambiguous suffix resolves to None
    rather than to an arbitrary layer's K."""
    if index is None:
        index = suffix_index(header)
    parts = key.split(".")
    for i in range(len(parts)):
        ks = index.get(".".join(parts[i:]))
        if ks is not None and len(ks) == 1:
            return header[next(iter(ks))]
    return None


def suffix_index(header, min_parts=1):
    """-> {suffix: {full_key, ...}} for every suffix of every header key of >= min_parts parts."""
    out = collections.defaultdict(set)
    for key in header:
        parts = key.split(".")
        for i in range(len(parts) - min_parts + 1):
            out[".".join(parts[i:])].add(key)
    return out


def walk_modules(root, seen=None):
    """Depth-first over the registered submodule tree (Module.modules), deduplicated."""
    if seen is None:
        seen = set()
    if id(root) in seen:
        return
    seen.add(id(root))
    yield root
    for sub in getattr(root, "modules", None) or []:
        yield from walk_modules(sub, seen)


def enumerate_linears(model_dir):
    """-> (model, [(K, k, n, linear)]) over every quantized Linear in the module tree, without
    loading weights. The tree walk uses the registered submodules, not attribute names: the model
    nests linears under attn/mlp/linear_attn and a name-based walk silently misses most of them."""
    header = read_header(model_dir)
    index = suffix_index(header)
    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    found = []
    seen = set()
    for node in walk_modules(model):
        if type(node).__name__ != "Linear" or id(node) in seen:
            continue
        seen.add(id(node))
        k, n = node.in_features, node.out_features
        if k % 128 or n % 256:
            continue
        hit = k_of(getattr(node, "key", ""), header, index)
        if hit is None:
            continue
        found.append((hit[0], k, n, node))
    return model, found


def pick_pool(found, K, rot, n_max=32768, prefer_n=None):
    """The widest (k, n) at this K with at least `rot` instances ->
    (k, n, [linear], instances_available)."""
    groups = collections.defaultdict(list)
    for K_, k, n, lin in found:
        if K_ == K and n % 256 == 0 and k % 128 == 0 and n <= n_max:
            if prefer_n is None or n == prefer_n:
                groups[(k, n)].append(lin)
    best = None
    for (k, n), lins in groups.items():
        if len(lins) < rot:
            continue
        if best is None or k * n > best[0] * best[1]:
            best = (k, n, lins, len(lins))
    return best


def load_pool(pool, device="cuda:0"):
    for lin in pool:
        lin.load(device=device)
    return pool


def pool_report(k, n, K, n_inst):
    by = trellis_bytes(k, n, K)
    total = by * n_inst
    return dict(k=k, n=n, K=K, instances=n_inst, bytes_mb=round(by / 1e6, 2),
                pool_mb=round(total / 1e6, 2), ic_resident=total < IC_BYTES)


def bench(fns, reps=6, warm=2):
    for _ in range(warm):
        for f in fns:
            f()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        for f in fns:
            f()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / (reps * len(fns))


def rates(k, n, K, m, t_ms_per_call, n_inst=None):
    """Derived rates for one (K, m) measurement. t_ms_per_call is bench()'s per-call mean (it
    already divides by the pool depth - the earlier version divided again and inflated every
    GB/s by the rotation depth). One call streams `by` bytes of trellis regardless of m; the
    dp4a/weight rates scale with m because m rows share one extraction pass."""
    by = trellis_bytes(k, n, K)
    gbps = by / (t_ms_per_call * 1e6)
    return {
        "t_ms_per_call": round(t_ms_per_call, 4),
        "gbps": round(gbps, 1),
        "g_dp4a_s": round(gbps * 0.25 / K * m, 2),
        "g_weights_s": round(gbps * 8 / K * m, 2),
        "instances": n_inst,
    }
