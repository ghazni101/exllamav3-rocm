# rocprofv3 on this container: the two blockers, diagnosed and fixed

Both failures reported earlier in this repo were real, but the diagnoses were guesses and the
workarounds (register-loaded env mode, giving up on `--attach`) silently dropped capabilities:
`--pmc` counter collection, HIP/API traces, and live-serve attach profiling. This document is the
measured root cause of each and the working invocation.

Everything below was reproduced with `tabbyapi-rocm:serve` (ROCm 10 SDK, rocprofiler-sdk 1.3.5,
torch 2.14.0+rocm7.14, triton-rocm).

---

## Blocker 1 — `rocprofv3 -- <app>` (LD_PRELOAD tool mode) segfaults

### What is actually happening (measured, not inferred)

The earlier note blamed `libLLVM`. That is wrong: **the tool libraries do not link libLLVM at all**.

```
$ objdump -p .../rocprofiler-sdk/librocprofiler-sdk-tool.so | grep NEEDED
  librocm_sysdeps_dw.so.1  libamd_comgr.so.3  librocm_sysdeps_elf.so.1
  librocm_sysdeps_sqlite3.so  librocprofiler-sdk.so.1  ...
$ nm -D --defined-only .../libamd_comgr.so.3 | grep -c llvm
0
```

The real mechanism, established by a 1-second CPU-only reproduction:

1. `librocprofiler-sdk.so` **exports 143 `std::filesystem` / `std::__cxx11` symbols of its own**
   (`nm -D --defined-only .../librocprofiler-sdk.so | grep -c filesystem` → 143), plus the
   matching `_ZTVSt`/`_ZTISt` typeinfo/vtables.
2. rocprofv3 LD_PRELOADs the tool, so those symbols land in the **global** scope.
3. `triton/_C/libtriton.so` (505 MB, statically links LLVM 23, exports 135,297 symbols) is dlopened
   later by CPython. Its own references to `std::filesystem`, `std::__cxx11::basic_string`, etc.
   resolve **to the tool's definitions first** (the global scope precedes the dlopened object's own
   scope), not to its bundled ones.
4. libtriton's LLVM was built against its own libstdc++ ABI, so the substituted implementations
   corrupt the heap. The process dies in `cfree` as soon as triton is imported.

Reproduction (no GPU, no torch, no profiler — run it to see both sides):

```python
# crash: tool libs in the global scope
import ctypes
ctypes.CDLL(".../librocprofiler-sdk-tool.so", mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(".../librocprofiler-sdk.so",     mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(".../triton/_C/libtriton.so")    # SIGSEGV in cfree

# no crash: same libs, local scope
ctypes.CDLL(".../librocprofiler-sdk-tool.so")
ctypes.CDLL(".../librocprofiler-sdk.so")
ctypes.CDLL(".../triton/_C/libtriton.so")    # triton OK
```

A second, weaker trigger exists (the tool's C++ gets corrupted by triton's exports in the other
direction); the shim below fixes both by keeping each library's own relocations bound to its own
definitions.

### Fix: `profiling/shims/ld_scope_shim.so`

Interposes `dlopen`/`dlmopen` and adds `RTLD_DEEPBIND` for libraries matching
`LD_SCOPE_DEEPBIND` (default `libtriton`). With deep binding, the dynamic linker searches the
newly loaded library's own scope ahead of the global scope for its own relocations, so libtriton
keeps its bundled libstdc++/LLVM.

```bash
gcc -shared -fPIC -O2 -o ld_scope_shim.so ld_scope_shim.c -ldl
LD_PRELOAD=$PWD/ld_scope_shim.so rocprofv3 --kernel-trace -f csv -d /out -- python3 app.py
LD_SCOPE_VERBOSE=1 ...          # logs each deep-bound dlopen
LD_SCOPE_DEEPBIND="libtriton:x" # extend the pattern list if another lib collides
```

Verified:

```
$ rocprofv3 --kernel-trace -f csv -d /out -- python3 -c "import torch, triton; \
    x=torch.zeros(1024,device='cuda'); print((x+1).sum().item())"
1024.0
E rocprofv3 output_stream.cpp:110] Opened result file: /out/<uuid>/1_kernel_trace.csv
```

and at the CLI level, without any GPU device exposed (the crash is at `dlopen` time, before any
driver access, so this needs no lock):

```
$ rocprofv3 --kernel-trace -f csv -d /out -- python3 -c "import triton; print('TRITON-OK')"
*** SIGSEGV received ... ***           # rc=139, no shim

$ LD_PRELOAD=/shims/ld_scope_shim.so rocprofv3 --kernel-trace ... -- python3 -c "import triton; ..."
[ld_scope_shim] deepbind: .../triton/_C/libtriton.so
TRITON-OK version 3.8.0
exit=0
```

Generalisation: this shim is the fix for *any* C++ library that statically links its own
libstdc++/LLVM and is dlopened into a process that already has a tool's symbols in the global
scope. It is not specific to exllamav3.

Scope, verified by elimination under the same CLI invocation:

| target | without shim | with shim |
|---|---|---|
| `python3 -c "import numpy"` | ok | ok |
| `python3 -c "import torch"` | ok | ok |
| `python3 -c "import torch, triton"` | **SIGSEGV in cfree** | ok |

So only `libtriton.so` collides — torch itself is unaffected (it does not bundle an LLVM that
exports these symbols). The shim is needed exactly when triton is imported.

---

## Blocker 2 — `rocprofv3 --attach PID` reports "attach support not enabled"

### What is actually happening (measured)

The target must expose a `rocp-bg-attach` thread, created by `librocprofiler-sdk-attach.so` when
that library is *invoked by* `librocprofiler-register.so`'s constructor — not when the attach
library is merely loaded.

Measured matrix (`torch` + `triton` target, first GPU work, thread count from `/proc/1/task`,
maps from `/proc/1/maps`, all inspected **inside** the container):

| target environment | `rocp-bg-attach` thread | attach lib mapped |
|---|---|---|
| (nothing) | 0 | 0 |
| `ROCP_TOOL_ATTACH=1` | 0 | 0 |
| `LD_PRELOAD=librocprofiler-register.so` | 0 | 0 |
| **`LD_PRELOAD=librocprofiler-register.so` + `ROCP_TOOL_ATTACH=1`** | **1** | **5** |
| `LD_PRELOAD=librocprofiler-sdk-attach.so` | 0 | 5 |
| `LD_PRELOAD=librocprofiler-sdk-attach.so` + `ROCP_TOOL_ATTACH=1` | 0 | 5 |
| `LD_PRELOAD=register.so:attach.so` + `ROCP_TOOL_ATTACH=1` | 1 | 5 |
| `HSA_TOOLS_LIB=librocprofiler-sdk-attach.so` + `ROCP_TOOL_ATTACH=1` | 0 | 0 |

Notes:
- `librocprofiler-register.so` is present in the process even with no env (12 rocprofiler
  mappings) — the HSA runtime loads it — but the **explicit `LD_PRELOAD` is what makes it run its
  constructor early enough to enable the attach handshake**. Both ingredients are required.
- Preloading the attach library directly is not enough: nothing calls into it.

### Fix (target side)

```bash
docker run ... \
  -e ROCP_TOOL_ATTACH=1 \
  -e LD_PRELOAD=/opt/rocm-venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/librocprofiler-register.so \
  --cap-add=SYS_PTRACE \
  <image> <app>
```

`--cap-add=SYS_PTRACE` is needed for the *attaching* side (docker's default seccomp blocks
`ptrace`); it does not need to be on the target.

### Attach invocation (profiler side)

**Resolution (session 2, later the same day): `rocprofv3 --attach` is itself the broken piece —
see the end of this section for the working recipe.**

```bash
rocprofv3 --attach <pid> --attach-children=false --attach-duration-msec 10000 \
  -f csv -d /out --kernel-trace
```

- **Use `--attach-children=false`.** The default walks the whole descendant tree; when the attach
  command itself is run via `docker exec` (a sibling of PID 1, not a descendant), the tree walk
  can hang indefinitely. Observed: a 10 s attachment still alive after 3.5 minutes.
- **Do not pipe the attach command's stdout through `tail`/`head`.** `rocprof-attach` prints only
  when it is done (after attach + duration + detach), so a buffered pipe shows nothing and a hang
  looks like silence. Redirect to a file and poll it.
- **Detach is slow by design**: `rocprof-attach` itself prints "Detaching. Please wait, this can
  take up to 1-2 minutes". Budget ~2 minutes after the attach window before declaring a hang.
- Attach durations are wall-clock; the profiler must be given a window in which the target is
  actually issuing work.

### Session 2 resolution: the handshake works — via `rocprof-attach` directly, on a quiescent target

Measured 2×2 (route × target state), all with the target correctly enabled per the matrix above
plus `ulimits nofile 65536` (docker exec defaults to 1024, which the tooling needs raised — the
SIGTERM path prints `Unable to get high fd … limit=1024` otherwise):

| route | idle target | target under HTTP load |
|---|---|---|
| `rocprofv3 --attach 1` | **hangs** (rc=124 at 4 min, both sides R-spinning) | hangs |
| `rocprof-attach` invoked directly | **SUCCESS** (attach `:: success`, detach `:: success`, rc=0) | hangs |

Working recipe (each step load-bearing):

```bash
# 1. target side (compose override): ROCP_TOOL_ATTACH=1 + LD_PRELOAD=librocprofiler-register.so
#    + cap_add [SYS_PTRACE] + ulimits nofile {soft: 65536}   -> profiling/attach_override.yml
# 2. verify: /proc/1/task/*/comm contains rocp-bg-attach (state S at idle, zero CPU cost)
# 3. target must be QUIESCENT when the attach lands
# 4. attach client, direct, unbuffered, config via client env:
docker exec exllamav3-rocm-serve sh -c "
  env ROCPROF_KERNEL_TRACE=1 ROCPROF_OUTPUT_PATH=/tmp/prof ROCPROF_OUTPUT_FORMAT=csv \
  python3 -u /opt/rocm-venv/lib/python3.12/site-packages/_rocm_sdk_devel/bin/rocprof-attach \
    -p 1 --attach-children=false \
    -t /opt/rocm-venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so \
    -d 60000"
# 5. drive workload only after the client prints "Attaching for 60000 msec"
```

Notes:

- `python3 -u` matters: the client is a python script that block-buffers stdout — earlier
  sessions' "silence" was partly this.
- The tool configuration does not come from `rocprofv3`'s wrapper; `rocattach` serializes the
  **client's environment** (`ROCPROF_KERNEL_TRACE`, `ROCPROF_OUTPUT_PATH`, …) into the target
  (`build_environment_buffer()` in rocattach.cpp), where `librocprofiler-sdk-tool.so` picks it up.
- Mechanism (from rocprofiler-sdk source, `source/lib/rocprofiler-sdk-rocattach/`): find the
  `rocp-bg-attach` thread by name, ptrace-session the target, write env buffer + tool path into
  the target, hijack the bg-attach thread to call
  `librocprofiler-register.so::rocprofiler_register_attach`. Under load the ptrace stop/iterate
  over ~90 live python threads never settles — hence the quiescent requirement.
- `rocprofv3 --attach` fails through this same library; its wrapper adds a sibling-process pipe
  dance that never completes on this SDK. Untested upstream whether a newer SDK fixes the
  wrapper; direct `rocprof-attach` is the workaround.

---

## What this unlocks (the reason it mattered)

| capability | register-loaded env mode (previous workaround) | `rocprofv3` CLI (now working) |
|---|---|---|
| kernel trace | yes | yes |
| **PMC counters** | `FETCH_SIZE`/`GL2C_*` silently returned 0 | `--pmc` with the frontend's counter resolution |
| HIP/API trace | env only, unverified | `--hip-trace` / `--sys-trace` |
| multi-pass counters | no | `--pmc "G1..." --pmc "G2..."` |
| available-counter listing | no | `-L` / `rocprofv3-avail` |
| attach a **running** serve | no | yes (`--attach`) |
| OTF2 / pftrace / Perfetto output | csv/json only | full format list |

The `FETCH_SIZE = 0` result in the earlier analysis was therefore not trustworthy: it came from an
invocation path that the frontend never configures. Re-measuring DRAM traffic under the CLI is
what the tg analysis (GEMV at "~60 % of peak") actually needs.
