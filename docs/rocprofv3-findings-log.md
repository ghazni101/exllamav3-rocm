# rocprofv3 session log — blockers, fixes, evidence, open items

Date: 2026-09-16. Host: RX 7900 XTX (gfx1100), container `tabbyapi-rocm:serve`
(ROCm 10 SDK, rocprofiler-sdk 1.3.5, torch 2.14.0+rocm7.14, triton-rocm 3.8.0).

Objective of the session: stop treating the two rocprofv3 failures as unavoidable (the earlier
note gave a wrong root cause for the first and abandoned the second), fix them, and profile
properly with the CLI instead of the register-loaded env workaround.

Status at end of session:

| item | status |
|---|---|
| Blocker 1 (LD_PRELOAD tool mode segfaults) | **SOLVED, verified at three levels** |
| Blocker 2 (attach) — target-side enablement | **SOLVED, verified (thread + library mapped)** |
| Blocker 2 (attach) — attaching handshake completing | **SOLVED (session 2): attach works via `rocprof-attach` invoked directly on a quiescent target; `rocprofv3 --attach` wrapper is itself broken (see §9)** |
| DRAM-traffic counters (`FETCH_SIZE`, `GL2C_*`, `GRBM_*`) | **UNAVAILABLE on this stack — closed with a full counter census (§8)** |
| Full-model CLI / live-serve profiling run | **DONE (session 2): complete decode+prefill CLI trace; live-serve profile captured through the attach route (§9)** |

---

## 8. Session 2 (16:00–16:40 UTC): full CLI profile + counter census + attach solved

### 8.1 Final counter census — DRAM traffic is unmeasurable on this stack

Every candidate counter was run through the official CLI (`--pmc`, `--cap-add=PERFMON`) against a
probe with known traffic (4× `sum()` + 4× `mul` on a 1 GiB fp32 tensor ≈ 4.3 GB read / 8.6 GB
read+write). Three disjoint outcomes:

| outcome | counters |
|---|---|
| **Rejected outright** (`error code 38: Request exceeds the capabilities of the hardware to collect`; the whole `--pmc` set is refused, nothing collected) | any set containing `GL2C_EA_RDREQ_*`, `GL2C_HIT_sum`, `GL2C_MISS_sum`, or the derived `FETCH_SIZE`, `WRITE_SIZE`, `GPUBusy`, `GPU_UTIL`, `L2CacheHit` |
| **Collectible but identically 0** on the known-traffic probe | `GL2C_MC_RDREQ_sum`, `GL2C_MC_WRREQ_sum`, `SQ_INST_CYCLES_VMEM`, `SQ_INSTS_FLAT`, `SQ_INSTS_VALU`, `SQ_INSTS_SMEM`, `SFetchInsts`, `SQ_WAVE_CYCLES`, `TA_BUFFER_LOAD_WAVEFRONTS`, `MemUnitBusy`, `ALUStalledByLDS` |
| **Alive** | `SQ_WAVES`, `SQ_BUSY_CYCLES`, `Wavefronts` (= `SQ_WAVES`) |

Host-level alternatives were also checked and ruled out: there is no `amdgpu` PMU in
`/sys/bus/event_source/devices/` (only CPU-side PMUs), so `perf` cannot measure VRAM traffic
either. **Consequence: every "GB/s" figure for this GPU remains inferred from timing + the
traffic model. Do not re-attempt memory counters on SDK 1.3.5/gfx1100; re-check only after an
SDK upgrade.**

### 8.2 Complete CLI kernel trace of the model (decode + prefill + batch)

`profiling/run_b_flagiso.sh` → `profiling/out_b/trace_main/` (816,900 dispatches, 165.3 s span,
`--kernel-trace` only). App-level numbers under trace: decode@ctx1k 27.6 tok/s, decode@ctx4k
27.0 (vs 29.3 unprofiled → CLI kernel-trace overhead ≈ 6 %); load 83 s.

- **Decode (pure window, warmup+64 tok run `iso_hiprt`): 86 % device-busy**, composition
  `gemv_sq` 74.5 % + `gemv_msq` 13.2 % (GEMV total 87.7 %), GDN 5.8 %, paged attn 2.5 %,
  elementwise 0.4 %, **hipBLAS 0 %**. Short-context decode is even more purely GEMV-bound than
  the earlier env-mode analysis suggested, and the earlier "hipBLAS in decode" rows in the mixed
  trace window came from the ctx1k job's *prefill*, not from decode (verified by correlation:
  the pure-decode window of a prefill-free run contains zero `Cijk` dispatches).
- **Grid census (Grid_Size_X is threads)**: `gemv_sq`/`gemv_msq` = 384 blocks × 256 threads
  (64–88 VGPR); `exl3_gemm_kernel` (coop) = **48 blocks × 512 threads on every launch** — the
  CUs/2 cooperative-grid defect (C1) re-confirmed through the CLI on the deployed image.
- **Batch-8 steady window: 88 % busy, `exl3_gemm_kernel` = 72 % of device busy** (23.8 s of
  33 s) — C2 re-confirmed. Dispatch rate falls 26.4 k/s (b1) → 4.4 k/s (b8): the msq m>1 path
  shares one launch across the batch, so dispatch count is not the batch bottleneck; the coop
  fallback is.
- **Prefill windows: 99 % busy, hipBLAS 89–92 %, reconstruct ≈ 4 %** — the device is already
  saturated within a prefill chunk; the gap to the 1743 tok/s roofline is (a) hipBLASLt reaching
  only ~28.7 TFLOP/s vs 83.5 measured in isolation on identical shapes, and (b) host time
  between chunks.
- **Load**: 32 s dispatch-free gap (disk/CPU) then fill bursts (~2.7 s `FillFunctor`); autotune
  after first job ≈ 13 s incl. two pure-coop windows (1.4 + 2.2 s). First-request cost, paid
  per process, not per request.

### 8.3 Flag isolation — the "--hip-runtime-trace stops the workload" claim is withdrawn

All of `--kernel-trace`, `--memory-copy-trace`, `--hip-runtime-trace` (alone and combined) and
`--hip-graph-trace` complete the full decode workload (21.6–22.3 tok/s under trace). The earlier
"workload did not proceed past load" observation was almost certainly the VRAM-containment
failure recorded in §5, not a trace-flag problem. HIP API traces are usable on this app, which
is what enables the host-gap attribution below.

### 8.4 Graph coverage (open item tg-3.3 closed)

`--hip-graph-trace` / API trace: **9,152 `hipGraphLaunch` calls in a 72-step run ≈ 130 launches
per decode step** — the generator's decode step already runs inside HIP graphs on this model
(matches the old ~138/token gfx1101 figure). There is no raw-launch problem to fix; graph
coverage survives on the msq/coop path as shipped.

### 8.5 Host gap, measured (C5 correction)

From the `--hip-runtime-trace` API trace, steady decode window:

- 44 % of the main-thread wall is inside HIP API calls; **56 % is outside any API call**
  (python/torch dispatch/sampler).
- Per step: 1 blocking `hipDeviceSynchronize` (structural — the next token depends on the
  sample), ~2,000 `hipGetDevice` calls (torch device-guard churn), ~146 event
  create/record/query/destroy cycles (generator timing churn), ~130 graph launches plus ~170
  raw kernel launches.
- In-trace idle is 14–19 % of decode wall; since runtime tracing adds ~2 µs to ~2,800 API
  calls/step, the unprofiled idle is materially smaller — **order 3–10 % (≈1–3 ms of the
  34.1 ms/step)**. The earlier "25–35 % GPU idle in decode" figure was an artifact of env-mode
  dispatch interception and is hereby corrected: **decode is ~90 %+ device-bound at batch 1.**

Implication for the plan: tg-3 (host gap) downgrades from a top-tier lever to hygiene (event
reuse, fewer device queries); the decode levers are the GEMV kernels themselves (tg-2) and the
batch path (tg-1).

### 8.6 In-process prefix reuse contaminates bench_rocm prefill numbers

`bench_rocm.py` builds its ctx1k/ctx4k/pp512/pp2048/pp4096 prompts as prefixes/extensions of one
another, and the generator reuses cached pages across jobs: the app reported prefill_ctx4k at
628 tok/s and prefill_2048 at 790 tok/s against a measured cold rate of 230–330 tok/s. Any
in-process prefill benchmark must nonce its prompts (HY-2), and cold vs prefix-hit must be
reported separately (which the HTTP probe in §9.3 does).

---

## 9. Session 2: attach SOLVED (two conditions), live-serve profile captured

### 9.1 `rocprofv3 --attach` itself is the broken piece

Reproducible against a quiescent, correctly-enabled target (bg-attach thread present, fds fixed,
ptrace verified working): `rocprofv3 --attach 1 --attach-children=false …` spins in R state
forever — the wrapper's `rocprof-attach` child never completes the handshake, and no ROCPROF_*
env reaches the target. The fd-limit fix below was necessary hygiene but not the cause
(the SIGTERM backtrace printed `Unable to get high fd … limit=1024` — docker exec runs at
nofile=1024; set `ulimits` in compose and/or `ulimit -n 65536` in the exec shell).

### 9.2 Working recipe (all steps required)

1. Target (serve) runs with `ROCP_TOOL_ATTACH=1` +
   `LD_PRELOAD=…/_rocm_sdk_devel/lib/librocprofiler-register.so` + `cap_add [SYS_PTRACE]` +
   `ulimits nofile 65536` (`profiling/attach_override.yml`). Verify with
   `/proc/1/task/*/comm` → `rocp-bg-attach` (state **S** at idle — the thread costs nothing
   until an attach attempt).
2. **The target must be quiescent when the attach lands.** Direct attach to the loaded-but-idle
   serve: success. The same attach while the serve is generating under HTTP load: hangs
   (rc=124 at timeout, both sides spinning in R state).
3. **Invoke `rocprof-attach` directly, not through `rocprofv3 --attach`**:
   `python3 -u …/_rocm_sdk_devel/bin/rocprof-attach -p 1 --attach-children=false \
      -t …/rocprofiler-sdk/librocprofiler-sdk-tool.so -d 60000`
   (`-u` matters: the client buffers stdout, which is why previous sessions saw "silence").
4. Tool configuration travels in the **client's environment** — `rocattach` serializes it into
   the target: `ROCPROF_KERNEL_TRACE=1 ROCPROF_OUTPUT_PATH=/tmp/prof ROCPROF_OUTPUT_FORMAT=csv`.
5. Drive HTTP traffic only after the client prints `Attaching for 60000 msec`.
6. Detach takes ~1 min and returns `:: success`; results land in the target's /tmp/prof
   (`docker cp` them out).

Script: `profiling/run_attach_live.sh`. Artifacts: `profiling/out_attach_live/`.

### 9.3 What the live-serve profile shows (TabbyAPI layer included)

`profiling/run_attach_live.sh` → `profiling/out_attach_live/` (220 MB kernel trace, attach
rc=0, clean detach, serve generating real HTTP traffic inside the 60 s window). Phase stats:

| window | wall | busy | composition |
|---|---|---|---|
| first request (warmup/JIT) | 0.8 + 1.6 + 3.3 s | 95/97/71 % | coop autotune sweeps, then first-token decode |
| served decode ×3 (256 tok each) | 33.9 s | 86 % | gemv_sq 60 % + gemv_msq 11 % + hipBLAS 9 % + coop 9 % + GDN 5 % (window mixes decodes with the first cold prefills) |
| cold 1.7 k prefills | ~10.9 s | 96 % | hipBLAS 82 % + reconstruct 5 % + gemv 5 % |
| warm-prefix repeats | ~0.1 s busy | — | almost no GEMM work — the prefix hit skips the layers |

Conclusions: the served path is kernel-identical to the sibling-container profiles (GEMV-bound
decode, hipBLAS-bound prefill); TabbyAPI adds nothing pathological at the kernel level; the
cold-TTFT cost is ~all device-busy GEMM, and the warm-prefix path nearly eliminates it. The
in-process traces remain the reference for single-request composition because the serve
overlaps requests in one window.

### 9.4 HTTP-level TTFT (clean, no profiler)

`profiling/ttft_probe.py` (streaming completions, greedy):

| case | TTFT | note |
|---|---|---|
| first request after restart (32 tok) | **7.58 s** | JIT/autotune one-time cost (pp-3) |
| short prompt, 256 gen, warm | **0.32 s** (6.0 → 0.76 → 0.32) | the warm floor; decode ≈ 28.1 tok/s served |
| cold ~1.7 k-token prompt | **3.43 s** | repeats identically (3.435/3.434) |
| same prompt again (warm prefix) | **0.60 s** | **prefix cache works at server level: 5.7×** |
| ~0.8 k-token cold prompt | 3.41 s | cold-prefill cost is round-count + host, not just tokens |
| two concurrent 192-tok decodes | both 4.06 s | aggregate 26.6 tok/s; requests queue behind one prefill round |

---

## 1. Blocker 1 — `rocprofv3 -- <app>` segfaults. SOLVED.

### The earlier explanation was wrong

The previous note claimed the preloaded SDK drags in `libLLVM`, whose symbols interpose into
triton's LLVM. **The tool libraries do not link libLLVM at all**:

```
$ objdump -p .../rocprofiler-sdk/librocprofiler-sdk-tool.so | grep NEEDED
  librocm_sysdeps_dw.so.1   libamd_comgr.so.3   librocm_sysdeps_elf.so.1
  librocm_sysdeps_sqlite3.so   librocprofiler-sdk.so.1   libprofiler-sdk.so.1  ...
$ nm -D --defined-only .../libamd_comgr.so.3 | grep -c llvm
0
```

There is no libLLVM in the picture and nothing to interpose *from* comgr.

### Measured root cause

`librocprofiler-sdk.so` **exports 143 `std::filesystem` / `std::__cxx11` symbols of its own**,
plus typeinfo/vtables:

```
$ nm -D --defined-only .../librocprofiler-sdk.so | grep -c filesystem
143
$ nm -D --defined-only .../libtriton.so | wc -l
135297            # triton statically links LLVM 23 + its own libstdc++ ABI
$ comm -12 <(triton syms) <(librocprofiler-sdk syms) | wc -l
210               # _ZNKSt10filesystem…, _ZTVSt…, _ZTISt…, _ZNSt10_Hashtable…
```

rocprofv3 LD_PRELOADs the tool → those symbols are in the **global** scope. CPython then
`dlopen`s `libtriton.so`; libtriton's own references to `std::filesystem` resolve to the **tool's**
implementations (global scope precedes the dlopened object's own scope). libtriton's LLVM was
built against its own libstdc++ ABI, so the substitution corrupts the heap → SIGSEGV in `cfree`
at import time. Direction proven by scope isolation:

| tool libs loaded as | result |
|---|---|
| `RTLD_GLOBAL` then libtriton | **SIGSEGV in cfree** |
| `RTLD_LOCAL` then libtriton | triton loads fine |

Scope of the crash (CLI, same invocation, no GPU device needed):

| target | without shim | with shim |
|---|---|---|
| `import numpy` | ok | ok |
| `import torch` | ok | ok |
| `import torch, triton` | **SIGSEGV** | ok |

So only `libtriton.so` collides; torch itself is unaffected.

### Fix

`profiling/shims/ld_scope_shim.c` — interposes `dlopen`/`dlmopen`, adds `RTLD_DEEPBIND` for the
libraries matching `LD_SCOPE_DEEPBIND` (default `libtriton`). The linker then searches the
library's own scope ahead of the global scope for its own relocations, so triton keeps its
bundled definitions.

```bash
gcc -shared -fPIC -O2 -o ld_scope_shim.so ld_scope_shim.c -ldl      # profiling/shims/build.sh
LD_PRELOAD=/shims/ld_scope_shim.so rocprofv3 --kernel-trace -f csv -d /out -- python3 app.py
```

### Verification (three levels, all reproducible)

1. CPU-only, no GPU device exposed, no profiler — ctypes with tool libs `RTLD_GLOBAL`: crashes
   without shim, loads with shim.
2. CPU-only, **real CLI**: `rocprofv3 -- python3 -c "import triton"` → SIGSEGV, rc=139 without the
   shim; with the shim it prints `TRITON-OK version 3.8.0`, `exit=0`, and the shim logs its
   deepbind line.
3. GPU, real CLI + real HIP work:
   `rocprofv3 --kernel-trace -- python3 -c "import torch, triton; x=torch.zeros(1024,device='cuda'); print((x+1).sum())"`
   → `1024.0` and `1_kernel_trace.csv` written.

The shim generalises to any C++ library that statically links its own libstdc++/LLVM and is
dlopened into a process whose global scope already holds a tool's symbols.

---

## 2. Blocker 2 — attach. Target side SOLVED, handshake OPEN.

### Target-side requirement (measured matrix)

The target must expose a `rocp-bg-attach` thread. It is created by
`librocprofiler-sdk-attach.so` **when that library is invoked by `librocprofiler-register.so`'s
constructor** — merely loading the attach library does nothing.

Target: `torch` + `triton`, GPUs initialised. Thread count from `/proc/1/task/*/comm`, mappings
from `/proc/1/maps`, both inspected **inside** the container.

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

Note: `librocprofiler-register.so` is mapped even with no env (the HSA runtime loads it), but the
**explicit `LD_PRELOAD` is what makes its constructor run early enough to enable the handshake**.
Both ingredients are required.

Working target recipe:

```bash
docker run ... --cap-add=SYS_PTRACE \
  -e ROCP_TOOL_ATTACH=1 \
  -e LD_PRELOAD=/opt/rocm-venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/librocprofiler-register.so \
  <image> <app>
```

### Attach side — OPEN

`rocprofv3 --attach 1 --attach-children=false --attach-duration-msec 5000 --kernel-trace` against
a target doing a continuous 2k matmul loop: the Python child `rocprof-attach` runs in **R state
(busy-spinning)** indefinitely, `/proc/1/environ` of the target never receives any `ROCPROF_*`
injection, and no output files appear (observed at t=15/30/45/60/75/90 s).

Eliminated / learned:

- The target **did** have the `rocp-bg-attach` thread in every one of these runs (so the
  enablement above is what was missing before, and it is now correct).
- The first attempt (default `--attach-children=true`) hung for 3.5 minutes; switching to
  `--attach-children=false` did **not** resolve the spin.
- Output was piped through `tail` in the first attempt, which hid everything; all later attempts
  redirect to a file. `rocprof-attach` block-buffers its stdout, so a file that shows only the
  C++ frontend's line does not by itself prove a hang — but the R-state spin and the absent env
  injection do.
- `rocprof-attach` is documented to spend up to 1–2 minutes in detach, so any attach test must be
  given ≥3 minutes before being called hung (the first attempt was cancelled inside that window —
  it may have been slower, not stuck; the later run was definitively spinning for 90 s with a 5 s
  duration and no attach-side output at all).

**Untested hypothesis (the last experiment was cancelled before it ran):** attach cannot handshake
while the target is mid-compute. The proposed test was: target initialises HIP → idles 25 s (attach
lands here) → compute burst 40 s (should be captured). If that works, the recipe for profiling the
serve becomes "attach while the serve is idle, then drive HTTP traffic", which is exactly what is
wanted. The script is reproducible via `profiling/run_attach_serve.sh` (with
`--attach-children=false`).

---

## 3. What the CLI unlocks — and a hard limit found with it

The register-loaded env workaround could not configure PMC counters, HIP/API traces, multi-pass
counters, `-L` counter discovery, attach, or non-CSV formats. The CLI can, and the first thing it
established is negative:

### DRAM-traffic counters are unavailable on this stack — proven through the CLI

`rocprofv3 -L` (full listing, 33 KB) shows the counters exist by name:

```
Counter_Name        :	FETCH_SIZE
Description         :	The total kilobytes fetched from the video memory...
Expression          :	(GL2C_EA_RDREQ_32B_sum*32+GL2C_EA_RDREQ_64B_sum*64+GL2C_EA_RDREQ_96B_sum*96+GL2C_EA_RDREQ_128B_sum*128)/1024
```

**`FETCH_SIZE` is a *derived* counter** — the frontend evaluates that expression. Three
`--pmc` passes through the official CLI, all on the real model workload:

| counter (via `--pmc`) | rows | nonzero |
|---|---|---|
| `FETCH_SIZE` | 12871 | **0** |
| `GRBM_GUI_ACTIVE` | 12871 | **0** |
| `GL2C_EA_RDREQ_128B_sum` | 15610 | **0** |
| `GL2C_HIT_sum` | 15610 | **0** |
| `SQ_WAVES` | 21080 / 12871 | all nonzero |
| `SQ_BUSY_CYCLES` | 15610 | all nonzero |
| `TCC_EA0_RDREQ_sum` / `TCC_EA0_RDREQ_32B_sum` | — | rejected (set fell back to `SQ_WAVES` only) |

Consequences that must be respected in any future analysis:

1. The earlier "GEMV runs at ~60 % of practical peak" figure is **inference from timing and a
   traffic model**, not a measurement. It cannot be confirmed on this SDK/gfx1100 combination
   without a different mechanism (a different counter set, a rocprofiler-sdk version that
   implements the GL2C/GRBM blocks for RDNA3, or a roofline probe kernel).
2. "FETCH_SIZE = 0" in the earlier workaround was **not** an artifact of using the env path — it
   reproduces through the official CLI. The earlier suspicion that the workaround caused it is
   answered: it did not.
3. `SQ_*` counters *do* work and are usable (occupancy and issue-side ratios).

### One more empirical finding: `--hip-runtime-trace` on this app

`rocprofv3 --kernel-trace --hip-runtime-trace --memory-copy-trace -- python3 bench_rocm.py`
produced a 40 MB HIP API trace, a 5.4 MB kernel trace and a memory-copy trace, but the kernel
trace covers **only the model-load phase** (span 92.6 s, 8.4 % busy, 25,642 dispatches, no
`exl3_gemv_int8_*` kernels at all; dominated by the load-time `FillFunctor` memsets and by
`Cijk_*` hipBLAS). The workload did not proceed to decode under that flag set. Hypothesis:
`--hip-runtime-trace` (or the volume of API tracing) breaks the app's graph-capture path. Not yet
isolated — correct procedure is to add flags one at a time (`--kernel-trace` alone is known good,
it produced a complete decode+prefill trace in the earlier env-mode run).

---

## 4. Data actually gathered this session

- Blocker 1: root cause, fix, three-level verification (§1).
- Blocker 2: target-side enablement matrix with thread/mapping counts (§2).
- Counter availability: `-L` listing for gfx1100 saved at `/tmp/avail.txt` (33 KB); derived
  `FETCH_SIZE` expression; per-counter nonzero counts through the CLI (§3).
- PMC run artifacts (SQ counters, kernel traces, model workload):
  `profiling/out_cli_pmc/<uuid>/{1_counter_collection.csv,1_kernel_trace.csv}`.
  Example (`SQ_BUSY_CYCLES`, one pass): `Cijk_*` 186 calls/2610 ms, `reconstruct_had_kernel` 260
  calls/110 ms, total 6067 ms across 702 Gcyc — note counter collection serialises dispatches
  (~3×), so these are ratios only.
- CLI kernel+HIP+copy trace of the model workload (load phase only, see §3):
  `profiling/out_cli_trace/<uuid>/`.
- Confirmation that the earlier committed analysis (`bcbc4e6`) is not invalidated on its
  *conclusions* (decode GEMV-bound, prefill reconstruct→hipBLAS, coop GEMM at 48 blocks on 96 CUs,
  batching flat beyond 4), but that its *bandwidth* claims are inference and must be labelled as
  such until a working DRAM counter exists.

---

## 5. GPU coordination — a real gap found

While my main profiling run was queued, `gpu-ctl status` reported:

```
== GPU coordination status ==
FREE
-- ground truth --
VRAM 9580/24560 MiB used (39%)
/dev/kfd holders: (none)
```

but the GPU was in fact occupied by another agent's work:

```
1765712  ./target/release/hipfire serve --model qwen3.5:4b --kv-mode q8 --idle-timeout 0
1765720  /src/target/release/daemon          (listed in /sys/class/kfd/kfd/proc/)
container: bcd8e194f9fd  scs-audit-v2  (containerd-shim pid 1257384 as ancestor)
```

Two distinct problems:

1. **Detection gap**: 9.5 GB of VRAM was held by a process the coordination tool does not see as a
   holder (`/dev/kfd holders: (none)` — its check misses this process; the KFD proc list and VRAM
   accounting disagree with it). Worth reporting to the gpu-coord developer.
2. **Park automation coverage**: `ornith-park-watch.sh`'s `SVCS` list covers
   `ornith15-mq4r-pm4` / `ornith-15-35b-mq4rp` only. The service actually running is
   `hipfire serve --model qwen3.5:4b`, which is **not** in that list, so a reservation would not
   park it. Consequently a full-VRAM job (this model needs ~18 GB of 24.5 GB) cannot start while
   it runs: my serve container went into a
   `RuntimeError: Insufficient VRAM in split for model and cache` restart loop when it started
   alongside it. That loop was stopped; it was contention, not a crash of the serve.

Implication: the remaining full-model profiling (CLI trace of the complete benchmark, and the
live-serve attach run) needs the GPU to itself. Either the other agent's `hipfire` serve is
stopped/parked, or the run waits.

---

## 6. State left behind (cleanup performed)

- All containers created during this session removed (`att2`, `attdiag`, `attachtgt`, `exl3-*`).
- `exllamav3-rocm-serve`: **stopped** (`docker compose stop serve`), original compose config
  (the temporary `ROCP_TOOL_ATTACH` / `LD_PRELOAD` / `SYS_PTRACE` overrides are not applied; the
  override file used was `/tmp/attach_override*.yml`, outside the repo). It cannot start while the
  other agent's 9.5 GB `hipfire` serve holds the GPU. To restore once the GPU is free:
  `docker compose -f ~/docker-containers/exllamav3-rocm/docker-compose.yml up -d serve`.
- GPU lock: released; the stale record from my cancelled run was cleared with `gpu-ctl clear-stale`
  (`gpu-ctl status` now reports FREE, with the 9.5 GB belonging to the other agent).
- No reservation of mine is left open.
- Scratch scripts under `/tmp` are disposable.

## 7. Reproduction commands

```bash
# Blocker 1: build the shim and use the CLI
profiling/shims/build.sh
docker run --rm --device /dev/kfd --device /dev/dri --group-add 44 --group-add 993 \
  --ipc host --shm-size 4g --ulimit nofile=65536:65536 \
  -e LD_PRELOAD=/shims/ld_scope_shim.so -v $PWD/profiling/shims:/shims:ro -v $OUT:/out \
  --entrypoint rocprofv3 tabbyapi-rocm:serve --kernel-trace -f csv -d /out -- python3 app.py

# Blocker 2 target side: verify the handshake thread exists
docker run -d ... -e ROCP_TOOL_ATTACH=1 -e LD_PRELOAD=<...>/librocprofiler-register.so ... <app>
docker exec <c> sh -c 'for t in /proc/1/task/*/comm; do cat $t; done | grep bg-attach'

# Blockers 2 (attach) once a quiescent-target test is wanted
docker exec <target> sh -c "timeout 300 rocprofv3 --attach 1 --attach-children=false \
  --attach-duration-msec 30000 --attach-sync-output --kernel-trace -f csv -d /out > /tmp/att.log 2>&1"

# Full-model runs (need the GPU exclusively; see §5)
profiling/run_cli_trace.sh          # CLI kernel trace of bench_rocm
profiling/run_cli_pmc.sh            # counter sets via --pmc
profiling/run_attach_serve.sh 90000 # attach-profile the live serve under HTTP load
```

## 8. Next steps, in order

1. Re-run the **quiescent-target attach test** (§2, untested hypothesis). If it passes, use
   attach+HTTP for the live-serve profile.
2. Isolate which trace flags the app tolerates: `--kernel-trace` alone, then add
   `--memory-copy-trace`, then `--hip-runtime-trace` (known to stop the workload after load).
3. Obtain a DRAM-traffic measurement by any means that works on this stack (different counter
   name set, `rocprofiler-sdk` upgrade, or a synthetic roofline kernel), so the decode
   "60 % of peak" claim can become measured rather than inferred.
4. Re-run the full CLI profile of the model (decode + prefill) and reconcile with the committed
   analysis in `docs/rocm-perf-gfx1100-tg-pp-plan.md`, correcting any number that was inferred.
5. Report the gpu-coord detection gap (§5) to the developer.

---

## 10. Session 3: plan execution — TTFT wins shipped, both kernel routes resolved

Date: 2026-09-16 evening. Branch: `rocm-perf` (off `rocm-port`). User scope decision: skip MTP,
focus on auto-regressive (decode/tg) speed.

### 10.1 Image provenance trap — know which image you are measuring

`exllamav3-rocm:serve` (the base) carries an extension .so built **Sep 14** — *before* the
`msq m>1 / sq m<=4` tuning landed (`28f29ee`, Sep 16 01:02). The deployed overlay
`tabbyapi-rocm:serve` carries the tuned extension (built Sep 16 11:48; its 705 MB layer
includes the exl3 pip install). Bench discrepancy that exposed it, fully reproducible:

| metric (bench_lean) | on stale base | on deployed overlay |
|---|---|---|
| decode b1 | 13.7 tok/s | **29.0** |
| batch4 aggregate | 13.1 | **31.7** |
| batch8 aggregate | 24.9 | **28.6** |
| prefill 2k | 324 | 342 (unchanged — reconstruct path identical) |

**Rule: any bench or gate run against `exllamav3-rocm:serve` is invalid for perf; build perf
images `FROM tabbyapi-rocm:serve` (see `profiling/Dockerfile.perf`).**

True deployed baselines (2026-09-16 18:2x, bench_lean): b1 29.01, batch4 31.67, batch8 28.62,
prefill-2k 342.4, load 83.5 s. Golden-token baseline for the deployed build:
`profiling/out_exec3/golden_tabbyapi.json` (8 prompts × 256 greedy tokens, includes a 4k-ctx
and two shared-prefix prompts).

### 10.2 pp-0 answered: the GEMM library is not the prefill bottleneck

`profiling/gemm_probe2.py` times both device paths on the model's exact (k, n) inventory
(51.2 GFLOP/token of prefill GEMMs, from checkpoint headers):

| m | torch.matmul | ext.hgemm_recon (what the model calls) |
|---|---|---|
| 512 | 72.5 TF/s | 91.6 TF/s |
| 1700 | 75.5 TF/s | 90.2 TF/s |
| 2048 | 74.5 TF/s | 93.1 TF/s |
| 4096 | 75.8 TF/s | 100.4 TF/s |

`hgemm_recon` in a clean loop runs at 3× the throughput the model achieves in-prefill
(~30 TF/s from the live-serve trace) on identical shapes. The gap is therefore call *context*:
the model allocates a fresh fp16 weight/output buffer per linear per chunk (`exl3.py:193/222`),
serializes reconstruct→GEMM per linear, and orchestrates ~450 calls from Python. When pp work
resumes, pp-1.2 (persistent scratch — makes the model match the probe's buffer-reuse condition)
and pp-1.3 (C++ driver) are the justified moves; pp-0 (algo/workspace hunt) is closed as
not-the-problem.

### 10.3 tg-1a: dead, hardware limit (documented at the patch site)

Doubling the cooperative grid on RDNA3 (`multiProcessorCount` = WGP count; 48 WGPs → 96 CUs)
asserts at launch: **"too many blocks in cooperative launch"** — the coop kernel (512 threads,
184+ VGPRs) cannot co-reside 96 blocks on 96 CUs. Not a reporting bug; the grid cannot be
widened without shrinking the block footprint. Reverted; the comment at `exl3_gemm.cu` keeps
the evidence. Batch decode must therefore be fixed with regular-launch kernels.

### 10.4 tg-1b: attempted, reverted — deadlock in a never-exercised kernel configuration

The msq kernel's work units already flatten (matrix × row), the had forms are prepared by the
caller, all model (k, n) pass its constraints, and `num_tokens` is unused in plain mode — so
routing single-matrix m>4 to `exl3_gemv_int8_msq` (patch `49e89f7`, image
`exllamav3-rocm:perf-tg1b`) looked safe. In the gate run the generator **deadlocked** on the
first 14-token prefill (the first-ever `bszm=1 × m>1` msq call): GPU idle, all 89 host threads
asleep. The kernel's validated coverage is m=1 with multiple matrices (`test_msq_ab.py`);
`bszm=1, m>1` was never exercised.

Follow-ups, in order:
1. Standalone kernel A/B for `bszm=1, m ∈ {5..16}` vs per-row sq (outside the generator), with
   `--cap-add=SYS_PTRACE` + py-spy/gdb in the container so a hang is debuggable.
2. Correctness-gate redesign for this change: **msq is not bit-exact vs sq/coop by design**
   (per-slice vs global activation scales — see the `test_msq_ab.py` header), so routing
   prefill m∈5..144 through it changes numerics; the token-exact golden gate must become
   "identical tokens on unchanged paths (m≤4, reconstruct) + KLD/tolerance on the m>4 prefill
   prompts" for exactly those runs.
3. Re-land with the G2 batch gate at m=8 (`correctness_gate.py batch` now runs 8 prompts).

### 10.5 Shipped: TTFT quick wins (no kernel change)

- `warmup.py` + compose entrypoint wrapper (short decode + 2k prefill at boot) and
  `sampling.override_preset: safe_defaults` in `config.yml` (backup: `config.yml.bak-*`).
- Verified over HTTP after redeploy: **first-request TTFT 7.58 s → 1.36 s (−82 %)**; decode
  256-token 27.9 tok/s, cold 1.7 k TTFT 3.42 s, warm prefix 0.58 s — all unchanged, as
  expected from a no-op-to-kernels change.

### 10.6 State at end of session

- Standing serve: running, warmed at boot, `tabbyapi-rocm:serve` (unchanged extension).
- Branch `rocm-perf`: tg-1a/tg-1b reverted with documentation; correctness/bench harness
  extended; golden baseline committed (`out_exec3/golden_tabbyapi.json`).
- Images: `exllamav3-rocm:perf-tg1a` / `perf-tg1b` exist as artifacts (do not promote);
  `Dockerfile.perf` is pinned to the correct base for the next attempt.
- Decode (tg) next lever remains tg-1b done right (§10.4 list), then tg-2 GEMV efficiency.

---

## 11. Session 4: tg-1b re-attempt — two latent bugs, an honest A/B, three refuted premises

Date: 2026-09-16 night → 17. Branch `rocm-perf`. Serve restored at session end (deployed
image, chunk 2048, re-verified golden-exact).

### 11.1 The original tg-1b "deadlock" was two stacked bugs

1. The patch (49e89f7) passed HOST stack arrays as the msq kernel's B/suh/svh pointer lists.
   The kernel dereferences the lists with device instructions (the mgemm path passes address
   tensors, `MultiLinear.ptrs_*`) — the first staged unit faults the queue. Fixed in the
   re-attempt with a device-side list buffer written by a 1-thread kernel whose launch
   arguments carry the pointers (capture-safe, no host/DMA race when calls run ahead on the
   stream — a pinned-staging variant was tried first and has exactly that race).
2. The msq workspace-growth loop terminated on `rows_per >= rows_max`, but `rows_per`
   saturates at `(rows_total + 7) & ~7` — below `rows_max` whenever `rows_total < rows_max`.
   lm_head-shaped calls (n = 248320) at m in (4, 144] then spin the host forever. Fixed:
   terminate when the growth step stops increasing `rows_per`. (Fixed on the branch; the
   deployed config cannot reach it — the single-matrix route was reverted — but the loop is
   live code in the msq launcher.)

### 11.2 Perf-image builds silently shipped the base .so

`COPY . /opt/exllamav3` carried the repo's stale `build/` tree → setuptools skipped the
rebuild → pip failed → `RUN pip install ... | tail -20` swallowed the exit → the image kept
the base's .so (timestamp Sep 16 11:48) while appearing rebuilt. Found by running `strings` on
the installed .so and grepping for a source marker: zero hits. Consequence: every perf-image
measurement earlier in the evening (including the first A/B rounds and the "coop reference")
was the base extension. `Dockerfile.perf` now: removes `build/` in-image, keeps pip's log and
fails on error, and verifies the installed .so contains a marker string from the patched
source. `.dockerignore` added (build/, .git, artifacts).

### 11.3 Cold-state kernel A/B (the numbers that decided tg-1b)

`profiling/msq_ab2.py`: rotates 2–4 same-shape layer instances so each call streams cold
weights (in-model re-reads happen after ~13 GB of other traffic; a same-tensor bench loop is
Infinity-Cache-resident and mis-ranks kernels — earlier warm numbers were 1.5–2× inflated).
Real trellis/scales from layers 3/7/11/15/19/23 (q_proj, k=5120, n=12288), K ∈ {3,4}.

| m | K=3 msq / coop (ms) | K=4 msq / coop (ms) | verdict |
|---|---|---|---|
| 1 | 0.112 / 0.115 | 0.060 / 0.065 | sq path (unchanged) |
| 5 | 0.570 / 0.575 | **0.279 / 0.477** | msq 1.7× at K=4; K=3 flat |
| 8 | 0.322 / 0.309 | 0.465 / 0.473 | **parity** (~530–610 GB/s both) |
| 16 | 0.694 / 0.472 | 0.932 / 0.500 | coop 1.5–1.9× faster |
| 32 | 1.446 / 0.938 | 1.836 / 0.954 | coop ~2× faster |
| 128 | 5.66 / 3.66 | 6.92 / 3.65 | coop ~1.9× faster |

Numerics: msq bit-exact vs per-row sq chunks for K=4 at m ∈ {5, 8} (maxrel 0.00e+00) and ≤
9.4e-3 elsewhere (workspace-growth changes the slice height at large m — expected).
Deterministic everywhere. Model-level KLD vs incumbent on the 5 short prompts: 1.1–4.1e-3
(with padded-vocab channels trimmed), zero greedy divergence in 8 steps — the per-slice-vs-
global scheme delta, the same one that already separates decode (sq) from prefill (coop).

The route was landed twice and reverted twice for cause:
- unbounded (m > 4): regresses m ≥ 16 prefill ~2×;
- bounded (4 < m ≤ 8) without the `!graph` guard: BC decode graphs abort at capture with the
  coop autotune sweeping inside the capture (eager run 0 takes msq, warms no coop key; the
  captured run 1 falls back to a cold coop key) — fixed by serving captured runs too, which
  then fails at replay with "Graph update failed": the route's staging kernel has pointer
  arguments the Graph param-update machinery does not record. Making that work means touching
  graph.cu's site kinds — for a measured batch-8 payoff of 1–2 % aggregate (below the ±3 %
  bench noise). Reverted per plan gate 9; the batch-8 150–250 target is refuted at the
  mechanism level (both kernels latency/traffic-bound at m = 8; the cache-sharing roofline
  requires B resident per call, which 24–32 MB/layer defeats).

### 11.4 Golden-token parity is binary-scoped

A rebuild whose only kernel-source change is dead code (the slice-major block, unreachable at
m=1) flips 2/8 golden prompts (short_general @ gen-token 20, unicode @ 122). The deployed
image re-verified EXACT against its own baseline in the same hold (control runs), so the drift
is codegen, not logic: editing a shared kernel header shifts inlining/register allocation for
the other kernels in the same translation units, fp16 rounding changes, greedy flips at knife-
edge decisions. Consequence for every future gate: token equality is only meaningful against
the same binary; across rebuilds use numcheck-KLD + a quality evaluation. (Also: returned
logits carry uninitialized channels ≥ actual_vocab_size — 243 NaNs at 248077–248319 in this
model; trim before any softmax/KLD math, both arms.)

### 11.5 Refuted premises (all measured, all documented in the plan doc §10)

- P4 chunk_size: 2048/4096/8192 TTFT-equivalent on cold 3.2k (3.41–3.48 s ± 1 %) — keep 2048.
- pp-1.2 persistent buffers: the P1 bisect shows allocation is invisible (all six variants
  54–60 TF/s, streaming-bound); the prefill gap is the reconstruct→GEMM serialization traffic.
- T3 batch-4 amortization: cold sq@4 = 4.1× sq@1 — extraction amortization does not reduce
  DRAM traffic; batch-4 ≈ batch-1 aggregate is traffic-true, not a generator bug.
- T1 batch-8 via msq route: parity at m=8, regression at m ≥ 16, reverted (11.3).

## 12. Session 5 (2026-09-17): pure-AR speedup — what the limiter actually is

Branch `rocm-perf`; images `exllamav3-rocm:perf-a/-b/-c` (perf-c = shipping candidate). Full
detail and per-item status in `docs/rocm-perf-gfx1100-tg-pp-plan.md` §11.

### 12.1 The dp4a-issue hypothesis is refuted, and the arithmetic behind it was wrong

The plan's central decode hypothesis was that the int8 GEMV is limited by V_DOT4 issue throughput:
"0.25 dp4a per weight x 26.5 G weights = 6.6 G dp4a/token, which at the achieved rate is ~25 ms of
the 34 ms token". Two measurements kill it:

- **ISA census** of the installed extension (`profiling/isa_census.py` + `iso_dump.sh` over 140
  gfx1100 code objects): every sq unit issues **16 v_dot4 per (16 k-rows x 32 columns) unit row**,
  i.e. **one dp4a per 32 weights** (dp4a/byte = 0.25/K) - not 0.25 per weight. That is **828 M
  warp-dp4a per token, 8x less than assumed**; the "25 ms" match was a coincidence of two wrong
  numbers.
- **A1 probe** (`profiling/dp4a_peak.hip`, `__builtin_amdgcn_sudot4` = the same instruction
  `__dp4a` lowers to): pure dp4a peaks at **240 G warp-dp4a/s** with one IMAD per dp4a (the GEMV's
  mix) at 93 G; a pure FMA loop reaches **490 G warp-inst/s**, which also **halves the plan's issue
  ceiling** (HIP reports 48 multiprocessors x 4 SIMD32 x 2.482 GHz = 476 G, and the FMA probe hits
  103% of that, so the 96-CU x 4 = 953 G figure in the plan double-counts).

The sq family therefore runs at **13.7% of the dp4a ceiling and ~35% of the issue ceiling** (K=4
wide unit, 525 GB/s x 84 insn/256 B = 172 G warp-inst/s) while streaming at 57% of the DRAM peak:
**no resource is saturated, the kernel is latency-bound**. That is why the plan's B2 (deeper row
pipeline for more MLP) was attempted - and it, too, failed: 2.5% *slower* cold (204.7/204.9 vs
210.0/209.6 GB/s) and it hung the generator in both end-to-end arms. The wide unit's prefetch depth
is not its limiter either.

### 12.2 Measurement hygiene: cold rotation needs a pool >> the Infinity Cache

`msq_ab2.py`'s rotation (6 same-shape instances) is **not** cold enough: 6 x 31-45 MB is only ~2-3x
the 96 MB Infinity Cache, and with a sequential pass over the pool a large fraction of each pass
re-reads what it just read. Measured on identical kernels: the same call takes 0.0917 ms/pass with a
267 MB pool and 0.234 ms with an 891 MB pool (8.9x IC) - a 2.5x difference. The protocol that
reproduces production is: pool >= 6x IC, **no warm-up pass** (the first pass *is* the cold
measurement), per-call event pairs, one configuration per process. Validation: that protocol's
m=4/m=1 ratio (3.6x) predicts the in-model batch-4 aggregate (+3.6% at b4) that `bench_lean` measures.

**Arm-order contamination is real and large.** With all `force_num_sms` values timed in one process,
`sms=0` and `sms=48` - which resolve to the same `num_sms=48` (`ext.g_get_num_sms(0)` = 48) -
differed by 2.2x on K=3, with tight min/max inside each arm. One configuration per process removes
it; every number quoted in §11 is one-config-per-process.

### 12.3 What shipped, and the honest decode ceiling

**Shipped: the ROCm slice-height default 48** (was 64) in both the sq and the msq int8 launchers:
+8.5% decode b1 (29.15 -> 31.63 tok/s), +3.1% b4, +1.3% b8, prefill unchanged, reproduced to 0.03%
across two independent arms. Jumping to **narrow instead of the smem-staged unit** for K=3/5 - which
is 9% faster in the cold kernel bench - is only +1.6% end-to-end and is therefore *not* shipped; the
per-arch routing stays, with `EXL3_SQ_STAGE_SMEM` left as an A/B knob.

**Decode concurrency is traffic-true.** b1 31.6, b2 34.9, b4 32.8, b8 32.5 tok/s aggregate: batching
buys ~10% and then goes flat, exactly as the cold-traffic measurements predict (m=4 costs 3.6x m=1).
The honest serving ceiling for this model on this GPU is **~31-35 tok/s**, and the b6 arm (6
concurrent) reproducibly *regresses* to 23 tok/s - a queue/scheduler artifact worth a follow-up, not a
property of the int8 path.

### 12.4 Prefill: named at last, and the fix is a library configuration

`profiling/prefill_profile.py` (torch.profiler over a cold nonced 2048-token prefill) shows **68.2% of
prefill device time in a single hipBLAS Tensile configuration** (`Cijk_Ailk_Bljk_HSS_BH_MT64x32x8_...`,
223 launches, 12.8 ms each = 28.5 TFLOP/s) with a second config at 11.2%, reconstruct at 2.5% and
everything else (norms, elementwise, copy, attention/GDN, host launches) under 12% combined. This
settles the three-way disagreement in the plan's fact 6: the 28.7 TFLOP/s in-model number is the
in-model truth, and the 83.5-100 TFLOP/s figures come from a *different library* - torch's backend
here is hipBLASLt, while the ext calls hipBLAS `cublasGemmEx` with `CUBLAS_GEMM_DEFAULT_TENSOR_OP`.
The prefill headroom is therefore configuration selection, ~2/3 of prefill.

---

## 13. Finding: the prefill "GEMM configuration" gap is the fp32-output path (session 6)

D1's 68.2% entry (`Cijk_Ailk_Bljk_HSS_...`, 12.8 ms/launch, 28.5 TFLOP/s) is an fp32 C/D GEMM: the
input projections (q/k/v, gate/up) are built with `out_dtype = torch.float`, the output projections
with `torch.half`. Measured through the extension's own entry point (m=2048, rotated buffers):
fp16 output 90.5-102.6 TFLOP/s, fp32 output 18.0-18.3 TFLOP/s - 5.0-5.6x, in the *same* call.

Corollary that corrects the earlier session's reading: hipBLASLt is not faster here. A standalone
hipcc probe with the incumbent's exact layouts gives 92.3 (fp16 D) and **19.0 (fp32 D)** TFLOP/s, so
the slow HSS kernel is slow in both libraries and the 3.7x of D2 was an output-dtype confound. The
`EXL3_HGEMM_LT` path was reverted.

Also recorded, because it cost a GPU fault: the incumbent call is
`cublasGemmEx(OP_N, OP_N, size_n, size_m, size_k, A=b lda=size_n, B=a ldb=size_k, C=c ldc=c_stride_m)`
- `ext.hgemm(a, b, c)` is the ordinary `a @ b`. Declaring the same operands with
`HIPBLASLT_ORDER_ROW` layouts makes the kernel read `b` at stride `k` instead of `n` and faults with
"Memory access fault by GPU node-1 ... Page not present".

Fix shipped: `EXL3_HGEMM_F16OUT` (default on) runs the fp32-output GEMM into an fp16 slab and widens
it. In-extension 4.8-5.0x per shape; end-to-end prefill 2k 324.7/345.3 -> 562.0/511.1 tok/s with
decode and every batch aggregate unchanged; worst model-level KLD 1.9e-3, zero greedy divergence in
8 steps, determinism bit-exact. Full numbers and gates: plan doc section 11.8.

## 14. Session 7 (2026-09-18): guided-goal continuation — 32.95 -> 34.6 tok/s, and a build-system landmine

Branch `guided-goal/qwen38-35bpw-50tps`, Qwen3.8-27B-EXL3-3.5bpw, RX 7900 XTX, harness
`profiling/guided_tg.py` (4096/256, b1 greedy). Session baseline: perf-final image = 32.95 tok/s
(reproduced 32.93/33.00/32.88 across holds), parity green.

### 14.1 The build-system landmine (read this before trusting any goal-session number)

`Dockerfile.goal` was authored against `Dockerfile.rocm10`'s documented `/opt/venv`, but the
perf-final lineage's venv is **`/opt/rocm-venv`** — and its `python3` resolves there too. The
first four goal images (goal-a16..a20) therefore **never installed anything**: the pip step died
with "/opt/venv/bin/pip: not found", masked by `| tail -5`, and every one of those images silently
benchmarked the stale perf-final binary + stale Python copy. Symptoms that gave it away: the
EXL3_SQ_STAGE_SMEM arms behaved like the *old* parser semantics (stage=2/3 both = "stage all"),
the EXL3_SQ_ROWS_PER_NARROWN and triton_paged file-mount arms were exact no-ops, and a baked-in
Python default flip (bc_attn) had no effect while the env var worked. Any docker image that
overrides a venv path MUST carry a staleness guard; Dockerfile.goal now runs
`python3 -c "import ...bc_attn as b; assert b.bc_attn_enable is False"` + a torch-first
`import exllamav3_ext` after install (the bare ext import fails on libc10 even when fine).

Consequences for the record: A16/A17's published conclusions ("K=4 staged/narrow units lose 5%")
were actually *stage-all-(K3/5/7)* arms on the old parser — K=4-staged was only ever measured for
real on goal-b1 (31.25, still a loss). The A18 rows-per-narrow sweep and the attention
splits/warps e2e arms were no-ops and were re-run for real on goal-b1/b2 (still no win: n96
33.18 = -4.5%, m16 34.18 = -1.6%). The gdn_ba vectorization "no e2e effect" reading was wrong —
on the correct binary it is worth -0.83 ms/token (14.1 below).

### 14.2 A19 (kept): the captured BC attention block replays slower than eager on gfx1100

`EXL3_BC_ATTN=0` on the stale binary: 34.07-34.20 across five runs vs 32.84-32.98 for BC-on
across five runs, all parity-green, both arm orders — clean +3.7%. The per-kernel trace shows
where it comes from: the eager dispatch path's `_paged_attn_decode_split_kernel` runs at 55 us
per call vs ~129 us inside the captured block (2.06 -> 0.89 ms/token for the 16 full-attention
layers); the combine pass is slightly bigger (+0.06) and the host gap shrinks 4.9 -> 4.0 ms.
Defaulted off for ROCm in bc_attn.py (`torch.version.hip` guard); EXL3_BC_ATTN=1 restores.

### 14.3 A21 (kept): the A14 `launch_bounds(256,4)` soft-keep is a regression on current codegen

goal-b1 (fresh install, all committed sources) measured 32.97 with `__launch_bounds__(256,4)`
vs 34.74 without (goal-b2) — a ~1.8 tok/s swing from the same source delta A14 had measured at
+0.32%. The forced minBlocksPerSM=4 caps the sq/msq kernels below their natural 80-VGPR
allocation ("natural register allocation measures faster", plan doc §11.2 — again). Reverted to
plain bounds; A14's soft-keep is hereby revoked.

### 14.4 A20 (kept): gdn_ba_gemv vectorized

The merged b/a projection GEMV ran one warp per output feature with 4 B half2 loads —
21.5 us/call x 48 GDN layers = 1.03 ms/token at ~30 GB/s effective, pure load latency. float4
loads (k%8 tail falls back to half2) cut it to ~4.2 us/call: 1.03 -> 0.20 ms/token in the final
trace. Parity green.

### 14.5 Shipped state and the remaining budget

goal-b2 defaults: **34.53 / 34.60 / 34.61 / 34.74 tok/s** across four independent runs, token
parity green throughout. Final trace (out_guided/b2-trace): wall 30.4 ms/token under profiling
overhead (28.9 unprofiled), device busy 26.0 ms, host gap 4.4 ms over ~921 launches/token.
Composition per token: sq GEMV 17.0 ms + msq GEMV 5.7 ms (22.7 ms = 87% of busy),
paged-attn split+combine 1.0 ms, GDN chain ~1.3 ms (ba now 0.20, recurrent 0.68, conv 0.21,
fused ops ~0.2), rms/gated norms 0.68 ms, elementwise+rope+misc ~0.3 ms.

Per-shape GEMV bandwidth (trace x model map, A18 analysis): lm_head K6 n=248320 runs at
~795 GB/s; GDN msq bundles ~645; K3 MLP ~566; gate/up/q (n>=12288) ~540; attn q/k/v msq
bundles ~446; down_proj/o_proj K4 (n=5120) ~340-370. The 22.7 ms GEMV block is the whole game
for 50 tok/s: it needs ~14 ms, i.e. ~800 GB/s average across shapes that have now resisted
unit type (wide/narrow/smem), slice height (24..128 global + narrow-only), grid multiplier,
prefetch depth, occupancy forcing, and BC-graph capture. The untried levers are all
structural: graph-capturing the whole decode step (gap 4.4 -> ~1 ms), fusing the residual-add +
rms_norm and GDN micro-chain (~1 ms device + ~300 launches), and a fundamentally different
weight-streaming scheme for the small-n shapes (n=5120 down/o family at half the bandwidth of
the same-size wide-n matrices, no configuration sensitivity found).

## 15. Session ISA (2026-09-23): DPP rotate is a keep; streaming loads are not

Image `exllamav3-rocm:goal-isa` (Dockerfile.goal, INSTALL_VERIFIED, `bc_attn` False).
Harness `profiling/guided_tg.py` 4096/256 b1 greedy vs `final-b2-a.json` tokens.
Two always-on / env-gated ISA changes in the K=4 wide unit, from the RDNA3 ISA
(Feb 2023 / 15-Aug-2023):

1. **Hot-loop shuffle → DPP16 `ROW_RR:1`** (`__builtin_amdgcn_update_dpp` dpp_ctrl
   0x121, ISA §7.7.1). Always on for ROCm. 256-thread GPU check: 0 mismatches vs
   `__shfl_sync`. ISA of the installed ext: 1 `v_mov_b32_dpp row_ror:1` in sq K=4
   M=1, 2 in msq K=4; the remaining `ds_bpermute` (83 / 134) are the xor reductions
   and Hadamard butterflies, not this permute.
2. **`EXL3_SQ_STREAM=1`**: `__builtin_nontemporal_load` of an `ext_vector_type(2)`
   (HIP `uint2` is a struct). sq encodings emit 3× `global_load_b64 … slc dlc`
   (ISA §4.1.1); msq uses saddr (`global_load_b64 v[..], vN, s[..]`) and LLVM
   drops the hint — msq_plain and msq_stream encodings are identical. Consume
   still waits `vmcnt(0)` (17 in sq, 28 in msq) because `r0=r1; r1=r2` reuse
   those VGPRs. Occupancy unchanged (max v[71] sq / v[56] msq, under the 96-VGPR
   16-wave/SIMD line).

| arm | median tok/s | runs | token_parity vs b2 |
|---|---|---|---|
| goal-b2 (`b2-def.json`) | 34.74 | 34.82 / 34.74 / 34.66 | (baseline) |
| final-b2-a | 34.61 | 34.66 / 34.61 / 34.59 | true |
| **goal-isa DPP-only** (`isa-dpp.json`) | **35.40** | 35.49 / 35.40 / 35.31 | **true** |
| goal-isa + `EXL3_SQ_STREAM=1` (`isa-stream.json`) | 35.36 | 35.48 / 35.36 / 35.32 | true |

DPP is +1.9% vs b2-def, +2.3% vs final-b2-a, greedy-identical. Stream is noise
on top of DPP and does not land on msq (5.7 ms of the 22.7 ms GEMV block).
**Keep DPP; leave `EXL3_SQ_STREAM` default off.** VOPD still does not cover
`v_dot4` / `v_mul_lo_u32` / `v_bfe`. Hash multiply (next) is a miss: see §16.

## 16. Session ISA (2026-09-23): 24-bit hash split is a miss

sq K=4 M=1 has 49 `v_mul_lo_u32` vs 48 `v_dot4`. ISA §V_MUL_LO_U32: "to multiply
integers with small magnitudes consider V_MUL_U32_U24, which is intended to be
a more efficient implementation." The trellis window is 16-bit, so

    w * 0x83DCD12D  ==  mul_u32_u24(w, 0xDCD12D) + (mul_u32_u24(w, 0x83) << 24)

is bit-identical (host + GPU, 0 mismatches on 0..65535). LLVM recombines the
C-level form (`__umul24` / `(hi<<24)+lo`) back into one `v_mul_lo_u32`; the
opcodes only appear with inline asm (`profiling/dp4a_peak.hip` MODE 6/8).

`dp4a_peak` on gfx1100, grid 384, 256 threads, 1024 iters × 20 reps, 2.482 GHz:

| mode | mix | ILP 8 G warp-dp4a/s (or mul/s) |
|---|---|---|
| 0 | pure dp4a | 129.4 |
| 1 | 32-bit `v_mul_lo_u32` + dp4a | 93.8 |
| 5 | 16-bit window `v_mul_lo_u32` + dp4a (GEMV-true) | **63.5** |
| 6 | 16-bit 2×`v_mul_u32_u24`+`v_lshl_add_u32` + dp4a | 43.4 |
| 7 | 16-bit `v_mul_lo_u32` only | 86.2 |
| 8 | 16-bit 24-split only | 67.2 |

Three full-rate ops lose to one `v_mul_lo_u32` at the GEMV's ILP. ILP 1 is a
tie (33.0 vs 32.3); ILP 4/8 (the consume_row shape) is −27%. **Leave the
codebook hash as `w *= 0x83DCD12Du`.** Do not add the 24-bit split.

The leftover `ds_bpermute` (xor tails) is a keep: see §18. Still open: CU vs
WGP (`-mcumode`), `buffer_load` vs `global_load`. Not VOPD / v_dot8 / WMMA.
Hadamard xor-16 still uses `ds_bpermute` (DPP ROW_XMASK only covers mask 1/2/4/8).

## 17. Session ISA (2026-09-23): NARROWN=128 is a miss

`EXL3_SQ_ROWS_PER_NARROWN=128` on `goal-isa` (DPP rotate already on), 4096/256 b1
greedy vs `isa-dpp.json` tokens.

| arm | median tok/s | runs | parity |
|---|---|---|---|
| DPP-only (`isa-dpp.json`) | **35.40** | 35.49 / 35.40 / 35.31 | (ref) |
| NARROWN=128 (`isa-narrow128.json`) | 33.58 | 33.66 / 33.58 / 33.55 | true |

−5.1% vs DPP. Taller slices on n<12288 raise per-unit occupancy cost more than they
cut slice-count overhead. **Leave NARROWN off.** Global `EXL3_SQ_ROWS_PER` stays 32.

The opposite (`NARROWN=16` = SQ_MINROWS) is not a legal A/B on this model: down_proj
k=17408 → rows_total=1088 → ksplit=68, and `SQ_KSPLIT_CAP` is 64, so the sq path
declines and the run dies (rc=1, no JSON, twice). Do not re-try 16 without raising
the cap. Floor that still fits the cap is rows_per=32 (ksplit=34).

## 18. Session ISA (2026-09-23): DPP ROW_XMASK on xor tails is a keep

Image `exllamav3-rocm:goal-isa-xmask` (Dockerfile.goal, INSTALL_VERIFIED). Same
harness vs `isa-dpp.json` tokens. `__shfl_xor_sync` in the int8 GEMV reductions
and max-reduces now goes through `exl3_row_xmask` (DPP16 `ROW_XMASK`, dpp_ctrl
0x160+mask). xor-16 stays on `ds_bpermute` (crosses the 16-lane group).

Probe (`profiling/dpp_xmask_probe.hip`): 0 mismatches on xor-1/2/4/8; standalone
xor-1 add 125.7 vs 30.7 G/s. First build died because the helpers sat below
`gemv_int8_row_sums`; moved them above the device building-blocks section.

ISA census vs goal-isa (K=4 M=1 residual-off):

| kernel | row_ror | row_xmask (1/2/4/8) | ds_bpermute was → now |
|---|---|---|---|
| sq | 1 | 42 (16/14/6/6) | 83 → 45 |
| msq | 2 | 52 (16/12/12/12) | 134 → 90 |

| arm | median tok/s | runs | parity vs dpp |
|---|---|---|---|
| DPP-only (`isa-dpp.json`) | 35.40 | 35.49 / 35.40 / 35.31 | (ref) |
| **xmask** (`isa-xmask.json`) | **35.67** | 35.78 / 35.67 / 35.62 | **true** |

+0.76% vs DPP, +2.7% vs b2 34.74. All three xmask runs sit above all three DPP
runs. **Keep.** Combined ISA permutes: rotate + xor-DPP.

NARROWN=16 on xmask died on SQ_KSPLIT_CAP (§17); do not retry.

## 19. Session ISA (2026-09-23): GRID_MULT=2 is a miss

Same xmask image, no rebuild. `EXL3_SQ_GRID_MULT=2` vs `isa-xmask.json` tokens.

| arm | median tok/s | runs | parity |
|---|---|---|---|
| xmask default (`isa-xmask.json`) | **35.67** | 35.78 / 35.67 / 35.62 | (ref) |
| GRID_MULT=2 (`isa-g2.json`) | 35.12 | 35.27 / 35.12 / 35.12 | true |

-1.54%. All three G2 runs sit below all three xmask runs. Extra blocks queue:
occupancy is already one 256-thread block per WGP (ROCm `num_sms`=48). Leave
`EXL3_SQ_GRID_MULT=1`. Do not retry 4/8.

Tiling knobs on this image are closed (NARROWN 128 miss, 16 illegal, GRID_MULT
miss). Next: `-mcumode` rebuild (`EXL3_CUMODE=1`) so each CU hosts its own
workgroup (48 WGPs → 96 CUs).

## 20. Session ISA (2026-09-23): VOPD / packed / WMMA closed by the ISA; CU-mode + buffer_load open

Read `rdna3-shader-instruction-set-architecture-feb-2023_0.md` against the K=4
wide-unit inner loop (16 `v_mul_lo_u32` hash + 16 `v_dot4` + DPP rotate):

- **VOPD (ISA §7.6 / §16.11)** is wave32 dual-issue, but X opcodes are F32
  (`FMAC`/`MUL`/`ADD`/`DOT2ACC_F16`) and Y adds only `ADD_NC_U32` / `AND_B32` /
  `LSHLREV_B32`. There is no dual `v_mul_lo_u32` or `v_dot4`. Do not emit VOPD
  in this kernel.
- **v_dot8** (`V_DOT8_I32_IU4`) is 8×4-bit. The codebook is 8-bit after the
  0x83DCD12D hash. Wrong datatype.
- **WMMA I32 16×16×16 IU8 (ISA §7.9)** needs 16 rows of A replicated across
  lanes 0–15/16–31. Decode GEMV is M=1. Wrong shape.
- **Packed 16-bit (ISA §7.5)** is F16/I16 pairs. Inner loop is I32 hash + I8
  dp4a. No fit.
- **CU vs WGP (ISA §12.1.2)**: `-mcumode` / `.workgroup_processor_mode=0x00`.
  Image `goal-isa-cumode` rebuilding. Measure with `EXL3_SQ_GRID_MULT=2`
  because ROCm still reports 48 SMs; MULT=1 would fill 48 of 96 CUs. Token
  parity vs `isa-xmask.json`.
- **buffer_load (ISA §9)**: texture-cache L0 vs `global_load` vector L0.
  Wired as `EXL3_SQ_BUFFER_LOAD=1` (`load_k==2`) for the next A/B. Isolated
  probe `profiling/buffer_load_probe.hip` must be 0-mismatch before e2e.
- **DS_SWIZZLE SWAPX16 (ISA §16.15, offset 0x401f)** for remaining xor-16.
  Not default-on until the same probe is 0-mismatch.


Probe (`profiling/buffer_load_probe.hip`) on gfx1100:

| check | result |
|---|---|
| BUFFER_LOAD_B64 vs global, flags `0x00020000` | 16384/16384 mismatch (format INVALID / dst_sel=0 → zeros; fake 5.5 TB/s) |
| BUFFER_LOAD_B64 vs global, flags `0x31004000` (CK gfx11 V# word3) | **0 mismatches** |
| DS_SWIZZLE SWAPX16 `0x401f` vs shfl_xor-16 | **0 mismatches** |
| synthetic GB/s global vs buffer (`0x31004000`) | 2468 vs **4747** (+92%) |

Re-run 14:04 UTC, probe rc=0. xor-16 swizzle is **not** default-on in the CU-mode
image (kept `ds_bpermute` so that A/B is `-mcumode` only). `EXL3_SQ_BUFFER_LOAD=1`
is the next one-variable e2e A/B after CU-mode, token parity vs xmask.

CU-mode Docker false fail: pip's pyproject backend swallows setup.py stdout, so
grepping pip.log for `-mcumode` killed a successful wheel. Stamp is now
`/tmp/exl3_hip_cflags.txt`; ELF `.workgroup_processor_mode=0x00` is the gate.

## 21. 50 tok/s budget (honest, 2026-09-23)

Incumbent: `isa-had-r40.json` 4096/256 b1 greedy **41.30 tok/s** (41.39 / 41.30 / 41.19),
parity green vs swizzle-r40 40.79. Goal is 50. That is **+21%**, ~24.21 → 20.0 ms/token.

Decode-only recapture at this keep (`had-r40-dec`, prefill drained, 63 tokens):
unprofiled ref **41.47 tok/s** = 24.11 ms/token. Device 1369 ms → **21.73
ms/token**. Host idle **~2.38 ms**. GEMV **18.30 ms/token** (84% of device,
76% of wall). Kernel averages are unchanged vs swizzle-r40-dec (18.29 ms);
the +1.2% e2e is occupancy (`maxb` 3→4), not a shorter inner loop.

Arithmetic to 50 from 41.30 (24.21 ms/token on 4096/256):

| lever | max save if it went to zero | tok/s if only this |
|---|---|---|
| host idle 2.38 ms | 2.38 ms | ~45.8 |
| attn+GDN+norm+act ~3.4 ms | 3.4 ms | ~48.0 |
| remaining to 20.0 ms | 4.21 ms | **50** |

Zeroing host does not reach 50. Zeroing non-GEMV device does not reach 50.
**GEMV has to get ~24% faster** (18.30 → ~14.1 ms) if host and the rest stay.
CU-mode occupancy wall: 256-thread blocks, 8 waves, 2 SIMD32 × 16 wave slots
= **4 blocks/CU**. `maxb=4` is that ceiling. Cannot raise occupancy without
shrinking the block.

Closed ISA: VOPD, v_dot8, WMMA, packed 16, 24-bit hash, STREAM/nontemporal,
NARROWN=128 (WGP) and NARROWN=64 (CU, −1.5%), GRID_MULT=2 on WGP, launch_bounds
minBlocks, BUFFER_LOAD (−2.6%), CU MULT=3 at maxb=3 (−3.7%) and at maxb=4
(−5.9%, grid=576 = 6/CU). Occupancy-shape 32/48/56/64. Closed keeps: CU
MULT=2, xor-16 `DS_SWIZZLE`, rows_per=40, Hadamard `had_xmask` (+1.2% → 41.30,
occupancy maxb 3→4). rows_per=36 rounds to 40; 41.18 ≈ 41.30. Closed miss: K=3
independent BFE (−0.17%). Open: replace inner-loop v_mul_lo (LDS 2×256 LUT probe). Do not call
50 done until a 4096/256 greedy median is measured ≥50 with token parity.

## 22. Session ISA (2026-09-23): CU-mode image verified; 4096/256 A/B in flight

`exllamav3-rocm:goal-isa-cumode` (Dockerfile.goal `EXL3_CUMODE=1`):

- hip cflags stamp: `-O3 -ffast-math -DHIPBLAS_USE_HIP_HALF -mcumode`
- `INSTALL_VERIFIED`
- ELF `.workgroup_processor_mode`: `{'0x0': 1472}` verdict **CU** (`ELF_MODE_VERIFIED`)
- xor-16 left on `ds_bpermute` so this A/B is `-mcumode` only

xmask K=4 M=1 residual-off ISA (`profiling/out_guided/isa-xmask-dump`):

| opcode | count |
|---|---|
| `v_dot4_i32_iu8` | 48 |
| `v_mul_lo_u32` | 49 |
| `v_mov_b32_dpp` | 43 |
| `ds_bpermute_b32` | 45 |
| `global_load_b64` | 7 |
| `buffer_load_b64` | 0 |
| `ds_swizzle_b32` | 0 |

One consume_row body is ~190 insns: 16 mul + 16 dot + 3 `global_load_b64` + 2 `ds_load_b128`.
VALU-bound. `BUFFER_LOAD` synthetic +92% GB/s is not the inner loop.

4096/256 vs `isa-xmask.json` (35.67), token parity true both arms:

| arm | median tok/s | runs | parity | launch (sq k=5120 n=17408) |
|---|---|---|---|---|
| xmask WGP (`isa-xmask.json`) | **35.67** | 35.78 / 35.67 / 35.62 | (ref) | (prior) |
| CU MULT=1 (`isa-cumode.json`) | 33.88 | 33.98 / 33.88 / 33.84 | true | grid=144 maxb=3 sms=48 |
| **CU MULT=2** (`isa-cumode-g2.json`) | **38.36** | 38.46 / 38.36 / 38.30 | **true** | grid=288 maxb=3 sms=48 |

**Keep `-mcumode` + MULT=2.** MULT=1 underfills (sms stays 48). MULT=2 on WGP was a
miss; the keep is the pair. New incumbent **38.36**, still not 50 (+30% remaining).
Default MULT=2 under `-DEXL3_CUMODE` (swizzle rebuild). BUFFER_LOAD closed: 37.37
vs 38.36 (−2.6%).

## 23. Session ISA (2026-09-23): BUFFER_LOAD is a miss (−2.6%)

`EXL3_SQ_BUFFER_LOAD=1 EXL3_SQ_GRID_MULT=2` on `goal-isa-cumode` vs
`isa-cumode-g2.json` (38.36). First attempt: docker/python **SIGSEGV 139**,
empty log (stdout not flushed; container-name clash). Retry with
`PYTHONUNBUFFERED=1` completed.

| arm | median tok/s | runs | parity |
|---|---|---|---|
| CU MULT=2 (`isa-cumode-g2.json`) | **38.36** | 38.46 / 38.36 / 38.30 | (ref) |
| BUFFER_LOAD (`isa-buffer-g2.json`) | 37.37 | 37.52 / 37.37 / 37.20 | **true** |

−2.6%. All three buffer runs sit below all three g2 runs. Isolated probe was
0-mismatch / +92% synthetic GB/s; the inner loop is VALU-bound (`v_mul_lo` +
`v_dot4`), so texture L0 does not help. **Leave `EXL3_SQ_BUFFER_LOAD=0`.**
Do not retry.

xor-16 `DS_SWIZZLE` image compiling.

## 24. Session ISA (2026-09-23): CU MULT=3 is a miss (−3.7%)

`EXL3_SQ_GRID_MULT=3` on `goal-isa-cumode` vs `isa-cumode-g2.json` (38.36).
Env only, same image, token parity true.

| arm | median tok/s | runs | parity | sq k=5120 n=17408 |
|---|---|---|---|---|
| **CU MULT=2** (`isa-cumode-g2.json`) | **38.36** | 38.46 / 38.36 / 38.30 | (ref) | grid=288 maxb=3 sms=48 |
| CU MULT=3 (`isa-cumode-g3.json`) | 36.93 | 37.21 / 36.93 / 36.91 | **true** | grid=432 maxb=3 sms=48 |

−3.7%. 432 blocks = 4.5 per CU; occupancy is maxb=3 (3 per CU = 288). Same
oversubscribe pattern as WGP MULT=2. **Keep MULT=2.** Do not raise the default.

Incumbent was 38.36 until rows_per=48.

## 25. Session ISA (2026-09-23): rows_per=48 is a keep (+4.5% → 40.08)

`EXL3_SQ_ROWS_PER=48 EXL3_SQ_GRID_MULT=2` on `goal-isa-cumode` vs
`isa-cumode-g2.json` (38.36). Env only, same image, token parity true.

| arm | median tok/s | runs | parity | sq k=5120 n=17408 |
|---|---|---|---|---|
| CU MULT=2 rows=32 (`isa-cumode-g2.json`) | 38.36 | 38.46 / 38.36 / 38.30 | (ref) | grid=288 ksplit=10 |
| **CU MULT=2 rows=48** (`isa-cumode-r48.json`) | **40.08** | 40.12 / 40.08 / 39.99 | **true** | grid=288 ksplit=7 |

+4.5%. Fewer K-splits, same 288-block grid. Pre-CU this model preferred 32
(32.69 vs 48 at 32.00); CU-mode occupancy flipped it. **Default 48 under
`-DEXL3_CUMODE`.** New incumbent **40.08**, still not 50 (−20%, 24.95 → 20.0
ms/token). Next was xor-16.

## 26. Session ISA (2026-09-23): xor-16 DS_SWIZZLE is a keep (+1.3% → 40.59)

`goal-isa-swizzle` (`ds_swizzle` SWAPX16 0x401f) vs `goal-isa-cumode` (`ds_bpermute`)
with `EXL3_SQ_GRID_MULT=2 EXL3_SQ_ROWS_PER=48`. Token parity true.

| arm | median tok/s | runs | parity |
|---|---|---|---|
| CU MULT=2 rows=48 (`isa-cumode-r48.json`) | 40.08 | 40.12 / 40.08 / 39.99 | (ref) |
| **+ xor-16 SWAPX16** (`isa-swizzle-r48.json`) | **40.59** | 40.64 / 40.59 / 40.53 | **true** |

+1.3%. No run overlap (min swizzle 40.53 > max r48 40.12). Keep `ds_swizzle` for
xor-16. New incumbent **40.59**, still not 50 (−19%, 24.64 → 20.0 ms/token).
Next was rows_per=64.

## 27. Session ISA (2026-09-23): rows_per=64 is a miss (−3.3%)

`EXL3_SQ_ROWS_PER=64` on `goal-isa-swizzle` vs `isa-swizzle-r48.json` (40.59).
Env only, same image, token parity true.

| arm | median tok/s | runs | parity | sq k=5120 n=17408 |
|---|---|---|---|---|
| **rows=48** (`isa-swizzle-r48.json`) | **40.59** | 40.64 / 40.59 / 40.53 | (ref) | grid=288 ksplit=7 |
| rows=64 (`isa-swizzle-r64.json`) | 39.26 | 39.34 / 39.26 / 39.13 | **true** | grid=288 ksplit=5 |

−3.3%. Taller slices cut ksplit 7→5 but lose the occupancy that made 48 beat 32.
**Keep 48.** Next was 56.

## 28. Session ISA (2026-09-23): rows_per=56 is a miss (−2.3%)

`EXL3_SQ_ROWS_PER=56` on `goal-isa-swizzle` vs `isa-swizzle-r48.json` (40.59).
Env only, same image, token parity true.

| arm | median tok/s | runs | parity | sq k=5120 n=17408 |
|---|---|---|---|---|
| **rows=48** (`isa-swizzle-r48.json`) | **40.59** | 40.64 / 40.59 / 40.53 | (ref) | grid=288 ksplit=7 |
| rows=56 (`isa-swizzle-r56.json`) | 39.64 | 39.77 / 39.64 / 39.58 | **true** | grid=288 ksplit=6 |
| rows=64 (`isa-swizzle-r64.json`) | 39.26 | 39.34 / 39.26 / 39.13 | true | grid=288 ksplit=5 |

−2.3% at 56, −3.3% at 64. Taller than 48 loses occupancy. Next was 40.

## 29. Session ISA (2026-09-23): rows_per=40 is a keep (+0.5% → 40.79)

`EXL3_SQ_ROWS_PER=40` on `goal-isa-swizzle` vs `isa-swizzle-r48.json` (40.59).
Env only, same image, token parity true.

| arm | median tok/s | runs | parity | sq k=5120 n=17408 |
|---|---|---|---|---|
| rows=48 (`isa-swizzle-r48.json`) | 40.59 | 40.64 / 40.59 / 40.53 | (ref) | grid=288 ksplit=7 |
| **rows=40** (`isa-swizzle-r40.json`) | **40.79** | 40.87 / 40.79 / 40.74 | **true** | grid=288 ksplit=8 |
| rows=56 | 39.64 | 39.77 / 39.64 / 39.58 | true | ksplit=6 |
| rows=64 | 39.26 | 39.34 / 39.26 / 39.13 | true | ksplit=5 |

+0.5%. No run overlap (min 40 40.74 > max 48 40.64). ksplit 8 vs 7 is the
occupancy sweet spot under CU+MULT=2. **Default 40 under `-DEXL3_CUMODE`.**
New incumbent **40.79**, still not 50 (−18%, 24.52 → 20.0 ms/token). Next was
CU NARROWN=64.

## 30. Session ISA (2026-09-23): CU NARROWN=64 is a miss (−1.5%)

`EXL3_SQ_ROWS_PER=40 EXL3_SQ_ROWS_PER_NARROWN=64` on `goal-isa-swizzle` vs
`isa-swizzle-r40.json` (40.79). Env only, same image, token parity true.

Launch log confirms the split: wide-n (`k=5120 n=17408`) stayed `rows_per=40`
ksplit=8; narrow-n (`n=2048/5120`) went to `rows_per=64` ksplit=5/6/17.

| arm | median tok/s | runs | parity |
|---|---|---|---|
| **rows=40 all** (`isa-swizzle-r40.json`) | **40.79** | 40.87 / 40.79 / 40.74 | (ref) |
| NARROWN=64 (`isa-swizzle-r40-nn64.json`) | 40.16 | 40.24 / 40.16 / 40.10 | **true** |

−1.5%. All three NARROWN runs sit below all three r40 runs. Taller slices lose
occupancy even on down_proj / o_proj. **Leave `EXL3_SQ_ROWS_PER_NARROWN=0`.**
Do not retry 128 or 56 here — global 56/64 already lost. Next was rows_per=32.

## 31. Session ISA (2026-09-23): rows_per=32 on xor-16 is a miss (−4.5%)

`EXL3_SQ_ROWS_PER=32` on `goal-isa-swizzle` vs `isa-swizzle-r40.json` (40.79).
Env only, same image, token parity true.

| arm | median tok/s | runs | parity | sq k=5120 n=17408 |
|---|---|---|---|---|
| **rows=40** (`isa-swizzle-r40.json`) | **40.79** | 40.87 / 40.79 / 40.74 | (ref) | grid=288 ksplit=8 |
| rows=48 | 40.59 | 40.64 / 40.59 / 40.53 | true | ksplit=7 |
| rows=32 (`isa-swizzle-r32.json`) | 38.96 | 39.05 / 38.96 / 38.90 | **true** | grid=288 ksplit=10 |
| rows=56 | 39.64 | 39.77 / 39.64 / 39.58 | true | ksplit=6 |
| rows=64 | 39.26 | 39.34 / 39.26 / 39.13 | true | ksplit=5 |

−4.5%. Unimodal: 32 < 40 > 48 > 56 > 64. CU-mode occupancy peak is 40
(ksplit=8). Pre-CU this model preferred 32 (32.69 vs 48 at 32.00); xor-16
does not restore that. **Keep 40.** Occupancy-shape A/Bs are closed.
Incumbent remains **40.79**, not 50 (−18%, 24.52 → 20.0 ms/token). Next was
CTX=4096 decode-mix recapture.

## 32. Session ISA (2026-09-23): CTX=4096 mix is 79% GEMV; host is not the 50-gap

`decode_profile.py` on `goal-isa-swizzle` (MULT=2, rows=40). Unprofiled
reference **41.11 tok/s** over 64 tokens (24.32 ms/token), consistent with
the 4096/256 median 40.79. Profiled wall 6.182 s includes a full 4096-token
**prefill** (first iterate mix). Recapture with prefill drained is in flight.

Prefill-contaminated device 4992 ms / 6.18 s wall → 81% busy is **not** the
decode host gap. Session-2 HIP-API (~1–3 ms idle) still stands for decode.

Decode-relevant kernels in that capture (64 tokens; hipBLAS / reconstruct /
exl3_gemm / paged_attn_prefill / GDN-chunk treated as prefill):

| family | ms | ms/token | share of 24.32 ms |
|---|---|---|---|
| `exl3_gemv_int8` sq+msq | 1227 | **19.17** | **79%** |
| rms_norm / gated_rms | 72 | 1.13 | 4.6% |
| paged-attn decode | 67 | 1.05 | 4.3% |
| GDN recurrent + conv1d + ba_gemv | ~99 | 1.55 | 6.4% |
| act_mul | 38 | 0.60 | 2.5% |
| rope / kv_update / fused_op | ~18 | 0.28 | 1.1% |

Zeroing attn+GDN+norm+act (~4.3 ms) without touching GEMV tops out ~49 tok/s
and is not realistic. **50 still needs GEMV ~24% faster** (19.17 → ~14.7 ms).

Hot GEMV is `sq_kernel<3>` (330 ms, 56 µs) and `sq_kernel<4>` residual-on
(199 ms, 38 µs). Confirmed `isa-swizzle-r40-dump` opcode census (whole kernel):

| kernel | insns | `v_dot4` | `v_mul_lo` | DPP | `ds_bpermute` | `ds_swizzle` | `global_load_b64` |
|---|---|---|---|---|---|---|---|
| sq K=3 M=1 r-off | 1557 | 32 | 33 | 40 | **40** | 6 | 4 |
| sq K=4 M=1 r-on | 1705 | 48 | 49 | 43 | **40** | 6 | 7 |
| msq K=4 r-on | 2706 | 32 | 34 | 54 | 80 | 12 | 11 |

xmask dump had 45 `ds_bpermute` + 0 swizzle. xor-16 converted ~5–6 of those to
`ds_swizzle` (slice-max / row-sum xor-16). The remaining **40 `ds_bpermute`**
match input+output Hadamard: `had_hf_r_128_inner` + `had_fh_r_128_inner`, each
`shuffle_had_f4x32` = 5 xor steps × 4 floats = 20 shuffles. Those still lower
to `ds_bpermute` because `had_xmask` is not in this image. VALU-bound inner
loop is 48 `v_dot4` + 49 `v_mul_lo`; BUFFER_LOAD / STREAM / 24-bit hash / VOPD
/ WMMA already closed.

Hadamard DPP/SWAPX16 (`had_xmask`) is a **decode GEMV** lever (sq stage_slice
+ epilogue), not only prefill `reconstruct_had`. Image `goal-isa-had` built
(`had_xmask` in `/opt/exllamav3/.../hadamard_inner.cuh`). A/B vs 40.79 in
flight. Do not expect it to close 50 by itself (40 shuffles vs 48 dots).

## 33. Session ISA (2026-09-23): decode-only mix — GEMV 18.3 ms, host ~3.1 ms

Prefill-drained recapture (`swizzle-r40-dec`). Drain iters=3 (first streaming
token excluded). Unprofiled reference **40.35 tok/s** / 64 tokens (24.78
ms/token). Profiled wall 2.496 s / 1366 ms device = 55% busy is **profiler
overhead**; do not treat 45% as the decode host gap.

63 profiled tokens (64 − first streaming):

| family | ms | ms/token | share of 24.78 ms |
|---|---|---|---|
| `exl3_gemv_int8` sq+msq | 1152 | **18.29** | **74%** |
| paged-attn decode split+combine | 63 | 1.00 | 4.0% |
| GDN recurrent + conv1d + ba_gemv + fused | 81 | 1.28 | 5.2% |
| rms_norm + gated_rms | 43 | 0.68 | 2.7% |
| act_mul + sigmoid | 7 | 0.10 | 0.4% |
| rope + kv_update | 9 | 0.14 | 0.6% |
| other device (add/elem/copy) | 11 | 0.18 | 0.7% |
| host idle (unprofiled − device) | — | **~3.1** | **12%** |

Hot GEMV (avg): `sq<3>` 55 µs × 5796, `msq<4 r-on>` 66 µs × 3023, `sq<4 r-on>`
38 µs × 5166, `sq<3 r-on>` 58 µs, `sq<4 r-off>` 60 µs, `sq<6>` 1175 µs × 63.
No hipBLAS / reconstruct / exl3_gemm in this window.

**50 still needs GEMV ~26% faster.** Host-to-zero tops out ~46.7 tok/s.

`goal-isa-had` ISA census (same kernels, `had_xmask` compiled in):

| kernel | insns | DPP | `ds_bpermute` | `ds_swizzle` | `s_waitcnt` |
|---|---|---|---|---|---|
| sq K=3 r-off swizzle | 1557 | 40 | 40 | 6 | 76 |
| sq K=3 r-off **had** | **1250** | **72** | **0** | **14** | **47** |
| sq K=4 r-on swizzle | 1705 | 43 | 40 | 6 | 78 |
| sq K=4 r-on **had** | **1406** | **75** | **0** | **14** | **48** |

Inner loop unchanged (48 `v_dot4` + 49 `v_mul_lo`). Hadamard shuffles moved
to VALU (`v_mov_b32_dpp` + `ds_swizzle`); LDS `s_waitcnt` dropped ~40%. e2e
completed below.

## 34. Session ISA (2026-09-23): Hadamard DPP/SWAPX16 is a keep (+1.2% → 41.30)

`goal-isa-had` (`had_xmask` DPP ROW_XMASK 1/2/4/8 + DS_SWIZZLE SWAPX16) vs
`goal-isa-swizzle` (Hadamard still `shfl_xor` → `ds_bpermute`). Same env:
`EXL3_SQ_GRID_MULT=2 EXL3_SQ_ROWS_PER=40`. Token parity true.

| arm | median tok/s | runs | parity | sq k=5120 n=17408 |
|---|---|---|---|---|
| xor-16 + rows=40 (`isa-swizzle-r40.json`) | 40.79 | 40.87 / 40.79 / 40.74 | (ref) | grid=288 maxb=3 |
| **+ had_xmask** (`isa-had-r40.json`) | **41.30** | 41.39 / 41.30 / 41.19 | **true** | grid=384 maxb=4 |

+1.2%. No run overlap (min had 41.19 > max swizzle 40.87). Occupancy API
`maxb` 3→4 (LDS `s_waitcnt` 78→48, `ds_bpermute` 40→0). Grid 288→384 is the
same MULT=2 × sms=48 × maxb. Inner loop still 48 `v_dot4` + 49 `v_mul_lo`.

**Keep `had_xmask`.** New incumbent **41.30**, still not 50 (−17%, 24.21 →
20.0 ms/token). Next was rows_per=48 at maxb=4.

## 35. Session ISA (2026-09-23): rows_per=48 at maxb=4 is a miss (−1.0%)

`EXL3_SQ_ROWS_PER=48` on `goal-isa-had` vs `isa-had-r40.json` (41.30).
Env only, same image, token parity true. Occupancy stayed `maxb=4` / grid=384.

| arm | median tok/s | runs | parity | sq k=5120 n=17408 |
|---|---|---|---|---|
| **rows=40** (`isa-had-r40.json`) | **41.30** | 41.39 / 41.30 / 41.19 | (ref) | grid=384 ksplit=8 |
| rows=48 (`isa-had-r48.json`) | 40.89 | 40.98 / 40.89 / 40.83 | **true** | grid=384 ksplit=7 |

−1.0%. All three r48 runs sit below all three r40 runs. Taller slices still
lose occupancy even after Hadamard freed LDS (`maxb` 3→4). **Keep 40.**
Occupancy-shape closed at maxb=4 as well (32/40/48/56/64). Next was decode-only
recapture.

## 36. Session ISA (2026-09-23): Hadamard mix — occupancy win, not shorter GEMV

Prefill-drained recapture (`had-r40-dec`) on `goal-isa-had`. Drain iters=3.
Unprofiled reference **41.47 tok/s** / 64 tokens (24.11 ms/token). Profiled
wall 2.049 s / 1369 ms device = 67% busy is **profiler overhead**.

63 profiled tokens:

| family | ms | ms/token | share of 24.11 ms | vs swizzle-r40-dec |
|---|---|---|---|---|
| `exl3_gemv_int8` sq+msq | 1153 | **18.30** | **76%** | 18.29 (unchanged) |
| paged-attn decode | 64 | 1.01 | 4.2% | 1.00 |
| GDN recurrent + conv1d + ba_gemv | 79 | 1.25 | 5.2% | 1.28 |
| rms_norm + gated_rms | 43 | 0.69 | 2.9% | 0.68 |
| other device | 30 | 0.48 | 2.0% | 0.44 |
| host idle (unprofiled − device) | — | **~2.38** | **10%** | was ~3.1 |

Hot GEMV avgs unchanged: `sq<3>` 55.8 vs 55.1 µs, `sq<4 r-on>` 37.4 vs 37.7,
`msq<4 r-on>` 65.9 vs 66.4, `sq<6>` 1143 vs 1175. Hadamard removed 40
`ds_bpermute` and raised occupancy 3→4; it did **not** cut VALU inner-loop
time (48 `v_dot4` + 49 `v_mul_lo` on K=4; K=3 is still 32-dot narrow extract).

CU-mode occupancy wall: 4 × 256-thread blocks = 32 waves = 2 SIMD32 × 16
slots. **maxb=4 is the ceiling** for this block size. GRID_MULT=2 × sms=48
× maxb=4 = 384 = 4/CU, occupancy fill. Occupancy-shape closed. Host-to-zero
tops out ~45.8 tok/s.

**50 still needs GEMV ~24% faster.** Inner loop: K=4 VALU (`v_mul_lo`+`v_dot4`);
K=3 is 7.76 ms/token on scattered `ext8w` (2 global loads/extract, wrap_idx).
Closed: VOPD, WMMA, packed 16, 24-bit hash, BUFFER_LOAD (K=4), STREAM,
NARROWN, STAGE_SMEM, A15 K=3 register pipeline, A12 prefetch, rows 32/48/56/64.

## 37. Session ISA (2026-09-23): GRID_MULT=3 at maxb=4 is a miss (−5.9%)

`EXL3_SQ_GRID_MULT=3 EXL3_SQ_ROWS_PER=40` on `goal-isa-had` vs
`isa-had-r40.json` (41.30). Env only, same image, token parity true.

| arm | median tok/s | runs | parity | sq k=5120 n=17408 |
|---|---|---|---|---|
| **MULT=2** (`isa-had-r40.json`) | **41.30** | 41.39 / 41.30 / 41.19 | (ref) | grid=384 maxb=4 |
| MULT=3 (`isa-had-r40-g3.json`) | 38.86 | 38.94 / 38.86 / 38.79 | **true** | grid=576 maxb=4 |

−5.9%. 576 = 6 blocks/CU; occupancy is maxb=4 (4/CU = 384). Same oversubscribe
pattern as CU MULT=3 at maxb=3 (36.93 vs 38.36) and WGP MULT=2. **Keep MULT=2.**
Do not raise the default. Occupancy fill is the CU-mode wave-slot ceiling
(4 × 256-thread = 32 waves). Next was rows_per=36 — invalid, rounds to 40.

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

## 39. Session ISA (2026-09-23): K=3 independent BFE is a wash (−0.17%)

`goal-isa-k3ext`: `ext8w` bits==3 uses two funnel shifts + independent `BFE16` instead of the
serial `w >>= 3` chain. Token parity true. Same env (MULT=2 rows=40).

| arm | median tok/s | runs | parity |
|---|---|---|---|
| **serial extract** (`isa-had-r40.json`) | **41.30** | 41.39 / 41.30 / 41.19 | (ref) |
| independent BFE (`isa-had-k3ext.json`) | 41.23 | 41.31 / 41.23 / 41.16 | **true** |

−0.17%, overlapping. Incumbent K=3 sq M=1 r-off already has 25 `v_bfe_u32` and only 4
`v_lshrrev_b32` — hipcc already broke the chain. **Revert.** Occupancy-shape and K=3
extract ILP are closed. 50 still needs GEMV ~24% faster (VALU `v_mul_lo`+`v_dot4`).

24-bit hash split was three VALU ops vs one `v_mul_lo` (−27% at ILP 8). Next is a
**2×256 LDS LUT**: `w*C == T0[w&255]+T1[w>>8]`. Two `ds_load_b32` can issue on the
LDS pipe against `v_dot4`; that is a different bet than the VALU split. Isolated
probe `profiling/hash_lut_probe.hip` before any e2e wire-up. Do not rebuild the
extension unless the probe beats `mul+dot` at ILP 8/16 (consume_row shape) with
0 mismatches.

## 40. Stop (2026-09-23): 50 not reached; incumbent 41.30 rechecked

User stop. Do not claim 50. Best measured 4096/256 b1 greedy (no speculation,
fp16 KV) remains `isa-had-r40.json` on `exllamav3-rocm:goal-isa-had`:

| run | engine tok/s |
|---|---|
| 1 | 41.387 |
| 2 | **41.296** |
| 3 | 41.193 |
| **median** | **41.30** |

Live recheck on the same image (`had-r40-recheck.json`, same env
`EXL3_SQ_GRID_MULT=2 EXL3_SQ_ROWS_PER=40`): 41.313 / **41.258** / 41.201,
median **41.26**. Token IDs match `isa-had-r40.json` and the pre-ISA golden
`final-post-revert.json` (33.00 tok/s era). 89 saved 4096/256 JSONs share
fingerprint `5c2c9f75e76d`. Three-run self-match true. ELF
`.workgroup_processor_mode` all `0x00` (CU). Launch log `grid=384 maxb=4
sms=48 rows_per=40`.

GEMV numerical (`test_msq_ab.py` on this image): **PASS**. msq vs per-matrix
int8 is bit-exact (rel_rms 0) at m=1 and m=4, sliced and plain. msq vs coop
rel_rms 0.006-0.008 (tolerance 0.02; per-slice vs global scales, by design).
Repeat-identical true.

Natural greedy smoke (`The capital of France is`, 64 tok): loops
`France is` (ids 9338, 369). The 4096/256 bench prompt likewise period-14
repeats a rotation of the filler sentence; that sequence is identical to the
pre-optimization golden, so it is not an ISA regression. The short-prompt
loop is a 3.5bpw greedy quality observation, not a kernel-parity failure.

50 is **not** reached (-17%; 24.21 ms/token vs 20.0). Host-to-zero caps
~45.8. GEMV is still 18.3 ms/token (76% of wall).

## 41. Session resume (2026-09-30): five new misses; GEMV streaming wall confirmed structural

Resumed on `exllamav3-rocm:goal-50tps-3e6bd94` (3e6bd94 + dirty ISA tree, CU-mode).
Baseline recheck: `resume-baseline.json` 41.42 / **41.36** / 41.28 = **41.36**
median, consistent with §40's 41.30. Decode-only recapture `resume-dec`
(`decode_profile.py`, CTX=4096, 64 tok): unprofiled 41.43 tok/s, device
1362ms/63tok; sq<3> 320.8, msq<4 r-on> 197.7, sq<4 r-on> 192.5, sq<3 r-on>
165.3, sq<4 r-off> 138.5, **sq<6> (lm_head k=5120 n=248320) 72.0 ms =
1143 µs/token ≈ 834 GB/s**, msq<4 f32> 57.9, paged-attn split+combine 63.2,
GDN chain 67.9, norms 26.1, misc ~30. GEMV block 17.9 ms/token.

New measured closures (all vs the resume baseline, canonical 4096/256 greedy):

| arm | median tok/s | vs 41.36 | verdict |
|---|---|---|---|
| narrow-unit register prefetch (bits 3/5/6/8 ext8w source words, next row in cur/nxt regs) | 39.42 | **−4.7%** | reject; same lesson as A12 — narrow does not want early loads |
| wide-unit pipeline depth 2→4 (r0..r3, +8 VGPR) | 40.01 | **−3.2%** | reject; per-warp MLP is not the limiter |
| `EXL3_SQ_STAGE_SMEM=1` (cp.async staged unit on all K) | 39.66 | **−4.1%** | reject (re-check on had stack) |
| `TG_KV_BITS=8` on had stack | 40.41 | **−2.3%** | reject (Attempt-10 re-check, still flat) |
| 2×256 LDS LUT for `w*0x83DCD12D` (`hash_lut_probe.hip`) | — | probe | reject: lut+dot 41.7 vs mul+dot 69.6 G hash/s at ILP 8; 2×ds_load loses to 1×v_mul_lo |

lm_head standalone (`profiling/lmhead_bench.py`, K=6 n=248320, 954 MB/call):
800–827 GB/s across force_sms {0,96,192,384,768}, rows_per {40,80,160},
STAGE_SMEM {0,1} — flat. Not launch geometry, not per-warp MLP (narrow-pf
unchanged at 813), not unit choice (smem same). The GEMV wall is DRAM
sector/TLB throughput on this access pattern: every shape caps at
300–830 GB/s regardless of unit (narrow/wide/smem), pipeline depth, or grid.

Remaining documented levers are unchanged from §14.5: whole-step graph
capture (host gap 2.4–3.1 ms → ~1 ms would land ~45–46 tok/s, still short),
GDN micro-chain fusion (~1 ms device). 50 via GEMV alone is closed by
measurement: the unit must stream ≥1.4× more bytes/s than every unit type
attains on real tensors.

## 42. Session resume (2026-09-30): trellis repack probe — closed (lm_head is math-bound)

`profiling/repack_probe.hip`: K=6 narrow-unit inner loop (ext4w x2 + hash mul +
dp4a per block pair) over a lm_head slice (rows 0..39, 7760 pairs). MODE 0 =
legacy scattered 4B loads per lane; MODE 1 = repacked `B2[kb][pair][lane][8]`
contiguous 32B/lane (2x uint4). Checksums bit-identical (0 mismatches) — the
repack mapping is exact.

| arm | GB/s |
|---|---|
| scattered cold | 1333 |
| repacked | **871** |
| scattered warm (119 MB slice fits Infinity Cache) | 2182 |
| repacked warm | 869 |

Repacked sequential streaming is *slower* than scattered warm and equal to the
real lm_head rate (834 GB/s e2e): the K6 unit is already at the extract+hash+dp4a
issue bound, not the DRAM pattern. The repack lever is closed; full integration
dropped.

Corollary for the 50 tps budget: device floor = GEMV 18.3 + attn/GDN/norm 3.4 =
21.7 ms/token ≈ **46 tps** even with zero host idle. 50 is unreachable without
either (a) fewer weight bytes/token (different quant format — out of scope) or
(b) speculation (user-closed §5). Whole-step graph capture (scout report
SuccessiveWren) tops out at ~45-46 tps for this reason.

## 43. Whole-step graph capture (EXL3_STEP_GRAPH) — implemented, closed as regression

Implementation (kept, env-gated): `Generator._sg_forward` in generator.py captures
fwd_modules[1:] (all modules after the CPU embedding gather) into a
torch.cuda.CUDAGraph keyed by (bsz, ids_width, block_table width, recurrent
slots). Pinned staging uploads record as memcpy nodes; recurrent-state device
updates ride along in-graph; host bookkeeping (r.position) replicated on the
replay path. ext `Graph::disabled` honors EXL3_STEP_GRAPH so BC modules run their
kernel sequence eagerly inside the outer capture (hipGraphLaunch-in-capture is
dead on ROCm and BC arg patching is a host call that never replays).
`SlicedMultiLinear.c_ptrs` stages through a persistent pinned buffer so the
pointer table records as a memcpy node.

Measured (goal-50tps-sg image, canonical 4096/256):
- capture engages cleanly (66 modules, key=(1,1,{16,32})), 3-run self-parity PASS
- 37.2 tps vs 41.3 baseline = **-10%**

Microbenchmark (/tmp/graphbench.py): 800-node graph replay = 3.22 us/node
device-side dispatch on gfx1100 (~2574 us/replay) vs 5.28 us/kernel eager
enqueue. The eager pipeline's enqueue cost is already hidden under GPU work;
the serial per-token cost is the post-sync host tail (~2.5 ms: sync, receive_sample
bookkeeping, staging, CPU embedding gather). An ~800-node whole-step graph swaps
that ~2.5 ms tail for ~2.5 ms of serial node dispatch: net ~zero, then negative
because BC's ~130 nodes were already replayed cheaply (and BC kernels run eager
inside the step graph carry extra arg-fixup cost). Same mechanism as the §14.2
BC-attn regression, now measured at whole-step scale.

Remaining lever to reach the 46 tps device ceiling: the post-sync host tail must
overlap the next step's GPU execution — i.e. launch step N+1 immediately after
the token readback, run receive_sample/staging while the GPU works. That needs
device token feedback (sampler writes the input-ids buffer in-graph) and
deferred host bookkeeping — a generator restructure, not a kernel change.

## 44. Prefill host-drain fixes (pinned staging + async recurrent stash)

Cold-2k prefill (bench_lean, nonce-poisoned so no prefix-cache hits):
**548 → 771 tok/s (+41%)**; profiled 2063-token job 1689 ms wall vs 1861 ms
device (~98% GPU-busy, was ~72%). Greedy A/B vs clean image: identical token
stream. Decode unaffected (42.2 tok/s on goal-50tps-3e6bd94 .so).

Sources of the ~1.4 s host stall removed:
- `Job.recurrent_checkpoint` -> `RecurrentCache.put` -> per-layer
  `*.stash()` did `.cpu()` on recurrent/conv state: blocking D2H per layer
  per page-boundary chunk = 96 x ~14 ms hipMemcpyWithStream for 2 chunks.
  Fix: stash into pinned buffers with `non_blocking=True` (CachingHostAllocator
  records stream events on free; unstash H2D is stream-ordered). Applied in
  gated_delta_net, short_conv, sliding_attn, ple.
- Embedding CPU output: `pinned_staging` param + double-buffered pinned ring
  with deferred cuda events (buffers alternate; writer waits on the event
  recorded after the buffer's previous upload was enqueued).
- `seq.sequence_ids` pinned (SeqTensor pin flag, growth preserved) and
  `block_index_tensor` allocated pinned -> prefill_ids slices + block table
  upload async.
- `cache_seqlens`: fresh 4-byte pinned tensor per chunk. Do NOT reuse one
  buffer: the host writes the next chunk's value while the previous upload
  is still queued behind ~700 ms of GEMMs (observed-as-designed allocator
  event tracking makes per-chunk allocs safe).

Gotchas:
- `pin_memory=True` under ROCm goes through torch's CachingHostAllocator;
  `hipHostMalloc` cost (~1 ms/pinned alloc, 98 allocs = 93 ms) is the
  remaining prefill host line — could be pooled, not worth it yet.
- bench_lean's original head-only nonce let prefix-cache hits mask ~40% of
  prefill; nonce now interleaved every 96 positions.
- HIP extension rebuilds are NOT bit/perf-reproducible in practice:
  goal-50tps-3e6bd94 .so = 42.3 dec, goal-50tps-sg (dirty kernel WIP) =
  37.3, goal-50tps-fix1 (HEAD kernels + CUMODE) = 32.6. The shipped .so is
  the best binary found; kernel WIP (xmask DPP, sq_k4 variants) regressed
  decode in this state. Serve overlay keeps old .so + repo python via
  sys.path shadowing.
