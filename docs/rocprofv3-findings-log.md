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
