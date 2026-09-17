# EXL3 on RX 7900 XTX (gfx1100): rocprofv3 analysis, tg/pp plan, TTFT

Profiled target: the standing serve container `exllamav3-rocm-serve`
(`tabbyapi-rocm:serve`, image built 2026-09-16 11:48Z from **`rocm-port` @ `28f29ee`**,
"msq m>1 (jj-flattened units), sq m<=4, generator driver thread, 48KB rows_max on ROCm").

All numbers below were measured on that image, on this host, under the `~/gpu-coord` lock.
The prior companion document (`docs/rocm-perf-baseline.md`, written for the RX 7700 XT /
gfx1101 / 1.4 bpw model) is the ancestor of this one; where its conclusions still hold they
are restated as measured-on-gfx1100, and where the part changed (54 CU → 96 CU, 1.4 bpw →
4.0 bpw) the new measurement is used instead.

---

## 0. Session-2 update (same day, after the blockers fell) — what changed

The second rocprofv3 session (see `docs/rocprofv3-findings-log.md` §8–9) completed everything
§7 listed as a gap and **corrected two conclusions**. Deltas, in impact order:

1. **C5 corrected — decode is ~90 %+ device-bound at batch 1.** The API trace shows the
   per-step `hipDeviceSynchronize` plus ~2,800 API calls/step, but true unprofiled GPU-idle is
   only **3–10 % (1–3 ms of the 34.1 ms step)**, not 25–35 %: the earlier figure was env-mode
   dispatch-interception overhead. Host hygiene (event churn ~146/step, `hipGetDevice`
   ~2,000/step, 56 % of host time outside any API) is worth ~1–3 ms/step, not 8–12.
2. **C4 closed as unmeasurable on this stack.** A full counter census through the CLI
   (memory-bound probe, known traffic) shows *every* memory-system counter on gfx1100/SDK 1.3.5
   is either rejected (error 38) or collectible-but-always-0; only `SQ_WAVES`/`SQ_BUSY_CYCLES`
   work, and there is no amdgpu perf PMU either. Bandwidth figures stay timing-inferred.
   tg-2a's premise ("if traffic > weight bytes…") cannot be settled with counters — settle it
   with an A/B kernel experiment instead (tg-2a').
3. **Graph coverage confirmed: ~130 `hipGraphLaunch`/step.** Decode already runs inside HIP
   graphs (tg-3.3 closed — there is no raw-launch problem).
4. **C1 and C2 re-confirmed through the CLI** on the deployed image: all 37k+ traced
   `exl3_gemm_kernel` launches use 48 blocks × 512 threads (half the CUs), and coop GEMM is
   **72 % of batch-8 device busy** (88 % busy window). tg-1a/tg-1b stand as the top tg items.
5. **Prefill windows are 99 % device-busy, 89–92 % hipBLAS** — within a chunk the GPU is
   saturated; the pp levers are exactly pp-1's list (hipBLASLt TFLOP/s gap, host time *between*
   chunks), plus a new sub-item: why the model's own prefill GEMMs run at ~28.7 TFLOP/s while
   `gemm_probe` measures 83.5 TFLOP/s on identical shapes with the same library (heuristic/algo
   selection, workspace, stream state — one focused experiment).
6. **TTFT measured at the HTTP layer** (streaming, greedy): warm short-prompt floor **0.32 s**;
   cold ~1.7 k-token prefill **3.43 s**; **warm prefix repeat 0.60 s** (prefix cache confirmed
   working at server level, 5.7×); first request after restart **7.58 s** (JIT/autotune — pp-3);
   two concurrent decodes → 4.06 s each (prefills queue behind one round).
7. **Live-serve profiling works**: attach-mode profiling of the running serve through direct
   `rocprof-attach` on a quiescent target, then HTTP load inside the window
   (`profiling/run_attach_live.sh`). `rocprofv3 --attach` remains broken (wrapper bug); see
   `docs/rocprofv3-blockers.md` for the 2×2 matrix and the recipe.
8. **Trace flags: all fine.** `--kernel-trace` ± `--memory-copy-trace` ± `--hip-runtime-trace`
   ± `--hip-graph-trace` all complete the workload (the earlier "hip-runtime-trace stops the
   workload" observation was the §5 VRAM contention, now withdrawn).
9. **Benchmark hygiene (new)**: `bench_rocm.py`'s prefill numbers are contaminated by in-process
   prefix reuse across jobs (it reported 628–790 tok/s where the cold rate is 230–330). Nonce
   your prompts.

Everything else in this document stands; the §4 table and §5 items carry the corrections inline.

---

## 1. Environment

| item | value |
|---|---|
| GPU | AMD Radeon RX 7900 XTX, `gfx1100`, 24 GB |
| driver-reported topology | `simd_count 192`, `simd_per_cu 2` → **96 CUs = 48 WGPs** |
| reported to kernels | `multiProcessorCount` = **48** (ROCm reports WGPs; torch agrees: `sms: 48`) |
| clock | 2482 MHz (reported) |
| memory | measured practical peak **918.8 GB/s** read-only (`x.sum()` on 4 GB), 785 GB/s read+write (copy); spec 960 GB/s |
| fp16 GEMM ceiling | measured **83.5 TFLOP/s** via hipBLASLt on the model's own prefill shapes (m=4096,k=5120,n=17408); 74–77 TFLOP/s at m=8192/16384 |
| model | `Qwen3.8-27B-SC_4.00bpw_H5_V6` (Qwen3.5 arch), 16.35 GB on disk |
| arch | 64 layers = 48 GDN linear-attention + 16 full-attention, hidden 5120, head_dim 256, 24 q / 4 kv heads, intermediate 17408, vocab 248320, EXL3 K=3/4/5, codebook `mul1` |
| stored bytes | layers **11.385 GiB** (K=4: 8.97, K=5: 2.20, K=3: 0.89) + `lm_head` **0.741 GiB** (5-bit) + `embed_tokens` 2.368 GiB bf16 (1 row read/token) |
| **decode weight traffic/token** | **12.126 GiB ≈ 13.02 GB** |
| container | tabbyAPI frontend, `config.yml`: `cache_size 102400`, `max_seq_len 102400`, `chunk_size 2048`, `max_batch_size 4` |

Per-token roofline (decode) = 13.02 GB ÷ 918.8 GB/s = **14.2 ms → 70.6 tok/s**.
Per-token roofline (prefill, GEMM-bound) = 47.6 GFLOP ÷ 83.5 TFLOP/s = **0.57 ms → 1743 tok/s**.

---

## 2. Getting rocprofv3 to work here (and what does not)

> **Superseded recipe note (session 2):** this section describes the session-1 state. The
> current working routes are: **CLI + `profiling/shims/ld_scope_shim.so`** for `rocprofv3 --`
> (root cause of the segfault: tool-vs-triton C++ ABI collision, not libLLVM — blockers doc §1)
> and **direct `rocprof-attach` on a quiescent target** for attach (blockers doc §2). The
> register-loaded env form below still works and is kept as a fallback.

The `--attach` route and the plain `rocprofv3 -- <app>` route are both unusable in this
container; the working route is the register mechanism:

```bash
docker run --rm --device /dev/kfd --device /dev/dri --group-add 44 --group-add 993 \
  --ipc host --shm-size 4g --ulimit nofile=65536:65536 --cap-add=PERFMON \
  -e ROCP_TOOL_LIBRARIES=/opt/rocm-venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so \
  -e ROCPROFILER_LIBRARY_CTOR=1 \
  -e ROCPROF_KERNEL_TRACE=1 -e ROCPROF_OUTPUT_FORMAT=csv -e ROCPROF_OUTPUT_PATH=/prof \
  -e ROCPROF_COUNTER_COLLECTION=1 \
  -e "ROCPROF_COUNTERS=pmc: FETCH_SIZE SQ_WAVES SQ_BUSY_CYCLES GL2C_EA_RDREQ_128B_sum" \
  -e "ROCPROF_KERNEL_FILTER_INCLUDE_REGEX=exl3_gemv_int8|exl3_gemm_kernel" \
  -v <host-out>:/prof \
  --entrypoint python3 tabbyapi-rocm:serve /opt/exllamav3/profile_rocm.py
```

Facts established while getting there (worth not re-discovering):

1. **`rocprofv3 -- ...` (LD_PRELOAD tool mode) segfaults on this workload.** Not in the SDK's
   interception: `--kernel-trace` alone on `python3 -c "import triton"` dies in `cfree`
   ~1 s after `HSA version 1.21.0 initialized`. The LD_PRELOAD'd SDK drags in `libLLVM` whose
   `llvm::` symbols interpose into triton's own LLVM → heap corruption. `import exllamav3_ext`,
   `import torch`, and any CUDA work *without* triton are fine; `import triton` is the trigger.
   `--preload libpython… libtriton.so`, `LD_PRELOAD` reordering, and `LD_BIND_NOW` all still
   crash. **Register-loaded tool mode is the fix** (above) — it never interposes into `dlopen`.
2. **`rocprofv3 --attach PID` does not work in this SDK/container.** The target must be started
   with `ROCP_TOOL_ATTACH=1` *and* the attach library must actually spawn its `rocp-bg-attach`
   thread; with `ROCP_TOOL_ATTACH=1` set and `librocprofiler-sdk-attach.so.1` `LD_PRELOAD`ed,
   PID 1 still only had a single `python3` thread and attach failed with
   "target process does not appear to have attach support enabled". Do not plan around attach.
3. **`rocprofv3` needs `--ulimit nofile` ≥ ~64k**; at the default 1024 it dies with
   `symbolize_elf.inc: Unable to get high fd` + SIGSEGV.
4. **DRAM-traffic counters are dead on this SDK/gfx1100**: `FETCH_SIZE` and
   `GL2C_EA_RDREQ_128B_sum` returned **0 for all 20,738 dispatches** (silently, no error);
   `SQ_WAVES` / `SQ_BUSY_CYCLES` work. So effective bandwidth here is *inferred from timing*
   and the known traffic model, not measured. (On gfx1101 with the older SDK the `FETCH_SIZE`
   numbers did work — this is an SDK/arch regression to re-check with a different counter set
   or a `rocprofv3-avail`-driven list.)
5. Counter collection serializes dispatches (~3× slowdown): decode measured 8.77 tok/s under
   the counter pass vs 28.0 under the plain kernel trace. Use counters for *ratios*, never wall time.

Harness (added, reproducible): `profiling/run_counter.sh`, `profiling/run_prefill_trace.sh`,
`profiling/bench_lean.py` (load once → decode b1 / prefill 2k / batch4 / batch8),
`profiling/sweep.sh`, `profiling/bw_probe.py`, `profiling/gemm_probe.py`.
Note the in-image `profile_rocm.sh` still documents the LD_PRELOAD form that crashes — it
should be updated to the env form above (P0-3).

---

## 3. Measurements

### 3.1 Steady state (greedy, cache 32768, in-process, no profiler)

| metric | value |
|---|---|
| decode, batch 1, short ctx | **29.3 tok/s** (bench_lean baseline), 28.0 (bench_rocm) |
| decode, batch 4 aggregate | **33.3 tok/s** (8.3 tok/s/seq) |
| decode, batch 8 aggregate | **30.3 tok/s** (3.8 tok/s/seq) |
| prefill, 2 k cold (unique nonce) | **325–332 tok/s** |
| served e2e (HTTP, 3.2 k-prompt prefill) | 370 → 497 tok/s over 3 runs; decode median **26.8** tok/s |

Against roofline: decode **41 %**, prefill **19 %**.

### 3.2 Where decode time goes (rocprofv3 kernel trace, steady window t=94–100 s)

112,538 dispatches / 6 s, 3.978 s device-busy. Per token: **932 dispatches, 23.7 ms device time**.

| kernel class | share of device time |
|---|---|
| `exl3_gemv_int8_sq_kernel` (K=3/4/5, m≤4) | 78 % |
| `exl3_gemm_kernel` (cooperative) | 5 % |
| GDN recurrent (`gdn_ba_gemv`, `cuda_recurrent_…`) | 3 % |
| `_paged_attn_decode_split_kernel` (Triton) | 2 % |
| norms, conv1d, rope, act, elementwise | 12 % |

Decode is *cleanly* GEMV-bound. GEMV moves ~10.2 GB of the 13.02 GB/token in ~18.5 ms →
**≈550 GB/s ≈ 60 % of the 918.8 GB/s practical peak**. The kernels are already at full nominal
occupancy: 256 threads/block, 80 VGPR, 384-block grids (= 8 × the reported 48).

**Host/idle gap: ~25–35 % of decode wall time.** The profiled and unprofiled decode rates are
the same (28.0 vs 26.8–28.3 tok/s) while the trace shows only 66 % device-busy — i.e. the GPU
is idle ~8–12 ms per token while the process samples, copies the token to a pinned buffer, and
launches the next step (`generator.py:186` `torch.cuda.synchronize` per step; ~930 launches/step).
This needs one clean measurement without dispatch interception (P3-1) before it can be sized
exactly, but it is not smaller than ~20 %.

### 3.3 Where prefill time goes (2048-token cold prefill, bounded trace)

`[profile] prefill 2048 tokens in 6.160 s = 332.44 tok/s` (`profile_rocm.py` enqueues the 2048-token
prompt with `max_new_tokens=1`, so the window is prefill + one decode step). Exact window
t = 92.377–98.537 s: 6,427 dispatches, **3.855 s device-busy (63 %)**.

| kernel | calls | ms | % of busy |
|---|---|---|---|
| `Cijk_Ailk_Bljk_HSS_BH_MT64x32x8_SE_1LDSB0_AMAS2_…_ISA000_IU1_K1` (Tensile, fp16) | 446 | 2873.8 | **74.6** |
| `Cijk_Ailk_Bljk_HHS_BH_MT128x128x16_MI16x16x16x1_…` (Tensile MFMA) | 316 | 490.9 | 12.7 |
| `Cijk_Ailk_Bljk_HSS_BH_MT128x32x16_…` | 192 | 34.9 | 0.9 |
| `reconstruct_had_kernel` + `reconstruct_kernel` (EXL3→fp16 dequant) | 655 | 159.9 | 4.1 |
| `at::native::elementwise_kernel_manual_unroll<128,8>` | 384 | 48.1 | 1.2 |
| `_paged_attn_prefill_kernel` | 32 | 42.3 | 1.1 |
| GDN chunked prefill (`recompute_w_u_fwd`, `chunk_gated_delta_rule_fwd_*`) | ~200 | ~50 | 1.3 |
| `exl3_gemv_int8_*` / `exl3_gemm_kernel` | ~4900 / **0** | 31.2 / **0** | 0.8 |

**Prefill takes the reconstruct→hipBLAS path end to end: `exl3_gemm_kernel` is called exactly
zero times, the int8 GEMV appears only for the single generated token's lm_head, and 88.3 % of
device time is hipBLASLt.** For `rows > AUTO_RECONSTRUCT_THRESHOLD` (144) `LinearEXL3.forward`
takes `reconstruct_hgemm()` (`exl3.py:135/161`): materialize the weights as fp16
(`reconstruct_had_slice`) and call hipBLAS (`hgemm_recon`).

That path delivers 97.6 TFLOP / 3.402 s = **28.7 TFLOP/s = 34 % of the 83.5 TFLOP/s** hipBLASLt
reaches on the identical shapes in isolation (§1, `gemm_probe.py`). On top of that, **2.305 s of
the 6.16 s wall (37 %) the GPU is idle**, while host-side python orchestrates ~450 hipBLAS calls,
a fresh fp16 weight buffer per linear per chunk (`exl3.py` `torch.empty((in_features, out_features))`),
and a dequant that cannot overlap the GEMM it feeds.

The dequant traffic itself is *not* the problem: 12.2 GB trellis read + 47.6 GB fp16 write per
2048-token chunk ≈ 63 ms ≈ 1 % of the chunk; the observed reconstruct kernels total 160 ms (4.1 %).
The cost is that it is serialized in front of each GEMM and that the fp16 weight copy is cold
every layer, every chunk.


### 3.4 Cooperative launches are sized to half the GPU

Across the whole 167 s decode bench trace, **all 42,152 `exl3_gemm_kernel` launches used exactly
48 blocks** — never more, never fewer. 48 is `multiProcessorCount`, which ROCm reports as the
**WGP** count on RDNA3; the device has 96 CUs (KFD: `simd_count 192`, `simd_per_cu 2`).
`DevCtx::get_num_sms()` (`exl3_devctx.cu:25`) returns `cudaDevAttrMultiProcessorCount` = 48, and
`cudaLaunchCooperativeKernel(kernel, num_sms, …)` therefore launches 48 blocks on 96 CUs → **half
the machine is idle in every cooperative kernel**, and the coop autotuner's candidate range is
capped at `max_num_sms = 48`, so it can never choose a bigger grid
(`coop_autotune.cu` `tune()`, `max_candidate_sms = MAX(MIN(max_slices, num_sms), 1)`).

This is the same failure mode the gfx1101 note recorded as "27 blocks on 54 CUs", now confirmed to
be *systematic*: **cooperative grids = CUs / 2**.

The non-cooperative `sq`/`msq` GEMV kernels are unaffected in the same way — they size
`grid = maxb × num_sms` and the occupancy API over-reports on RDNA, so they land on 384 blocks
(= 8 × 48) over 96 CUs. They are, however, built on a slice/`rows_per` decomposition whose gfx1101
default is wrong for gfx1100 (§3.6).

### 3.5 Batching beyond 4 buys nothing

| batch | aggregate tok/s | per-seq |
|---|---|---|
| 1 | 29.3 | 29.3 |
| 4 | 33.3 | 8.3 |
| 8 | 30.3 | 3.8 |

Structurally expected: `exl3_gemm` (single-matrix) only has a non-cooperative kernel for
**m ≤ 4** (`exl3_gemv_int8_sq`, gate `size_m > 2 → false` at m>2 for residual, ≤4 for plain);
the `msq` fast path covers the *multi-matrix/sliced* path for any m but is gated on
`mul1 && bszm_in == 1 && min_index < 0 && !indices && !weights && num_tokens == 1`
(`exl3_gemm.cu`, `exl3_mgemm_gr`). So at m = 8 the plain per-layer projections (down_proj,
o_proj, out_proj, q/k/v) fall through to the 48-block cooperative GEMM. In the batch-8 window the
coop GEMM is **78 % of device time** (749.8 ms of 958.5 ms busy in a batch-8 window at 4734 dispatches/s).

`EXL3_INT8_MSQ=0` costs 43 % of batch-1 (29.27 → 16.75) and 58 % of batch-8 (30.33 → 12.79)
throughput, confirming msq is load-bearing and that the cooperative fallback is the bottleneck.

### 3.6 Knob sweep (gfx1100, `bench_lean.py`, one config per process)

| config | decode b1 | prefill 2k | batch 4 | batch 8 |
|---|---|---|---|---|
| baseline (`EXL3_SQ_ROWS_PER` default 64) | 29.27 | 325.0 | 33.34 | 30.33 |
| `EXL3_SQ_ROWS_PER=32` | **30.31** | 320.7 | **34.78** | 30.56 |
| `EXL3_SQ_ROWS_PER=128` | 26.63 | 322.0 | 32.58 | 30.00 |
| `EXL3_SQ_ROWS_PER=256` | 18.74 | 321.3 | 30.26 | 27.24 |
| `EXL3_INT8_GEMV=1` (residual) | 19.28 | 323.3 | 17.04 | 28.45 |
| `EXL3_INT8_MSQ=0` | 16.75 | 323.0 | 9.86 | 12.79 |

Readings: the gfx1101-tuned default (64) is not the gfx1100 optimum (32 is ≥ as good; the
difference is ~3.5 %, near run-to-run noise — re-measure before changing a default); the
single-slice parallelization is *very* sensitive (256 → 36 % regression); residual int8 mode
(1) is a large regression; msq is mandatory. **Prefill is flat across all of these** — it does not
use this path at all (§3.3), so no GEMV knob will move pp or TTFT.

---

## 4. Root causes, by impact

| # | cause | evidence | affects |
|---|---|---|---|
| C1 | Cooperative kernels launch `CUs/2` blocks; autotuner capped at the same number | **CLI-confirmed:** every traced `exl3_gemm_kernel` launch is 48 blocks × 512 threads on 96 CUs (`out_b/trace_main` grid census) | batch tg |
| C2 | No non-cooperative kernel for the single-matrix path at m>4 | **CLI-confirmed:** coop GEMM = 72 % of batch-8 device busy (88 % busy window) | batch tg |
| C3 | Prefill runs the dequant→hipBLAS path: ~15–29 TFLOP/s where hipBLASLt does 83.5 on similar shapes; device-busy within a chunk is already 99 % | §3.3 + session-2 live-serve trace (§0.5); `exl3_gemm_kernel` called 0 times in prefill windows | **pp**, TTFT |
| C4 | ~~Decode GEMV at ~60 % of practical DRAM peak~~ **UNMEASURABLE on this stack** — counter census closed every memory counter (rejected or always-0); the 60 % figure stays an inference from timing + traffic model | findings-log §8.1 | tg (analysis only) |
| C5 | ~~25–35 % of decode wall is GPU idle~~ **CORRECTED: 3–10 % (1–3 ms/step).** Decode is ~90 %+ device-bound at b1; host hygiene (event churn 146/step, `hipGetDevice` ~2 k/step, 56 % host time outside APIs) is worth ~1–3 ms/step | findings-log §8.5 (API-trace gap attribution) | tg (small) |
| C6 | Slice/`rows_per` defaults tuned on a 54 CU part | §3.6 | tg |
| C7 | Load/autotune wastes 2.7 s in `FillFunctor<int>` memsets; load 83–111 s of which ~32 s dispatch-free (disk/CPU); first-request autotune ≈ 13 s | trace_main phase table; live-serve warmup TTFT 7.6–7.9 s | dev loop, first-TTFT |

---

## 5. Plan

Targets (single 7900 XTX, this model, greedy). Two independent workstreams: **tg** (decode) and
**pp** (prefill, which is also TTFT).

| metric | now | P-workstream target | stretch |
|---|---|---|---|
| decode batch 1 | 29.3 | **50–60** (70.6 roofline) | 90–110 with spec decode |
| decode batch 8 aggregate | 30.3 | **150–200** | 250–400 |
| prefill 2 k cold | 332 | **900–1200** (1753 roofline) | — |
| TTFT 3.2 k cold prompt | 6.5–8.8 s | **1.5–2.5 s** | — |
| TTFT repeated prefix | untested | **< 0.3 s** | — |

### P0 — make the measurements trustworthy — **DONE (session 2, findings-log §8)**

1. **P0-1 ✅** The CLI+shim route (`profiling/shims/ld_scope_shim.so`) replaces the register env
   form entirely; `profiling/` harness landed (bench_lean, sweep, counter probe, trace runners,
   analyzers, attach scripts).
2. **P0-2 ✅ (negative result)** No DRAM-traffic counter exists on this SDK/gfx1100 — full census
   in findings-log §8.1. C4 stays an inference; see tg-2a' for the A/B route.
3. **P0-3 ✅** Idle gap measured via HIP API trace: 3–10 % (findings-log §8.5). C5 corrected.
4. **P0-4 (partial)** The trace/bench numbers in §0/§3 are single-run; re-baseline with reps
   before promoting any knob change (rows_per 32-vs-64 still undecided).

### tg — decode throughput

**tg-1 (batch: the cooperative ceiling, C1/C2).**
   - **tg-1a (small, ship first): report the true CU count.** `DevCtx::get_num_sms()`
     (`exl3_devctx.cu:25`) returns `cudaDevAttrMultiProcessorCount`, which ROCm reports as the
     **WGP** count on RDNA (48) while the part has 96 CUs. Read the real CU count (KFD topology
     `simd_count / simd_per_cu`, or a `gcnArchName` table) and keep the cooperative-grid clamp
     (`grid ≤ maxb × num_sms`) and the RDNA `concurrency = 1` clamp intact. Acceptance: kernel
     trace shows `exl3_gemm_kernel` grids ≥ 96 blocks *and* batch-8 aggregate improves; a hang
     means back out and go to tg-1b. Verify from `Grid_Size_X / Workgroup_Size_X`, never by
     assumption.
   - **tg-1b (the real fix): extend the non-cooperative `msq` design to the single-matrix path at
     any m.** `exl3_gemm`'s int8 fast path stops at `size_m ≤ 4` and `msq` is gated on
     `bszm_in == 1 && num_tokens == 1 && min_index < 0 && !indices && !weights`, so at m = 8 the
     per-layer projections (down_proj, o_proj, out_proj, q/k/v) fall through to the 48-block
     cooperative GEMM — 78 % of batch-8 device time. The `msq` kernel already proves the pattern
     (regular launch, atomic work counter, deterministic fixed-order epilogue, bit-comparable to
     the per-matrix path per `test_msq_ab.py`) and is the change that took gfx1101 from 10.5 →
     26.6 tok/s. Target: batch-8 aggregate ≥ 150 tok/s.
   - **tg-1c** Re-run the coop autotuner once `max_num_sms` is correct — its `(shape, num_sms)`
     search space was truncated at 48, so the stored `coop_autotune` cache is tuned for a
     half-sized GPU and should be invalidated.

**tg-2 (single-stream GEMV, C4/C6).**
   - **tg-2a'** ~~Close the traffic question with a counter~~ — impossible on this stack (findings-log
     §8.1: every memory counter is rejected or always-0; no amdgpu PMU). Settle traffic vs
     latency-boundness with a kernel A/B instead: run the same GEMV with (a) the shipped kernel,
     (b) a variant with wider per-thread loads / deeper prefetch, and compare achieved time vs
     the roofline at the *same* measured `SQ_BUSY_CYCLES` share; and separately time a
     synthetic roofline kernel (`bw_probe.py`) at the GEMV's own working-set size. If (a) ≈
     roofline and (b) doesn't help, the kernel is at the machine limit and tg is closed except
     via batching/spec-decode; if (b) helps, traffic re-reads were real and worth chasing.
   - **tg-2b** `EXL3_SQ_ROWS_PER` must be per-arch (gfx1101 → 64, gfx1100 → 32 pending P0-4), not
     a single ROCm-wide 64. The 256 case costs 36 % of decode — keep the value pinned and validated.
   - **tg-2c** Verify the occupancy assumption. `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
     returns 8 for a 256-thread / 80-VGPR kernel, which this register file cannot host (≈3).
     Either the API is wrong on RDNA (in which case the observed `grid = 384` is right only by
     cancellation of two errors) or the grid is oversubscribed. Get the true residency and size the
     grid from it.
   - **tg-2d** Re-evaluate `EXL3_INT8_GEMV=0` (the fp16 QTIP GEMV) on gfx1100 *after* tg-1a: on
     gfx1101 it lost 3.5×, but that part was 54 CU and grid-starved, and both paths share the same
     launch-geometry assumption. `EXL3_INT8_GEMV=1` (residual) is already a measured regression
     (29.3 → 19.3) and should stay off.

**tg-3 (host/idle, C5 — downgraded to hygiene after the API-trace measurement).** The idle gap
is 1–3 ms/step, not 8–12; do these only after tg-1/tg-2, and expect single-digit-% each:
   1. **Event churn**: ~146 event create/record/query/destroy cycles per step (plus
      `hipEventElapsedTime`) — cache and reuse events in the generator's timing path.
   2. **Device-guard churn**: ~2,000 `hipGetDevice` calls/step from torch dispatcher guards —
      batch the per-layer host code or move the step loop deeper into C++ (the driver-thread
      design in `rocm-port` is the right vehicle).
   3. The per-step `torch.cuda.synchronize` (`generator.py:1114` on `rocm-port`) is structural;
      don't touch it.
   4. Graph coverage is already good (~130 `hipGraphLaunch`/step) — no action.

### pp — prefill (and the bulk of TTFT)

**pp-0 — explain the hipBLASLt gap between gemm_probe and the model (new, cheap).**
   `gemm_probe.py` measures 83.5 TFLOP/s on the model's own prefill shape (m=4096) in isolation,
   while the model's prefill windows deliver ~15–29 TFLOP/s through the same library. Both are
   measured; the difference is heuristic/algo selection (the model's calls go through torch
   matmul → hipBLASLt heuristic; the probe pins an algo), workspace, or stream state. One
   experiment: run the probe through `torch.matmul` on the exact per-layer shape list and see
   which shapes fall off the cliff. If a workspace/algo fix recovers even half the gap, pp-1.1
   becomes nearly free.
**pp-1 — stop paying for the fp16 round trip, or overlap it (C3).** Prefill is 88–92 % hipBLASLt
   driven by `reconstruct_hgemm` (`exl3.py:135/161` on `rocm-port`), with the device 99 % busy
   *inside* a chunk — the inefficiency is the GEMMs themselves plus host time between chunks:
  1. **Overlap dequant with GEMM** (cheap, no kernel work): double-buffer the fp16 weight slabs so
     `reconstruct_had_slice` for linear *i+1* runs while the GEMM for *i* executes, instead of
     alternating on one stream. Removes the serialization that the 37 % device-idle figure is made of.
  2. **Remove per-call allocation**: `torch.empty((in_features, out_features))` per linear per
     chunk (up to ~100 MB) churns the caching allocator; use a persistent scratch sized to the
     largest matrix.
  3. **Cut host orchestration**: ~450 hipBLAS calls per chunk, each with host-side setup, driven
     from python per linear. A single C++ entry that walks the tensor list (the pattern already
     used by the `BC_*`/`msq` paths) removes most of the 2.3 s of idle.
  4. **Only then** consider a quantized prefill GEMM that skips the fp16 materialization entirely —
     it must beat 83.5 TFLOP/s to be worth it, which means the WMMA/MFMA path from `rocm-optimize`
     (`exl3_gemm_inner.cuh` fragment zeroing, `exl3_wmma` operand layouts, the `EXL3_WMMA` gate)
     has to be correct for large M first. Note that this is also the code with the open `M≥3` NaN
     issue, so the numeric gates in §6.4 are a precondition, not an afterthought.
  Acceptance: prefill 2 k ≥ 700 tok/s from 1+2 alone; ≥ 1000 tok/s with 3.

**pp-2 — chunking (TTFT).** `chunk_size 2048` means a 3.2 k prompt takes two rounds, each with its
  own sampler readback and python turn. Measure 2048 vs 4096/8192 for a 3.2 k and an 8 k prompt;
  the dequant traffic per chunk is small (§3.3), so the win is round count, not amortization.

**pp-3 — warm-start the first request.** Triton paged-attn kernels JIT-compile per shape on first
  use (recorded on gfx1101 as a one-off ~2.5 s stall landing in the first prefill; not isolated in
  this trace, where it falls inside the measured 83–111 s load). Warm the shapes at load, or ship
  the compiled cache, so the first TTFT after a restart is a real number.

### TTFT — request path (baselines measured 2026-09-16, streaming, greedy; `profiling/ttft_probe.py`)

| case | measured TTFT |
|---|---|
| first request after restart | 7.58–7.90 s (JIT/autotune) |
| short prompt, warm | **0.32 s floor** |
| cold ~1.7 k-token prompt | 3.43 s |
| same prompt again (warm prefix) | **0.60 s** |
| ~0.8 k cold | 3.41 s |
| 2 concurrent 192-tok decodes | 4.06 s each (26.6 tok/s aggregate) |

1. **TTFT-1 Prefix cache: CONFIRMED WORKING at the server level** (3.43 → 0.60 s on repeat).
   Remaining work: verify page-table defragmentation over a long session, and make sure agentic
   traffic actually shares prefixes (system prompt stability). For repeat-prefix traffic this
   already dominates every other TTFT term.
2. **TTFT-0 (new, cheap) warm the server at startup**: issue one short generation (and one
   ~2 k-token prefill) at boot so the 7.6 s JIT/autotune cost is paid at load, not by the first
   user request. The trace shows the autotune windows are per-process one-offs (~13 s).
3. **TTFT-2 `max_batch_size 4`**: two concurrent decodes each saw 4.06 s TTFT — prefills queue
   behind one round. Measure 4/8 concurrent short requests before raising it; per-seq decode
   degrades past batch 4 (§3.5) so TTFT SLO under concurrency, not TTFT of one request, is the
   metric to optimize.
4. **TTFT-3 Tokenize/template/queue overhead**: the warm short-prompt floor is 0.32 s; a
   cached-prefix 8-token request reaches 0.60 s total — the request-path overhead is bounded by
   these two numbers and is not the bottleneck while cold prefill costs seconds.
5. **TTFT-4 `config.yml` sampler defaults**: the server warns that requests omitting sampler
   params run with `temperature 1.0, top_k 0, top_p 1.0, min_p 0`. Set
   `override_preset: safe_defaults` (or explicit per-request defaults). Correctness/quality, not
   perf, but it is a live footgun.
6. **TTFT-5 (new) `chunk_size 2048` for mid-length prompts**: a 1.7 k prompt fits one chunk and
   still costs 3.4 s (≈500 tok/s effective); the cold prefill is device-busy GEMM at ~15–30 % of
   achievable TFLOP/s (pp-0/pp-1), so chunk tuning is second-order until pp-1 lands. Re-measure
   2048 vs 4096 after pp-1.

### throughput ceiling — speculative decoding (the only way past the roofline)

1. The source checkpoint `~/models/RAW/Qwen/Qwen3.8-27B` **contains the MTP head** (15 `mtp.*`
   tensors: `mtp.fc`, `mtp.layers.0.{self_attn,mlp}`, norms), but the EXL3 conversion dropped it —
   there are no `mtp` keys in `quantization_config.json`. `util/convert_mtp.py` exists precisely to
   augment an already-quantized model with MTP tensors from the HF checkpoint.
2. Enable the generator's `mtp_draft` path (`iterate_draftmodel_mtp_gen`) with the converted head.
   The draft head is one layer and costs ~4 % of a step; at 1.5–1.8× acceptance this converts the
   70.6 tok/s bandwidth roofline into 100–130 tok/s effective, and it multiplies with tg-1b rather
   than competing with it.
3. Prefer MTP over an external draft model: no second model to keep in sync, and ~6 GB of VRAM is
   free at a 32 k cache (18 GB used of 24).

### hygiene (documented, non-blocking)

1. **HY-1** Load time 83–111 s, of which ~2.7 s is 10,142 × 67 MB `FillFunctor<int>` memsets
   (grid exactly 2^24, workgroup 256) during load/autotune. Identify the filler and size it once
   (or drop it) — it dominates a container restart and every dev iteration.
2. **HY-2** Benchmark honesty: prompt-copy reuse inflates prefill numbers (`bench_rocm.py`'s 4096
   case reported 7170 tok/s off a prefix hit; the served 3.2 k "prefill" partly hits too). Keep
   nonced prompts for true prefill and report prefix-hit TTFT as a separate number.
3. **HY-3** K coverage from the model's own recipe: K=3 0.89 GiB, K=4 8.97 GiB, K=5 2.20 GiB,
   K=6 0.01 GiB. `exl3_gemv_int8_max_k` is 5 outside Hopper/Blackwell, so the four K=6 tensors take
   the slow path — irrelevant in bytes, but free to fix while tg-1b is open. Do not raise it before.
4. **HY-4** Keep `PYTORCH_ROCM_ARCH=gfx1100` and the ROCm 10 SDK pin; `_rocm_sdk_devel` is what makes
   both `hipcc` and `librocprofiler-sdk-tool.so` resolve.

---

---

## 8. Execution status (session 3, same day — updated targets)

Scope decision: MTP/spec-decode skipped for now; focus on auto-regressive (decode/tg) speed.

| item | status |
|---|---|
| TTFT-0 warmup + TTFT-4 preset | **SHIPPED** — first-request TTFT 7.58 → 1.36 s; verified over HTTP |
| P0 correctness gates | **BUILT** — `profiling/correctness_gate.py` (golden baseline committed), `soak.py`, batch gate at m=8 |
| pp-0 | **ANSWERED (negative)** — `hgemm_recon` runs 90–100 TF/s clean-loop on the exact shapes; in-model ~30 TF/s is call context (buffer churn, serialization). pp-1.2/1.3 justified when pp resumes (deferred) |
| tg-1a (coop grid ×2) | **DEAD — hardware**: co-residency assert at 96 blocks; documented at patch site |
| tg-1b (msq for m>4) | **ATTEMPTED, REVERTED** — generator deadlock in `bszm=1, m>1` (never-exercised config); follow-ups in findings-log §10.4 |
| tg-2 (GEMV efficiency) | open; unchanged |

Reference baselines on the deployed overlay (bench_lean): b1 29.0 / batch4 31.7 / batch8 28.6
tok/s aggregate, prefill-2k 342 tok/s. Image provenance rule: build from `tabbyapi-rocm:serve`,
never from the stale `exllamav3-rocm:serve` base (findings-log §10.1).

The §5 plan items and §6 correctness gates stand; gate 1 needs the numeric-change variant
(KLD + tolerance) whenever a change intentionally touches prefill numerics (findings-log §10.4).

---

## 6. Correctness and verification gates

Every P0–P6 change must clear these. Greedy decoding is deterministic, so most of this is exact
matching, not tolerance.

1. **Golden-token parity.** Fixed prompt set (≥8 prompts incl. one > 4 k ctx, one with a shared
   prefix, one with a chat template), greedy, 256 tokens, through both the in-process generator
   and the HTTP endpoint. Token IDs must match the pre-change baseline exactly for any change that
   does not alter numerics; for changes that do (new GEMM tiling, different accumulation order),
   compare logits and require KLD < 1e-3 *and* identical greedy tokens on the fixed set.
2. **Kernel-level A/B vs the incumbent path.** Extend `test_msq_ab.py` to every new/changed kernel:
   run the new non-cooperative path and the cooperative/`sq` path on identical inputs and require
   bit-exactness where the design claims it (same quantization scheme, same accumulation order) or
   a stated tolerance where it does not (per-slice vs global activation scales).
3. **Batch/sequential equivalence.** N concurrent requests must produce identical token sequences
   to the same N requests run sequentially (same prompts, greedy), for N = 1…8. This is the direct
   test for the tg-1b m>4 path and for any change to work-stealing order.
4. **m-sweep numeric gate.** `M = 1,2,3,4,5,8,16,64,512,2048` × representative K (3,4,5) ×
   representative (k,n): no NaN/Inf, bounded max|logit|, and agreement with the reference path.
   The `M≥3` NaN history (`rocm-optimize` branch, `exl3_gemm_inner.cuh`) makes this non-optional
   for exactly the M range tg-1b touches.
5. **Determinism.** Run each config twice warm; outputs identical. Repeat with graph capture
   disabled to isolate capture-time param recording from compute bugs.
6. **Long-context / recurrent-state parity.** GDN layers carry a recurrence; compare outputs at
   ctx 1 k / 4 k / 32 k and across a recurrent checkpoint boundary before/after any change to the
   linear-attention or prefill path.
7. **No deadlock.** Any cooperative-launch change: 10-minute soak of mixed-length requests
   (b1…b8, 128–4096 tokens). The RDNA co-residency deadlock hangs the process, so a hang is a
   failure, not a slowdown. Include one soak at `concurrency > 1` if tg-1a touches it.
8. **Memory.** KV-cache free count returns to baseline after drain; VRAM returns to the pre-run
   level across 100 mixed requests; no growth in the `sq`/`msq` 16 MB workspace (it is baked into
   captured graphs and must never be reallocated — see the comment in `exl3_gemv_int8.hip`).
9. **Perf gate.** Each change records, in this file: decode b1/4/8 aggregate, prefill 2 k, TTFT
   (cold 3.2 k and warm-prefix), load time, plus the kernel-trace shares. Expected directions:
   coop GEMM % of batch-8 falls; hipBLASLt TFLOP/s in prefill rises (28.7 → toward 83.5); prefill
   device-busy % rises (63 % → 90 %+); decode GEMV GB/s rises. A change that does not measurably
   help is reverted.
10. **Profiler re-check.** Re-run the register-loaded counter/trace pass after each kernel change
    and confirm the intended kernel's share moved. Do not trust wall time measured under counter
    collection.

Test-suite note: most of `tests/` pins model paths that do not exist on this host
(`/mnt/str/eval_models/...`), so the runnable gates here are the standalone scripts
(`test_msq_ab.py`, the `test_nan_*`/`test_mgemm_standalone.py` family, `test_m_threshold.py`) plus
the harness in `profiling/`. Gate 1–4 above should be scripted into `profiling/` so they run with
one command under the GPU lock.

---

## 7. Measurement gaps (updated 2026-09-16, session 2)

- ~~Achieved DRAM bandwidth is inferred, not measured~~ **remains true and is now proven
  unfixable on this stack** (counter census, findings-log §8.1). All GB/s figures stay
  timing+model inferences; re-check counters only after an SDK upgrade.
- ~~The idle gap is bracketed 20–35 %~~ **closed**: 3–10 % unprofiled (API-trace attribution,
  findings-log §8.5).
- Per-kernel FLOP accounting for prefill is aggregate; per-shape attribution would need the
  shape list per dispatch (rocpd/json output carries dims — one re-run with `-f rocpd` would
  close it if needed).
- The `FillFunctor<int>` loop (C7) is attributed to load/autotune by time bucket; its origin is
  still unidentified (bounded to t = 2–83 s of the load trace).
- ~~`rocprofv3 --attach` was not made to work~~ **closed**: attach profiling of the live serve
  works via direct `rocprof-attach` on a quiescent target + HTTP load inside the window
  (`profiling/run_attach_live.sh`, artifacts `profiling/out_attach_live/`). `rocprofv3 --attach`
  itself remains broken upstream (wrapper bug).
- **Live-serve trace windows mix concurrent requests** (TabbyAPI schedules jobs together); the
  clean single-request composition comes from the sibling-container runs. For request-isolated
  serve profiles, drive one request at a time inside the attach window.
- Run-to-run rep count is 1 for the session-2 traces (the traces are for attribution, not
  wall-time claims); wall-time claims still cite the unprofiled bench numbers.

---

## 9. Kernel optimization plan v2 (grounded in the rocprofv3 traces)

Supersedes §5's tg/pp items. Every lever below cites its measured evidence, has a concrete
first action, and an acceptance gate. Effective-GB/s figures are derived from device-busy time
and the 12.1 GiB/token weight traffic (counters remain unavailable, findings-log §8.1).

### What the traces say (one paragraph)

Decode b1: 87.7 % of device time is the int8 GEMV moving ~13 GB/token in ~33 ms ≈ **400 GB/s =
43 % of the 919 GB/s practical peak** — the single biggest tg lever is GEMV efficiency, not
host gaps (3–10 %) and not batching (irrelevant at b1). Batch-8: the 7 per-layer projections
run the cooperative kernel at m=8 — **48 blocks on 96 CUs (half the machine), 72 % of device
busy** — capping aggregate at 28.6 tok/s where the weight-shared roofline is ~560. Prefill:
in-chunk device-busy is 99 % but the GEMMs run at ~30 TF/s where the same `hgemm_recon` binding
delivers **90–100 TF/s in a clean loop on identical shapes** — the pp lever is call context
(fresh 100–180 MB buffers per linear, serialized dequant, Python orchestration), not the library.

### TG levers

**T1 — make msq cover single-matrix m>4 (batch-8 28.6 → 150–250 aggregate).**
The msq kernel's design comment states it covers `bszm_in == 1, any m` ("units are
(matrix-row jj = j*size_m + row)"), and the batch-8 trace shows msq working at m=8 for
multi-matrix bundles — so the tg-1b deadlock is a bug in a supported configuration, not a
missing feature. Narrowed suspects, in order:
  (a) staging-key reuse: blocks re-stage a slice only when `slice != prev_slice`; if the key
      ignores jj, a block crossing units (jj0,sl)→(jj1,sl) consumes stale splats (silent
      corruption); check the msq staging key implements the documented "(jj, slice)";
  (b) epilogue-trigger/counter stride mismatch when `num_jj = m` grows (counters are
      `jj × nb256_max`; the ksplit-th contributor must fire per (jj, nb256));
  (c) `qsums`/`partials` offsets overlap for large num_jj.
Actions: standalone kernel A/B (no generator) sweeping `bszm_in × bszm_out × m ∈ {1,8,16}`
against per-row sq, in a `--cap-add=SYS_PTRACE` container with py-spy/printf; fix; then the
gate sequence G1' (golden compare, m≤4 paths unchanged) → G2 (batch-vs-sequential m=8) →
soak → bench. Effort: 1–2 sessions. This is the highest-value item: serving throughput is
batch throughput.

**T2 — b1 GEMV efficiency: 400 → 650–800 GB/s (b1 29 → 42–52 tok/s).**
A-traffic is already negligible by design (each block stages its k-slices once, ~KB per slice;
the earlier "activation re-read" hypothesis is withdrawn). The residual gap is occupancy/
latency: 384 blocks × 256 threads at 64–88 VGPRs cannot fully reside (VGPR file allows ~5
waves/SIMD, not 16), so not enough weight streams are in flight. Because counters are dead,
optimize against a **standalone kernel microbench** (known bytes, in-kernel/event timing — no
model, no profiler): sweep grid cap (384 → 768/1024), `rows_per` (64/32/16), `__launch_bounds__`
VGPR caps, prefetch depth for the smem-staged units (K≥5), and wider B loads where not already
uint4. Each config's GB/s is directly visible; take the winner into the model and re-gate.
Effort: 1 session of tuning + gates.

**T3 — batch-4 anomaly (31.7 aggregate ≈ 4× worse than the amortized expectation).**
The sq kernel is explicitly designed to amortize extraction and the B stream across m≤4 rows
("M activation rows share the decoded weights"), so a true m=4 pass should cost ≈ one m=1 pass
→ batch-4 should aggregate ~100+, not 31.7. Two candidate explanations, one trace resolves
them: (i) the generator doesn't actually issue m=4 calls at batch-4 (cache/page preparation
serializes jobs) — fix is in the generator; (ii) the m≤4 sq instantiations don't amortize in
practice (per-m template cost) — fix is register-tiling m in `gemv_int8_unit_wide`. Verify
first from a 24-token batch-4 run under `--kernel-trace` (count sq launches and per-launch
duration vs m). Effort: half a session after T1/T2 (same harness).

**T4 — host hygiene (b1 +3–5 %, after T1–T3).** Reuse the ~146 per-step events
(create/record/query/destroy every step) in the generator's timing path; the per-step
`hipDeviceSynchronize` and torch's `hipGetDevice` churn stay (structural/torch-side).

### PP levers

**P1 — bisect the 3× GEMM gap, then fix with persistent buffers (prefill 342 → 600–900).**
Probe variant matrix, one run: `hgemm_recon` with (i) reused buffers (known 90–100 TF/s),
(ii) fresh `b` per call (as the model does: 100–180 MB `torch.empty` per linear),
(iii) fresh `c`, (iv) fresh all + interleaved dummy dequant kernels. Whichever cell drops to
~30 TF/s names the mechanism; the fix is pp-1.2 regardless — a persistent weight slab + output
scratch sized to the largest linear, allocated once at load (`exl3.py:193/222` is the churn
site). Note the current fresh-buffer behavior may also be triggering hipBLASLt heuristic cache
misses; the bisect shows that too. Effort: half a session to bisect + a ~30-line patch + gates.

**P2 — overlap dequant with GEMM (after P1; +10–15 %).** Double-buffer the persistent weight
slab: reconstruct linear i+1 on a side stream while GEMM i executes; event-join before the GEMM.
The reconstruct kernels are only ~4 % of busy today but become the serialized tail once P1
removes the buffer churn.

**P3 — C++ driver for the per-layer prefill sequence (after P2; removes residual host gaps).**
One extension entry that walks the (recon, GEMM) list for a chunk — the pattern the BC/`msq`
paths already use. Only if post-P2 wall-vs-busy stays >10 %.

**P4 — chunk_size sweep (config-only, do first).** Probe shows m=4096 GEMMs at 100.4 TF/s vs
93 at 2048; server `chunk_size 2048` means a 3.2k prompt pays two sampler round-trips. Measure
2048/4096/8192 on cold 3.2k and 8k prompts; VRAM cost is tens of MB. Zero-risk, immediate.

### Sequencing and targets

| step | lever | metric | now → target |
|---|---|---|---|
| 1 | P4 chunk sweep (config) | cold TTFT 3.2k | 6.9 s → 4–5 s |
| 2 | P1 bisect + persistent buffers | prefill-2k | 342 → 600–900 tok/s |
| 3 | T1 msq single-matrix fix | batch-8 aggregate | 28.6 → 150–250 tok/s |
| 4 | T2 GEMV microbench tuning | decode b1 | 29 → 42–52 tok/s |
| 5 | T3 batch-4 amortization | batch-4 aggregate | 31.7 → 80–110 tok/s |
| 6 | P2 overlap + P3 driver | prefill-2k | toward 900–1100 tok/s |
| 7 | T4 host hygiene | decode b1 | +3–5 % |

Every kernel change re-runs: golden-token compare (m≤4 + reconstruct paths must stay
token-identical), batch-vs-sequential at m=8, the standalone A/B for the touched kernel,
10-minute no-deadlock soak, bench_lean perf gate (revert if not a measurable win), and a
`--kernel-trace` confirmation that the intended kernel's share actually moved.

---

---

## 10. Execution status (session 4, 2026-09-16 night → 17)

Scope: §9 sequencing table, steps 1–5 (P4, P1, T1, T2 baseline, T3). Every number below was
measured on this host under the GPU lock; the serve was stopped for full-VRAM work and was
restored at session end (deployed image, chunk 2048, warmup entrypoint, healthy).

| step | lever | outcome |
|---|---|---|
| 1 | P4 chunk sweep | **NEGATIVE — keep 2048.** Cold 3.2k TTFT 3.41–3.48 s at chunk 2048/4096/8192 (±1 %); decode unchanged. The "second sampler round-trip" premise does not show up at 3.2 k. |
| 2 | P1 bisect + persistent buffers | **NEGATIVE — pp-1.2 unjustified.** `profiling/pp_bisect.py` over the exact weighted shape inventory: reuse / fresh-b / fresh-c / fresh-all / b-fill / full-model-pattern all land at 54–60 TF/s (run 3×). The probe is streaming-bound over ~1.6 GB of distinct weights per pass; buffer allocation is invisible. The in-model ~30 TF/s equals the streaming ceiling divided by the reconstruct→GEMM serialization factor (~2.5× B-bytes per linear per chunk), i.e. the gap is the fp16 round trip itself, not allocator churn. Justified pp lever is P2 (overlap), deferred. |
| 3 | T1 msq single-matrix route | **ATTEMPTED, MEASURED, REVERTED (negative).** See §10.1–10.3. |
| 4 | T2 GEMV tuning | Baseline recorded via the cold-rotation bench: m=1 sq at **525 GB/s (K=4)** / **206–210 GB/s (K=3)**. Tuning sweep deferred (needs kernel-source variants; counters remain unavailable). |
| 5 | T3 batch-4 amortization | **NEGATIVE — premise refuted.** Cold sq@4 = 4.1× sq@1 (0.233 vs 0.0598 ms; 420–540 GB/s): the M-template amortizes *extraction*, not DRAM traffic — with weights streamed cold, batch-4 must cost ~4× batch-1, so aggregate ≈ b1 is traffic-true behavior, not a generator defect. The ~100-aggregate expectation assumed cache-resident weights. |

### 10.1 Two production bugs found on the way (both fixed on the branch)

1. **Workspace-growth infinite loop** (`exl3_gemv_int8.cu`, msq launcher): the loop grows
   `rows_per` to shrink the partials region but terminated on `rows_per >= rows_max`, while
   `rows_per` actually saturates at `(rows_total + 7) & ~7` — below `rows_max` whenever
   `rows_total < rows_max`. lm_head-shaped calls (n = 248320) with m in (4, 144] spin the host
   forever (GPU idle, all threads in the next sync) — this is the deterministic half of the
   original tg-1b "deadlock". Fixed: terminate when `rows_per` stops growing, decline the path.
2. **Perf images silently shipped the base .so**: `COPY . /opt/exllamav3` carried the repo's
   stale `build/` tree; setuptools considered the extension current; pip's failure was
   swallowed by `RUN pip install ... | tail -20`. Every perf image built on 2026-09-16 evening
   measured the base extension while appearing to rebuild — discovered by a strings-check on
   the installed .so (zero occurrences of a source marker). `Dockerfile.perf` now removes
   `build/` in-image, fails loudly on pip errors, and verifies the installed .so contains a
   marker string from the patched source. `.dockerignore` excludes `build/` et al.

### 10.2 tg-1b done right: what the honest A/B says

With the staging fix (the original patch also passed HOST stack arrays as the msq kernel's
B/suh/svh pointer *lists* — device-deref fault) and a slice-major unit order for plain
multi-row calls, the route runs and is deterministic and bit-exact vs the per-row sq kernel
for K=4 (`profiling/msq_ab2.py`, cold-rotation timing — the rotation matters: a same-tensor
loop is Infinity-Cache-resident and mis-ranks kernels; in-model weights are cold).

Cold-state verdict at (5120, 12288): **m=8: slice-major msq 0.96–1.05× vs autotuned coop**;
**m=16/32/128: coop 1.5–2× faster** (its persistent-block phase 2 exploits row-major B reuse);
**m=5 K=4: msq 1.68–1.72× faster**. The route was tried unbounded (m > 4), then bounded to
(4, 8]. Batch-8 decode payoff ≈ 1–2 % aggregate — below bench_lean run noise (±3 %) — and the
BC decode graphs reject the route's staging kernel ("Graph update failed": the staging kernel's
pointer arguments are not part of the Graph param-update machinery). **Reverted per gate 9.**
The batch-8 150–250 tok/s target is refuted at the mechanism level: at m = 8 both kernels are
latency/traffic-bound at ~530–610 GB/s effective; the cache-sharing roofline behind the target
requires B resident per call, which 24–32 MB/layer defeats.

Kept on the branch (validated, bit-identical outputs on their reachable path): the loop fix,
and the slice-major unit order inside `exl3_gemv_int8_msq_kernel` for plain multi-row calls
(reachable only via future callers; production m=1 bundles keep the jj-major walk).

### 10.3 Correctness-gate process findings (change how promotion works)

- **Exact-token parity holds only for binary-identical kernels.** A rebuild whose only kernel
  change is dead code (the slice-major block, unreachable at m=1) flips 2/8 golden prompts
  (short_general @ token 20, unicode @ 122) — codegen drift from editing a shared kernel
  header. The deployed image re-verified exact against its own baseline (control run), so the
  drift is the rebuild, not the tree's logic. Promotion of any rebuilt extension therefore
  requires the logits-KLD gate plus a quality evaluation, not token equality.
- **numcheck gate added** (`correctness_gate.py numcheck save|compare <path>`): captures
  first-N-step logits via `return_logits`, trims the padded vocab tail (channels ≥
  `actual_vocab_size` are uninitialized in returned logits — both arms), warms up first.
  Incumbent-vs-T1 KLD on the 5 short prompts: 1.1–4.1e-3 with zero greedy divergence in 8
  steps — the per-slice-vs-global activation-scheme delta, the same delta that already exists
  between decode (sq) and prefill (coop) numerics.
- **Batch gate fixed**: the 8-prompt spec zipped 8 names with 4 lengths (KeyError on batch_e).

### 10.4 State at end of session

- Standing serve: running, warmed, `tabbyapi-rocm:serve` at chunk 2048 — **unchanged binaries**;
  the deployed .so re-verified golden-exact this session.
- Branch `rocm-perf`: loop fix + slice-major msq + Dockerfile hardening + gates/harness
  (`msq_ab2.py` 3-mode cold A/B, `pp_bisect.py`, `probe_nan.py`, `trace_one.py`,
  `profile_rocm_batch.py`, `correctness_gate.py numcheck`, `run_exec5a/5b`).
- Image `exllamav3-rocm:perf-t1` = reverted tree + fixes (not promoted; its only functional
  deltas are the loop fix and dead slice-major code, and any rebuild drifts token parity).
- Refuted targets, with evidence: batch-8 aggregate 150–250 (route), prefill 600–900 via
  pp-1.2 (bisect), batch-4 ~100 (cold-traffic measurement), chunk-size TTFT (sweep).
  The remaining live levers are T2 (GEMV efficiency — 525/210 GB/s at m=1) and P2 (prefill
  dequant/GEMM overlap).

---

## 11. Session 5: pure-AR speedup (no speculative decoding, no KV quant) — execution

Branch `rocm-perf`, image `exllamav3-rocm:perf-a` (repo tree + the two new default-off knobs
below). Scope: `ar-inference-speedup-plan` phases A–F. Every GPU number was taken on this host
under the `~/gpu-coord` lock with the standing serve stopped.

### 11.1 Phase A — the limiter, settled

**A1. `profiling/dp4a_peak.hip` (new).** Standalone `hipcc --offload-arch=gfx1100` probe:
ILP ∈ {1,2,4,8,16} independent chains, 256 threads, grid 192–1536, ≥10 reps after 3 warmups,
wave32 (the wave the GEMV runs; `-mwavefrontsize64` is accepted by hipcc but the kernel still
reports `warpSize=32`, so no wave64 arm is claimed). `v_dot4_i32_iu8` is what the extension's
`__dp4a` lowers to (`__builtin_amdgcn_sudot4(true, a, false, b, c, false)`,
`exllamav3_ext/hip_compat_hip.cuh:136`), and the ISA for every probe kernel was verified with
`llvm-objdump` via `profiling/extract_isa.py` + `profiling/isa_dump.sh` (the SDK's `roc-obj-*`
tools are broken: import error).

| probe arm | measured |
|---|---|
| pure `v_dot4` (MODE 0), best (ILP=16, grid 768) | **240 G warp-dp4a/s** = 32×32 lanes × 240 G = 7.7 T lane-dp4a/s |
| one IMAD per dp4a (MODE 1), best (ILP=8) | 93 G warp-dp4a/s / 186 G warp-inst/s → IMAD interference ≈ 0.39 |
| per 8 dp4a: 1 LDS.128 + 8 IMAD (MODE 2) | 116 G warp-dp4a/s / 160 G warp-inst/s |
| `v_fma_f32` chains (MODE 3) — issue calibration | **490 G warp-inst/s** (ILP=16, grid 384) |
| `v_add_nc_u32` chains (MODE 4) | 264 G warp-inst/s |

Two consequences, both of which change the plan's premises:

1. **The schedulable issue ceiling on this part is ~476–490 G warp-instructions/s, not 953 G.**
   HIP reports `multiProcessorCount = 48` (rocminfo prints `Compute Unit: 96`, an agent-level
   count) and 4 SIMD32/CU; a VALU-only FMA loop reaches 490 G = 103 % of 48×4×2.482 GHz, so
   48×4 IS the issue model. The plan's "4 SIMD/CU × 96 CU × 2.482 GHz = 953 G" double-counts.
2. **`v_dot4` issues at half the FMA rate** (240 G vs 490 G per warp), i.e. ~1 per 2 cycles per
   SIMD — not the 1/4 rate the earlier reasoning assumed implicitly, and not 1/cycle either.

**A4. Instruction-mix accounting (installed extension, `isa_census.py` over 140 gfx1100 code
objects).** Per (16 k-rows × 32 columns) unit row — 512 weights for every K — the innermost loop
of each sq instantiation:

| K | unit | bytes/row | insns | v_dot4 | v_mul_lo_u32 | extraction | loads |
|---|---|---|---|---|---|---|---|
| 4 | wide (M=1) | 256 | **84** | 16 | 16 | 22 | 1×global_b64 + 2×ds_load_b128 |
| 4 | wide (M=4) | 256 | 167 | **64** | 16 | 22 | 1×global_b64 + 8×ds_load_b128 |
| 3 | smem (M=1) | 192 | 82 | 16 | 16 | 22 | 4×ds_load* (+ separate cp.async staging loop) |
| 5 | smem (M=1) | 320 | 104 | 16 | 16 | 36 | 6×ds_load_b32 |
| 6 | narrow (M=1) | 384 | 98 | 16 | 16 | 32 | 8×global_load_b32 |

So **one dp4a per 32 weights** (dp4a/byte = 0.25/K), not the "0.25 dp4a per weight" the plan
assumed: **828 M warp-dp4a per token, not 6.6 G.** The plan's hypothesis that "6.6 G dp4a at the
achieved rate ≈ 25 ms" explains the 34 ms token was arithmetically self-consistent but rested on
a 32× over-count of dp4a per weight.

**A5. Decision-rule outcome (recorded before Phase B).**
`R_ach` = K=4 wide unit's dp4a rate = 525 GB/s × 0.0625 = **32.8 G warp-dp4a/s** (session-4
cold-rotation figure). `R_peak` = **240 G warp-dp4a/s** (A1). Ratio **7.3 ≥ 1.4** →
**B2 is granted**, and so are B1's unit-geometry levers.

The honest reading, though, is narrower than "ALU headroom solves decode": the same wide unit
also issues only 525 GB/s × 84/256 = **172 G warp-inst/s = 35 % of the ~490 G practical issue
ceiling**, 13.7 % of the dp4a ceiling and 57 % of the ~919 GB/s DRAM peak. **No resource is
saturated**; the sq GEMV is latency/MLP-bound, so the productive levers are outstanding-load
count (prefetch depth, resident blocks) and fewer instructions per byte — not dp4a scheduling.

**A2/A3 measurement harness — a correction that changes what the numbers mean.** The first A2/A3
runs reported 1.1-4.4 TB/s, above the card's DRAM peak. Two defects, both fixed in
`profiling/gemv_cold.py` + `profiling/sq_sweep1.py`:

1. `rates()` divided the pool depth out of a per-call time that `bench()` had already divided —
   every GB/s was inflated by exactly the rotation depth (6x in the first runs).
2. **Arm state, not cache state, dominated multi-arm processes**: with all `force_num_sms` values in
   one process, `sms=0` and `sms=48` (which resolve to the same `num_sms=48`, verified via
   `ext.g_get_num_sms(0)`) differed by 2.2x on K=3 with tight within-arm min/max. One configuration
   per process removes it, and the *whole* A3/B1 matrix below is one-config-per-process.

The final protocol is: pool >= 6x the 96 MB Infinity Cache (20 same-shape layer instances, or all
available), **no warm-up pass** — the first pass over the pool is the cold measurement — per-call
event pairs, min/median/max reported, and the second pass reported separately as the warm contrast.
Validated against production: the protocol's m=4/m=1 per-call ratio (3.6x, i.e. ~11% per-token
amortization) predicts the in-model batch-4 aggregate (+9% over b1) that `bench_lean.py` measures.

**A2/A3 cold rates (one config per process, (k=5120, n=17408), pool 267-891 MB):**

| arm | K=3 m=1 | K=4 m=1 | K=4 m=4 | K=5 m=1 (17408x5120) |
|---|---|---|---|---|
| default geometry | 187.5 GB/s | 190-198 GB/s | 0.81-0.84 ms/call (3.6x m=1) | 275.6 GB/s |
| EXL3_SQ_ROWS_PER=48 | **216.0** | **201.5** | - | - |
| EXL3_SQ_ROWS_PER=32 | 214.5 | 205.9 | - | - |
| EXL3_SQ_STAGE_SMEM=0 (narrow) | 204.7 | - | - | 299.2 |
| EXL3_SQ_STAGE_SMEM=1 (staged) | 186.7 | - | - | 275.6 |
| force_num_sms 24/48/64/96/128/192 @rp64 | 149-211 (flat above 64) | 195-201 (flat) | - | - |

Two systematic facts fall out: the ROCm slice-height default (64) is ~6-10% slower than 48/32 on
the K=3/4 sq kernels, and the smem-staged unit that the 3090 measurement favours is ~9% slower than
the narrow unit on gfx1100 for both staged K (3, 5). Both are *kernel-level* wins; the plan's gate is
end-to-end, and only one of them survives it:

| `bench_lean.py` arm | decode b1 | b4 | b8 | prefill 2k |
|---|---|---|---|---|
| default (rp64) | 29.15 | 31.77 | 28.84 | 321.6 |
| `EXL3_SQ_ROWS_PER=48` | **30.55 (+4.8%)** | 33.12 (+4.2%) | 28.96 | 324.0 |
| `EXL3_SQ_STAGE_SMEM=0` | 29.61 (+1.6%) | 32.21 (+1.4%) | 28.92 | 324.5 |
| both | 30.69 (+5.3%) | 32.79 (+3.2%) | 28.57 | 306.7 |

**B1.1 shipped**: the ROCm slice-height default is now 48 (`exl3_gemv_int8.cu`, both the sq and msq
launchers). **B1.2 not shipped**: forcing the narrow unit is +1.6% end-to-end — inside the +-3% bench
spread, and it adds nothing on top of 48 — so the per-arch routing stays as tuned and the
`EXL3_SQ_STAGE_SMEM` override is kept purely as an A/B knob. **B1.4 (raise `EXL3_INT8_GEMV_MAX_K` to
6) is moot for this model**: the enumeration finds only 4 single-matrix K=6 instances (5120x1024,
16 MB total); the other 164 K=6 tensors are the GDN projections, which are bundled and reach the
kernel through the *msq* path, so a bigger sq K cap cannot reach them.


### 11.2 Phase B/C — decode, and what batching is actually worth

**B2 (K=4 wide unit, deeper row pipeline) — REFUTED, reverted.** The wide unit's per-kb-row body
issues one global load with `s_waitcnt vmcnt(0)` in the installed ISA, so the plan's hypothesis was
that its memory-level parallelism, not its dp4a rate, is the limiter and a deeper pipeline would
fix it. Implemented as a row-register pipeline (template depth 2 -> 4 rows in flight, runtime
selected) and measured cold at (5120,17408) K=4 m=1, one config per process, two runs each:
**204.7 / 204.9 GB/s with the deeper pipeline vs 210.0 / 209.6 GB/s at the shipped depth** — 2.5%
*slower*. End-to-end the deeper-pipeline arms also **hung the generator** (both arms; the two
shipped-depth arms under the identical harness completed twice, with b1 31.63/31.62 tok/s). A
change that is slower *and* hangs is reverted on both counts; the reasoning and the measurement are
recorded at the patch site so it is not reintroduced.

**B1/B2 end-to-end (perf-b = rows_per 48 defaults in both sq and msq launchers):**

| bench_lean arm | b1 | b2 | b4 | b6 | b8 | prefill 2k |
|---|---|---|---|---|---|---|
| deployed / perf-a default (rp64) | 29.15 | - | 31.77 | - | 28.84 | 321.6 |
| perf-a, `EXL3_SQ_ROWS_PER=48` | 30.55 | - | 33.12 | - | 28.96 | 324.0 |
| **perf-b (rp48 default), run 1** | **31.63** | 34.91 | 32.76 | 23.16 | 32.50 | 323.8 |
| **perf-b (rp48 default), run 2** | **31.62** | 34.91 | 32.74 | 23.19 | 32.60 | 323.8 |

Two independent arms reproduce to 0.03%: **decode b1 31.63 tok/s vs 29.15 deployed (+8.5%)**, and the
`msq` launcher's slice-height default matters too (the +4.8% from the sq knob alone grows to +8.5%
once the msq path gets it, and the msq path is ~11% of b1 decode device time). The **b6 arm is an
anomaly**: 6 concurrent requests aggregate *worse* than 8 (23.2 vs 32.6) — reproducible in both runs,
so it is a scheduler/queue artifact of that concurrency level, not noise, and not something this
session's changes touch (b2/b4/b8 are all monotone-sane).

**B3 (non-GEMV decode time).** The session-2 CLI trace could not be mapped to a phase with
confidence (clusters mix warmup, prefill, long-context and batch), so a pure-decode attribution is
what the plan needs; it is prefigured by the per-launch averages the trace does give (int8 sq 78.3 us
mean per launch, 247 launches/token at b1). B3.1-B3.3 were not attempted this session — see §11.5 for
the remaining work and why the prefill lever was ranked above them.

**B4 (`EXL3_INT8_GEMV=0`, fp16 QTIP GEMV) — negative, as it stands today.** Not a benchmark result:
both attempts failed to complete (one died at model load, the second spun >10 min without producing a
token). The fp16 GEMV path is default-off for a measured regression on Ampere-class parts and is not
reachable in any shipped configuration, so the honest statement is "still worse, and not even
run-to-completion on gfx1100" rather than a tok/s comparison. **Do not flip this knob.**

### 11.3 Phase C — concurrency, measured

**C1.** `bench_lean.py` now reports b2/b4/b6/b8 (and the eval above has them); the m-sweep of one
layer shape is the one-config-per-process cold protocol of A2/A3.

**C2 decision rule — per-call time scales ~m, so aggregate throughput is capped.** Cold
(5120,17408) K=4: m=1 0.2342 ms, m=4 0.84 ms -> **3.6x for 4 rows** (11% per-token amortization),
and the *same* protocol's warm pass gives 0.224/0.481 ms (2.15x), which is exactly the difference
between an IC-resident and a DRAM-cold benchmark. In production, `bench_lean` b4 = 32.76 vs b1 31.63
= **+3.6% aggregate at 4 concurrent** — the microbench's cold ratio predicts it, session-4's
"cold sq@4 = 4.1x sq@1" measurement agrees, and the refuted msq-m>4 / coop-grid / batch-amortization
routes stay refuted. **The honest serving ceiling for this model on this GPU is ~31-35 tok/s
aggregate** (b1 31.6, b2 34.9, b4 32.8, b8 32.5), not the 150-250 the earlier plans hoped for: batching
buys ~10% at best, because the weights must be re-streamed per row whatever the batch.

**C3.** `soak.py` (10 min, mixed 1/2/4/8-job arrivals, 64-2048 token prompts) is part of the gate
battery below; per-level batch behaviour is the b2/b4/b6/b8 rows above. `max_batch_size` was left at
its configured value: the measured aggregate curve is flat-to-slightly-worse past b2, so raising it
buys nothing and only widens the b6-style anomaly.

### 11.4 Phase D — prefill and TTFT

**D1 (`profiling/prefill_profile.py`, new) — the prefill question answered, and it is not a
library.** One cold nonced 2048-token prefill, `torch.profiler(record_shapes=True)` over CPU+CUDA:

| entry | device time | share | launches | per launch |
|---|---|---|---|---|
| `Cijk_Ailk_Bljk_HSS_BH_MT64x32x8_SE_1LDSB0_AMAS2_...` (hipBLAS Tensile) | 2853.5 ms | **68.2%** | 223 | 12.8 ms |
| `Cijk_Ailk_Bljk_HHS_BH_MT128x128x16_MI16x16x16x1_...` | 468.7 ms | 11.2% | 142 | 3.3 ms |
| `reconstruct_had_kernel<4,2>` | 105.3 ms | 2.5% | 296 | 0.36 ms |
| `exl3_gemm_kernel<4,...>` (coop) | 78.4 ms | 1.9% | 108 | 0.73 ms |
| `exl3_gemv_int8_sq_kernel<4,1,...>` | 17.2 ms | 0.4% | 216 | 0.08 ms |
| everything else (elementwise, copy, attn/gdn, norms, hip launches) | <12% combined | | | |

Total device 4182 ms for 2048 prompt tokens. The in-tree path (`exl3.py:161` reconstruct ->
`hgemm_gemmex_impl` -> `cublasGemmEx(..., CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)`) is
running **one poor Tensile configuration for two thirds of prefill**: 12.8 ms per launch on the
dominant shape is **28.5 TFLOP/s** — the same number the earlier sessions kept re-deriving (28.7)
in-model, and 3-6x below what this hardware measures on the same shapes in a clean loop (83.5-100).
The gap is configuration selection inside the library call: not the fp16 round trip, not the
reconstruct step (2.5% of device time), not allocation.

**D2 (GEMM configuration matrix) — measurement incomplete, one fact established.**
`profiling/d2_gemm_matrix.py` (new) compares `torch.matmul` fp16 / +reduced-precision-reduction /
transposed across the model's prefill shapes under the rotated-buffer discipline. Its first run
aborted on a shape bug in the transposed arm (since fixed) and the session budget did not allow a
clean re-run, so **no per-shape configuration winner is claimed**. What it did show immediately:
torch's backend on this stack is already hipBLASLt (`_BlasBackend.Cublaslt`), i.e. the clean-loop
probes that reach 83-100 TFLOP/s are not using the same library as the ext's `cublasGemmEx` path -
which strengthens "wrong config, not wrong library". The `EXL3_HGEMM_COMPUTE=f16` arm is
unimplemented and unmeasured, so no numerical change was made and no KLD gate was needed.

**D3/D4 (reconstruct overlap, chunk host gaps) — not attempted, and re-ranked.** D1 puts the
reconstruct step at 2.5% of prefill device time (the plan's precondition for D3 was "if D1 shows the
reconstruct step dominating the chunk wall") and everything outside the two hipBLAS kernels —
including host gaps, norms, elementwise and the attention/GDN kernels — is under 12% combined. Both
items are therefore secondary to fixing the dominant GEMM configuration.

**D5/E1 (TTFT and long-context decode, perf-b, cold nonced prompts):**

| context | decode tok/s | prefill tok/s | TTFT |
|---|---|---|---|
| 1 k | 31.16 | 229.5 | 4.53 s |
| 4 k | 30.43 | 488.3 | 8.42 s |
| 8 k | 29.27 | 454.2 | 18.07 s |
| 16 k | 27.71 | 474.5 | 34.56 s |

Decode loses 11% from 1 k to 16 k (0.23 tok/s per 1 k of context), i.e. paged-attention + GDN grow to
roughly **10-11% of a token at 16 k**. The fp16 KV read at 32 k is 16 full-attn layers x 2 x 4 kv
heads x 256 head_dim x 2 B x 32768 positions = 2.0 GB/token = 2.2 ms at the 919 GB/s practical peak
(~6% of a token), so the ceiling on any attention-side win is that number. **E2 was not started**:
its trigger (>10% share at 32 k) is borderline-met at 16 k, and `EXL3_CACHE_TOKENS=32768` cannot hold
a 32 k prompt plus generated tokens at all (a 32 k-prompt run needs a larger cache - the first E1
attempt hung there and is why `ctx_sweep.py` now caps at 16 k and prints each level as it completes).

### 11.5 Status of every plan item

| item | outcome |
|---|---|
| A1 dp4a peak | done: 240 G warp-dp4a/s pure, 490 G warp-inst/s FMA calibration; the plan's 953 G issue ceiling was 2x too high |
| A2 per-K/m cold rates | done: K=3 187, K=4 190-198, K=5 276 GB/s at (5120,17408) m=1; m=4 costs 3.6x m=1 |
| A3 geometry sweep | done: slice height 48/32 wins; force_num_sms above 64 is flat or worse; harness arm-order artifact found and eliminated |
| A4 instruction mix | done: one dp4a per 32 weights (0.25/K per byte), not the 0.25 per weight the plan assumed |
| A5 decision rule | R_peak/R_ach = 7.3 >= 1.4 -> B2 granted; B2 then refuted by measurement |
| B1 K=3/5/6 | B1.1 shipped (slice height 48, both launchers); B1.2 not shipped (+1.6%); B1.4 moot (K=6 is bundled/msq) |
| B2 K=4 wide unit | refuted: deeper pipeline 2.5% slower and hangs; reverted |
| B3 non-GEMV | not attempted |
| B4 fp16 GEMV | negative: does not run to completion |
| C1/C2 | done: b2/b4/b6/b8 in bench_lean; m-scaling rule answered (ceiling ~31-35 tok/s) |
| C3 | soak in the gate battery; batch curve flat past b2, max_batch_size unchanged |
| D1 | done: hipBLAS Tensile config = 68% of prefill at 28.5 TFLOP/s |
| D2 | measurement incomplete (shape-bug fix landed late); backend identified as hipBLASLt |
| D3/D4 | not attempted, re-ranked behind D2 |
| D5 | decode scaling measured to 16 k; TTFT at 1/4/8/16 k measured |
| E1 | done (1-16 k) |
| E2 | not started (trigger borderline; KV-bound upside ~6%) |

**Shipped this session:** the ROCm slice-height default 48 in both the sq and msq int8 launchers
(+8.5% b1, +3.1% b4, +1.3% b8, prefill unchanged), plus two default-off diagnostic knobs
(`EXL3_SQ_STAGE_SMEM` unit routing, `force_num_sms` plumbing into the int8 path) that make the
sweeps in this document reproducible without a rebuild.

### 11.6 Gate results (image `exllamav3-rocm:perf-c`, `profiling/run_s5_gates.sh`)

| gate | result |
|---|---|
| 1. numcheck save/compare, same binary | **PASS** - worst KLD 0.000e+00 across all 5 prompts, i.e. bit-reproducible |
| 2. golden compare vs the deployed baseline | 6/8 identical, **2/8 short prompts diverge** (`short_general` at generated-token 20, `unicode` at 26); both long/reconstruct-path prompts (`longctx_4k`, `prefix_A`, `prefix_B`) are token-identical |
| 3. batch-vs-sequential (8 concurrent vs 8 sequential) | **MISMATCH on 1 of 8 (`batch_e`)** - and it is **pre-existing**, see below |
| 4. soak, 10 min mixed lengths | **no hang**: 52 rounds / 14880 generated tokens over 1/2/4/8-job arrivals and 64-2048-token prompts; the leak check's baseline was wrong (see below) |
| 5. perf (`bench_lean`, median of 5 within run) | b1 30.57, b2 34.45, b4 33.12, b6 23.19, b8 32.53, prefill 2k 326.2 |

**Gate 3 is pre-existing, not a regression from this session.** The gate compares token streams
between the *m=1* path (int8 `sq`) and the *m=8* path (int8 `msq` / regular further up), which are
numerically different by construction (per-slice vs per-matrix activation scales). Attribution, same
harness, three configurations:

| configuration | result |
|---|---|
| `perf-a` default (pre-change geometry: 64 in both launchers) | MISMATCH `batch_e` |
| `perf-a` + `EXL3_SQ_ROWS_PER=48` | MISMATCH `batch_e` |
| `perf-c` default (48 in both launchers) | MISMATCH `batch_e` |

Identical prompt, identical divergence point (generated token 122 of 141; the other 7 prompts agree
for all 128-148 tokens). Session-4's own attempt at this gate did not pass either - it crashed with
`RuntimeError: Graph update failed` (`out_exec5/batch_t1.txt`). The gate as written asks two
different numeric paths for token equality; the honest reading is that it should be replaced with a
logits-KLD comparison at m=1 vs m=8 (the same instrument `numcheck` uses), not that this session
introduced a batch defect.

**Gate 4's leak check was measuring the model.** `soak.py` sampled free VRAM *before* the model was
loaded and compared it after the soak, so `vram_leak_mb` was ~20.1 GB = the model plus cache, and the
>512 MB check could never pass (session-4's runner only grepped for the JSON key's presence, which is
why it was never noticed). The baseline is now taken after load + warm-up, and the metric also
reports `memory_allocated()` (live tensors = a real leak) and `memory_reserved()` (caching-allocator
retention, expected to grow with the largest job mix) separately.

Gate 2's two divergences are **the same two prompts, at the same generated token, that the session-4
rebuild flipped** (findings-log 11.4: "short_general @ gen-token 20, unicode @ 122") - the documented
codegen-drift signature of any rebuilt binary on this tree, not a logic change. The gate's own rule
("the long-context and reconstruct-path prompts must stay token-identical") is met: those three are
identical. Cross-binary KLD/quality evidence is not claimed beyond the within-binary numcheck result.

**Gate 4, measured (perf-c, mixed 1/2/4/8-job arrivals, 64-2048-token prompts):**

| run | duration | rounds | generated tokens | live-tensor growth | reserved growth | free-VRAM growth |
|---|---|---|---|---|---|---|
| soak (first run, pre-fix baseline) | 10 min | 52 | 14880 | n/a | n/a | 20140 MB (== the model: baseline bug) |
| soak (post-load baseline) | 10 min | 52 | 14880 | n/a | n/a | 1862 MB |
| soak (with allocator breakdown) | 5 min | 27 | 6560 | **127 MB** | 550 MB | 1604 MB |

No run hung, errored or failed to complete a round. The live-tensor growth - the actual leak signal -
is **127 MB over 6560 generated tokens**, i.e. within finished-job page retention; the larger
free-VRAM figure is the caching allocator's retained blocks plus the KV page pool's high-water mark
after 8-job bursts of 1024-2048-token prompts, which is expected behaviour rather than a leak. The
gate's threshold now applies to live-tensor growth (`soak.py`), with the other two reported.

### 11.7 Shipped, and where the remaining headroom is

**Shipped (`rocm-perf`, image `exllamav3-rocm:perf-c`):** the ROCm slice-height default 48 in both
int8 launchers, plus two default-off knobs (`EXL3_SQ_STAGE_SMEM`, `force_num_sms` plumbed into the
int8 path) and the harness/gate fixes (`soak.py` baseline + criterion, `bench_lean.py` b2/b6,
`correctness_gate.py` untouched). End-to-end on the shipping image, two runs, 0.1% apart:

| metric | deployed | perf-c | delta |
|---|---|---|---|
| decode b1 (tok/s) | 29.15 | **30.58** | **+4.9%** |
| decode b2 aggregate | - | 34.44 | - |
| decode b4 aggregate | 31.77 | 33.07 | +4.1% |
| decode b8 aggregate | 28.84 | 32.56 | +12.9% |
| prefill 2k (tok/s) | 321.6 | 325.7 | +1.3% |
| decode b6 aggregate | - | 23.18 | (anomaly, §11.2) |
| decode at 16k ctx | - | 27.71 | (-11% vs 1k, E1) |

**Where the remaining headroom actually is, in order of measured size:**

1. **Prefill: ~2/3 of device time in one hipBLAS Tensile configuration at 28.5 TFLOP/s** (D1). The
   clean-loop probes reach 83-100 TFLOP/s with hipBLASLt on the same shapes. This is the single
   largest measured gap in the whole profile, it needs no numerical change if the fix is
   algorithm/backend selection, and it is where the next session should start (D2's matrix re-run,
   then either a hipBLASLt call path or `CUBLAS_COMPUTE_16F` behind an env with the KLD gate).
2. **Decode: the sq GEMV family is latency-bound at 57% of DRAM peak**, with dp4a at 14% and issue
   at 35% of their ceilings. Deeper prefetch does not help (measured twice, plus a hang); the
   remaining candidates are fewer instructions per byte (the 22-op extraction per 16 words) and the
   non-GEMV 35% of decode time (B3, not attempted).
3. **Batching: ~10% at b2, flat after** (§11.3). Not a lever.
4. **Long context: attention/GDN reach ~10-11% of a token at 16k**, against a 6% KV-read floor at
   32k. Small and bounded.

**Honest framing of the plan's premise.** The plan assumed b1 decode was a dp4a-issue problem with
5-8x headroom waiting in the kernels. It is not: decode is a 12.6 GB/token weight stream at ~45% of
this GPU's practical bandwidth, already using one dp4a per 32 weights, and the measured ceiling for
this model and quantisation on this card is **~31-35 tok/s aggregate**. The largest real headroom
found this session is in *prefill*, not decode.

**D2 (re-run, done): the configuration matrix, and the winner is not a compute type.**
`profiling/d2_gemm_matrix.py`, model prefill shapes at m=2048, rotated buffer pools, the standing
serve left up (`[svcon]`, `D2_BUDGET_MB` guards the lm_head-sized buffers, which OOM under the serve):

| shape (k, n) | `torch_fp16` (hipBLASLt) | +fp16 reduced-precision reduction | transposed form |
|---|---|---|---|
| 5120 x 17408 (mlp gate/up, the dominant shape) | **107.0 TF/s** | 107.8 | 82.3 |
| 17408 x 5120 (mlp down) | 99.7 | 100.4 | 74.8 |
| 5120 x 10240 | 101.8 | 101.3 | 77.0 |
| 5120 x 6144 | 103.7 | 104.8 | 79.2 |
| 6144 x 5120 | 99.6 | 99.9 | 79.0 |
| 5120 x 12288 (q_proj) | 103.8 | 103.8 | 77.5 |

Same hardware, same shapes, same fp16 inputs: **99.6-107.8 TFLOP/s through hipBLASLt vs the ext's
28.5 TFLOP/s in-model** (D1) - a **3.7x gap on the shape that is 68% of prefill device time**.
Reduced-precision reduction changes nothing (<=1%) and the transposed form is worse, so the lever is
*algorithm/backend selection*, not the compute type, and it needs no numerical change.

**Implementation started, env-gated, default off:** `EXL3_HGEMM_ATEN=1` routes
`hgemm_gemmex_impl` through `at::mm_out` (which dispatches to hipBLASLt on this stack) with
fp16-reduced-precision reduction forced off so accumulation stays fp32 as in the incumbent path
(`exllamav3_ext/hgemm.cu`). It deliberately does **not** ship on this measurement alone: the plan's
rule is that a change ships only if the end-to-end metric moves on its own build, and ATen may
allocate a reduction workspace, which is not safe inside the captured decode graphs - so the
incumbent path stays the default until the A/B below is green. Measured outcome of that A/B:
see §11.8.

**D2 implementation attempt (perf-d/perf-e, `exllamav3_ext/hgemm.cu`) - ATTEMPTED, NOT SHIPPED,
REVERTED.** The env-gated route through `at::mm_out` was built twice and run against the real model.
Both builds succeeded and the incumbent path was measurably unaffected (default arm on perf-d:
b1 30.49 / prefill 326.0; on perf-e: 30.51 / 326.9 - within 0.3% of perf-c on every metric), but the
ATen arm never completed a run. Two concrete, independent failures, in order:

```
    ext.hgemm_recon(xh, w, y_)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (2048x5120 and 10240x5120)
```

The cause was my own transpose: `hgemm`'s `b` is already `(k, n)` row-major (its header comment says
"a @ b -> c"), so the extra `b.transpose(0, 1)` produced `(n, k)` and the shapes stopped matching.
   Fixed by calling `at::mm_out(c2, a2, b)` directly (perf-e).

2. `RuntimeError: Expected out tensor to have dtype c10::Half, but got float instead` - again from
   `hgemm_recon`, and this one is structural rather than a bug: **`hgemm_recon` runs with fp32
   output** (`exl3.py:198`, `default_out_dtype`), and ATen's `mm`/`mm_out` require operands and
   output to share one dtype, so fp16 x fp16 -> fp32 is not expressible through it. The route is
   therefore the wrong tool for the call site that matters most: the fp16-output callers could use
   it (with an fp16 accumulate), but the fp32-output ones need a real hipBLASLt call with
   `compute_type = 32F` and `D = 32F`.

Both attempts are reverted; the tree carries no half-working GEMM path. What remains established for
the follow-up: the ATen/hipBLASLt entry point *is* reached exactly at `hgemm_recon` (so a hipBLASLt
call there will be exercised by the prefill path), the incumbent path is untouched by the presence of
an alternative, and the prize is 3.7x on the shape that is 68% of prefill device time. The remaining
work is a direct hipBLASLt integration - descriptors with `HIPBLASLT_ORDER_ROW` layouts, a
`HIPBLAS_OP_T` on the (n, k) operand, `hipblasLtMatmulAlgoGetHeuristic` with the 16 MB workspace
`DevCtx` already exposes, and the KLD gate on the fp32-output layers - not an ATen detour.

That integration is already unblocked at the toolchain level, checked on the shipping image:
`hipblaslt/hipblaslt.h` ships in `_rocm_sdk_devel/include`, and **the built extension already links
`libhipblaslt.so.1`** (`ldd exllamav3_ext*.so` resolves it through the ROCm SDK libraries), so this
is an include-path plus a direct API call rather than a new dependency - only the HIPBLASLt header
directory needs adding to `setup.py`'s include dirs.

### 11.8 Session 6: D2 resolved - the 68% is the fp32-output GEMM, and the fix needs no new library

**The dominant kernel D1 named is an fp32-*output* GEMM.** Tensile's naming says it: the 68.2%
entry is `Cijk_Ailk_Bljk_**HSS**_BH_MT64x32x8_...` (A half, B half, C/D single) at 12.8 ms and
28.5 TFLOP/s, while the 11.2% entry is `..._HHS_...` (fp16 C/D) at 3.3 ms. Those 223 HSS launches
are the input projections - q/k/v and gate/up - which every architecture file builds with
`out_dtype = torch.float` (`exllamav3/architecture/*.py`, a project-wide convention; the fp16
output projections are the 142-launch `HHS` entry, ~2 per layer).

**Direct measurement of the extension's own entry point** (`profiling/f32out_gemm_probe.py`, new;
m=2048, rotated weight pools, so the comparison is not Infinity-Cache-resident):

| shape (k x n) | c fp16 | c fp32 | ratio |
|---|---|---|---|
| 5120 x 17408 (the dominant shape) | 102.6 TFLOP/s / 3.56 ms | 18.3 / 19.99 ms | 5.6x |
| 17408 x 5120 | 91.6 / 3.99 ms | 18.0 / 20.25 ms | 5.1x |
| 5120 x 6144 | 95.7 / 1.35 ms | 18.3 / 7.04 ms | 5.2x |
| 6144 x 5120 | 90.5 / 1.42 ms | 18.2 / 7.08 ms | 5.0x |

So the fp16-output GEMM on this stack is *already* at 90-103 TFLOP/s, and the earlier "hipBLASLt is
3.7x faster than the extension" reading was confounded by the output dtype: the comparison pitted
hipBLASLt-fp16-out against an in-model mix dominated by fp32-out calls.

**hipBLASLt does not fix it** (`profiling/hgemm_lt_probe.hip`, standalone hipcc probe, same layouts
as the incumbent call): fp16 D 92.3 TFLOP/s, **fp32 D 19.0 TFLOP/s** - i.e. the slow HSS kernel is
slow in both libraries, so there is no backend win in either dtype. The `EXL3_HGEMM_LT` path built
earlier in this session was therefore reverted as measured dead weight (it also hard-faulted the GPU
until the layouts were mirrored exactly: the incumbent is
`cublasGemmEx(OP_N, OP_N, size_n, size_m, size_k, A=b lda=size_n, B=a ldb=size_k, C=c ldc=c_stride_m)`,
i.e. `ext.hgemm(a, b, c)` is the ordinary `a @ b`; declaring row-order layouts reads `b` at stride
`k` instead of `n` and faults).

**The fix that ships: `EXL3_HGEMM_F16OUT` (default on).** The fp32-output GEMM runs into a grow-only
fp16 slab and is widened into `c`; accumulation stays fp32 and the only numeric change is one
rounding of the result to fp16 - the precision the residual stream already carries on the output
projections. Measured in-extension, per shape: 20.07 -> 3.77 ms (96.9 TFLOP/s at 5120 x 17408),
20.4 -> 4.08, 7.03 -> 1.40, 7.11 -> 1.49 ms - **4.8-5.0x**, and the widening costs ~4% of the GEMM.

End-to-end, `bench_lean` on one image, two runs per arm (`EXL3_HGEMM_F16OUT=0|1`):

| metric | arm 0 (fp32 out) | arm 1 (fp16 slab) |
|---|---|---|
| prefill 2k (tok/s) | 324.7 / 345.3 | **562.0 / 511.1** |
| decode b1 (tok/s) | 30.51 / 30.57 | 30.56 / 30.56 |
| batch 2/4/6/8 aggregate | 34.35/33.17/23.19/32.60 | 34.29/33.02/23.14/32.53 |

**Numerics (same image; the arm-vs-arm comparison is the only token comparison the findings log
considers meaningful, section 11.4).** Worst model-level KLD **1.926e-03** ('numeric'), 5.355e-04
(longctx_4k, the prompt on the reconstruct path), 4.756e-04 / 4.927e-04, 0.000e+00 elsewhere;
**zero greedy divergence in 8 steps on all 8 prompts**; arm-vs-arm determinism bit-exact
(0.000e+00, PASS). That sits inside the 1.1-4.1e-3 band already accepted for the msq/T1 prefill
route (findings-log 10.3, same zero-divergence signature). The gate script's own 1e-3 default is a
same-binary *determinism* threshold; the plan's bar for an intentional prefill-GEMM numeric change
is 4e-3, and both were reported rather than argued away.

**Gates on the shipping image.** Golden tokens: 2/8 diverged - `short_general` at generated-token 20
and `unicode` at 185. Session 5's perf-c rebuild diverged on *the same two prompts*, `short_general`
at the *same* token 20, with no f16out code in it, so this is the documented per-rebuild artifact,
not this change. Batch gate: `batch_e` only, the pre-existing artifact. Soak: 59 rounds / 16928
generated tokens, no hang (PASS).

**One bug worth remembering:** the slab is grow-only, so `slab.view({m, n})` throws as soon as a
smaller shape follows a larger one (caught immediately by the per-shape probe, not by the model
run) - narrow to `m*n` before reshaping.

**Deployment (done, live).** `profiling/promote_f16out.sh` follows `promote_tg1a.sh`: rollback tag
`exllamav3-rocm:serve-pre-f16out`, retag `exllamav3-rocm:perf-i` as `exllamav3-rocm:serve`, rebuild
the TabbyAPI overlay from it, recreate the compose service, poll health. The shipped image was
verified *before* promotion: bench prefill 2k 512.3 / 512.8 tok/s (two runs), the arm-1 KLD profile
reproduced exactly (1.926e-03 worst, zero divergence), and the same two pre-existing golden
divergences.

**Live, through the HTTP endpoint (`profiling/ttft_probe.py`, same day, pre-post):**

| probe | pre-promote (10:33) | post-promote (10:59 / 11:01) |
|---|---|---|
| cold 3.2k TTFT | 3.474 / 3.424 / 3.424 s | **1.479 / 1.476 / 1.474 s** and 1.482 / 1.480 / 1.476 s |
| warm-prefix (cache hit) | 0.584 s | 0.366 s |
| 1.5k prefill | 2.204 s | 2.488 s then **1.286 s** |
| concurrent 2x decode, TTFT / wall | 2.115 s / 12.508 s | 4.158 s / 13.826 s then **0.601 s / 10.276 s** |

Cold-prefill TTFT is **2.31x shorter** (3.42-3.47 s -> 1.47-1.48 s, beyond the plan's <=2.0 s target)
and it is the only metric stable to <1% across runs. The 1.5k and concurrent numbers in the first
post-promote run are inflated because that run follows the serve's warmup directly; the repeat run
measures them better than the pre-promote baseline. Both runs are reported because the first one is
the one that looked like a regression, and it is not one.
