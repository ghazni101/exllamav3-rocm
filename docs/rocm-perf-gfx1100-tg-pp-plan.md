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
