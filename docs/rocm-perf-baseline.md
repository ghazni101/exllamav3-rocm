# EXL3 on RX 7700 XT (gfx1101): baseline, profiling, optimization plan

Serving target: `Qwen3.8-27B-EXL3-SC_1.40bpw_H3_V3` (hybrid GDN + full attention,
64 layers, 248k vocab, 3bpw lm_head, bf16 embeddings).

## Environment (reproducible)

- GPU: AMD Radeon RX 7700 XT, `gfx1101` (RDNA3, 54 CU, wave32, 12 GB, ~432 GB/s peak).
- Host: pop-os, i7-12700, docker. Docker daemon restarts kill running containers —
  long runs should be committed to an image or restarted with the same script.
- Image: `local/rocm-base:10.0.0` (native ROCM 10.0 SDK, gfx1101, ships rocprofv3)
  + `Dockerfile.rocm10` → `local/exl3-rocm:gfx1101-v1` (torch 2.14.0+rocm7.14
  `--no-deps`, triton-rocm 3.8.0, exllamav3 HIP extension built for gfx1101 from
  the `rocm-port` branch).
- Branch: `rocm-perf` (based on `rocm-port`).
- Run container with `--ulimit nofile=1048576:1048576` (rocprofv3 needs fds) and
  `--cap-add=PERFMON` for counter passes.

## Baseline (rocm-port, no tuning), `bench_rocm.py`, greedy, cache 32768

| metric | value |
|---|---|
| load | ~74 s |
| VRAM at 8k cache | 8.38 / 11.98 GB |
| **decode, short ctx (batch 1)** | **6.0 t/s** (165 ms/token) |
| decode @1k ctx | 6.0 t/s |
| decode @4k ctx | 5.9 t/s |
| prefill 512 / 1k / 2k | 208 / 181 / 431 t/s |
| prefill 4096 | (prompt-cache hit, not a real prefill) |
| batch 8 aggregate | 17.8 t/s |

Roofline: per-token weight traffic ≈ 4.74 GB quantized layers + 0.48 GB (3bpw
lm_head) ≈ 5.2 GB → theoretical ≈ 83 t/s, ~70 t/s at 85% of peak bandwidth.
Current 6 t/s is ~10× below the hardware ceiling, and roughly constant across
context (1k→4k), i.e. not KV-bound: the time is going somewhere else.

## rocprofv3 status (SOLVED 2026-09-15 — root cause was wrong)

- The SIGSEGV in `cfree` ~1 s after `HSA version 1.21.0 initialized` is NOT in
  the SDK's code-object interception. gdb shows the real fault: `dlopen` of
  `triton/_C/libtriton.so` runs LLVM static initializers
  (`llvm::DebugCounter`/`DenseMap`) that free a poisoned pointer — the
  LD_PRELOAD'd `librocprofiler-sdk.so` drags in `libLLVM.so.23` whose `llvm::`
  symbols interpose into triton's own LLVM (ABI mismatch → heap corruption).
  The absl `symbolize_elf.inc` backtrace was the handler crashing on the
  corrupted heap, not the cause.
- **Working recipe**: do NOT let rocprofv3 LD_PRELOAD the SDK. Load the tool
  through the register mechanism instead — either `HSA_TOOLS_LIB=` or:

  ```bash
  env ROCP_TOOL_LIBRARIES=/opt/rocm/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so \
      ROCPROFILER_LIBRARY_CTOR=1 \
      ROCPROF_OUTPUT_FORMAT=csv ROCPROF_OUTPUT_PATH=/out \
      ROCPROF_OUTPUT_FILE_NAME=decode \
      ROCPROF_KERNEL_TRACE=1 ROCPROF_TIME_FORMAT=nsec \
      python3 profile_rocm.py
  ```

  Counter collection: add `ROCPROF_COUNTER_COLLECTION=1` and
  `ROCPROF_COUNTERS="pmc: <names>"` (single pass only — multi-pass needs the
  frontend's per-group re-run; keep groups ≤ ~4 counters or
  `rocprofiler_create_counter_config` fails with error 38). gfx1101 counter
  names: `FETCH_SIZE` (KB fetched from VRAM), `SQ_WAVES`, `SQ_BUSY_CYCLES`,
  `SQ_WAIT_ANY`, `GRBM_GUI_ACTIVE`, `GL2C_EA_RDREQ_{32,64,128}B_sum`,
  `GL2C_HIT_sum`. `TCC_*` does not exist on gfx1101 (it's GL2C).
  HIP API trace: `ROCPROF_HIP_RUNTIME_API_TRACE=1`.
- Caveats observed: (a) counter collection serializes dispatches (~3×
  slowdown) — use it for ratios, not wall time; (b) one flaky GPU page fault
  seen during `model.load()` under the tool (2/4 runs) — latent race in the
  load path worth a coherency check, not a profiler artifact to ignore;
  (c) `Grid_Size_X` in the CSV is work-items, not blocks.

## Served-endpoint measurements (2026-09-15, port 8420, `serve_openai.py`)

End-to-end HTTP timing, greedy, cache 32768:

| scenario | result |
|---|---|
| 11-tok prompt, 128 decode | **17.0 t/s e2e** (7.5 s wall) |
| 4 concurrent × 96 tok | **21.8 t/s aggregate — batched** |

- e2e 17.0 t/s vs in-process 26.6 t/s → ~36% serving overhead
  (tokenize/template/queue/lock).
- **Fixed**: `_run_job` held `_lock` for the whole generation, serializing
  `gen.iterate()`. Now a dedicated driver thread owns `iterate()`; workers
  `enqueue`/`cancel` under `_gen_cv` and receive chunks via per-job queues.
  4 parallel requests batch instead of taking ~4× wall time.

## rocprofv3 decode hotspot table (2026-09-15, kernel-trace, 64 tokens)

Steady-state decode window: 9.58 s wall, 6.12 s device-busy → **36% device
idle** (profiler-inflated; unprofiled gap ~12–18%). 62k dispatches, ~970/token.

| kernel | total | share | calls | avg | measured fetch | GB/s |
|---|---|---|---|---|---|---|
| exl3_mgemm_kernel<1,true> (1 bpw) | 2335.9 ms | 38.2% | 1495 | 1.56 ms | 10.1 MB/call | **7** |
| exl3_gemv_int8_sq<1,1,f,f> | 789.2 ms | 12.9% | 7168 | 110 µs | 10.3 MB/call | 96 |
| exl3_mgemm_kernel<2,true> | 716.6 ms | 11.7% | 455 | 1.57 ms | 12.7 MB/call | **14** |
| exl3_gemv_int8_sq<2,1,t,f> | 443.3 ms | 7.2% | 6720 | 66 µs | 11.8 MB/call | 188 |
| exl3_gemv_int8_sq<1,1,t,f> | 319.7 ms | 5.2% | 3328 | 96 µs | — | ~97 |
| exl3_gemv_int8_sq<3,1,f,f> | 279.4 ms | 4.6% | 64 | 4.37 ms | 455 MB/call | 107 |
| exl3_gemv_int8_sq<2,1,f,f> | 250.5 ms | 4.1% | 2048 | 122 µs | — | ~188 |
| exl3_mgemm_kernel<3,false> | 172.4 ms | 2.8% | 845 | 204 µs | — | ~20 |
| exl3_gemm_kernel<1,false> | 144.8 ms | 2.4% | 112 | 1.29 ms | 10.4 MB/call | **8** |
| exl3_mgemm_kernel<3,true> | 101.4 ms | 1.7% | 65 | 1.56 ms | — | ~21 |
| GDN recurrent + gdn_ba_gemv | ~156 ms | 2.5% | 6240 | ~25 µs | — | fine |
| paged attn + norms + misc | ~130 ms | 2% | — | — | — | fine |

PMC counters (FETCH_SIZE = KB fetched from VRAM, single pass, counter
serialization inflates durations — ratios only):

- **gemm/mgemm kernels: 7–24 GB/s** — 30–60× under the 432 GB/s peak. These are
  `cudaLaunchCooperativeKernel` launches with `grid.sync()`; grid = 13824
  work-items / 512 = **27 blocks on 54 CUs** (half the GPU idle), and each
  `grid.sync()` serializes the surviving blocks. Latency-bound, not
  bandwidth-bound.
- **gemv_int8 kernels: 96–197 GB/s** — same launch style but 108 blocks
  (27648/256); ~2× more parallelism → ~20× more bandwidth. Still <50% of peak.
- Total weight fetch ≈ 4.9 GB/token → at a realistic 350 GB/s the whole decode
  GEMM/GEMV budget is ~14 ms/token (~70 t/s ceiling). Currently ~96 ms/token
  device-busy.
- `exl3_gemv_int8_sq<3,1,f,f>` (lm_head, 248k vocab): 4.37 ms/call at
  107 GB/s — biggest single kernel; grid 6912/256 = 27 blocks, under-parallel.
- Host side (API trace): `hipDeviceSynchronize` ~1/token (~6 ms each, sampling
  readback), `hipGraphLaunch` ~130/token at ~100 µs each (profiler-inflated),
  and 41 one-time graph captures during the first ~4 tokens (GDN BC slots —
  expected, not a leak).

## Optimization plan (updated 2026-09-15 with rocprofv3 counters, priority order)

### P0 — serving concurrency — **DONE** (2026-09-16)

Driver thread owns `gen.iterate()`; workers enqueue/cancel under `_gen_cv`
and receive chunks via per-job queues. 4-way aggregate 8 → 21.8 t/s.
Required msq m>1 support (jj-flattened units) since batched decode hits
mul1 mgemm at m=2..4 — now bit-exact vs the per-matrix int8 path.

### P1 — decode GEMM/GEMV at DRAM bandwidth (target 12 → 40–70 t/s)

rocprofv3 counters: gemm/mgemm run at **7–24 GB/s**, gemv_int8 at 96–197 GB/s,
vs 432 GB/s peak. The gemm/mgemm kernels launch cooperatively with only
**27 blocks on 54 CUs** (grid 13824 work-items / 512) and serialize on
`grid.sync()`; gemv_int8 gets 108 blocks. Both are latency-bound.

  a. ~~Split the two-stage `grid.sync()` kernels into non-cooperative launches~~
     **DONE for the dominant case (2026-09-15/16)**: `exl3_gemv_int8_msq_kernel`
     covers every `mul1` mgemm call at any m (sliced SlicedMultiLinear bundles
     and plain multi-matrix projections) in one regular launch — (matrix,row)
     flattened into the unit index, atomic work counter, per-(jj,slice) staged
     splats, per-(jj,256-column) completion counters gating a deterministic
     fixed-order epilogue. Same argument list as `exl3_mgemm_kernel` →
     identical graph recording. Verified: bit-exact vs the per-matrix int8
     path at m=1 and m=4, deterministic across replays.
     Result: 10.5 → 22.6 t/s (m=1); batched serving now engages.
     Remaining coop kernels: `exl3_gemm_kernel` (single-matrix mul1 at K>5,
     and all cb=0/cb=1 tensors) and `exl3_mgemm_kernel` (filtered/weighted,
     bszm_in>1).
  b. ~~K=1 coverage~~ **DONE** — msq covers K=1..8 for mul1 mgemm; the 1 bpw
     `mgemm<1>` line (38% of device time) is gone from the profile.
  c. ~~`exl3_gemv_int8_sq<3,1,f,f>` (lm_head): 4.37 ms/call, 455 MB at 107 GB/s,
     only 27 blocks~~ **DONE** — `rows_per` override + gfx1101 default of 64
     (was 512 → 1 block/CU). Swept 32/48/64/96/128/256; 64 wins e2e.
  d. Re-tune int8-GEMV ksplit × grid for gfx1101 once uncapped.

### P2 — host overhead (~12–18% unprofiled; 36% under rocprofv3)

  e. Per-token `hipDeviceSynchronize` in the sampler readback
     (`generator.py:1114`) costs ~6 ms/token under the tool — check whether the
     pinned-buffer staging can overlap the next token's launches instead of a
     full device sync.
  f. After P1, re-measure host gap; verify graph capture covers any new
     non-cooperative launches (they record params the same way — check
     `add_graph_args` sites).
  g. Serving path: tokenize/detokenize and template rendering are on the
     request thread — fine once P0 lands; re-check.

### P3 — prefill

  h. `rocm-optimize` WMMA M≥3 NaN on RDNA3: bisect/fix or gate by arch.
  i. GDN prefill fla Triton kernels on gfx1101: benchmark vs HIP.

### P4 — throughput ceiling

  j. Speculative decoding with an external ~4B EXL3 draft (fits in ~3 GB
     headroom at ≤8k ctx).
  k. Re-measure batch-8 after P0+P1.

## Correctness & coherency checks (gate every P0–P4 change)

1. **Golden-output parity** — before/after each change, greedy-decode a fixed
   prompt set (≥8 prompts, 256 tokens, incl. one >4k-ctx prompt) through the
   HTTP endpoint and through `bench_rocm.py`; token IDs must match the
   pre-change baseline exactly (greedy → deterministic). Any divergence = bug.
2. **Unit tests** — `pytest tests/` must pass; specifically `test_qgemm.py`,
   `test_quant_fn.py`, `test_gated_delta_rule.py`, `test_cache_rotate.py`,
   `test_triton_paged_*.py` cover the touched paths.
3. **Numerical spot-check** — for kernel changes: `eval/prequant_test.py` /
   qbench KLD on a fixed text set; KLD vs pre-change logits < 1e-3.
4. **Concurrency coherency** (P0) — N parallel requests must produce identical
   outputs to N sequential requests with the same prompts+seeds; KV pages must
   be released (check `cache` free count / VRAM returns to baseline after
   drain); client-disconnect mid-stream must cancel the job without poisoning
   the next request.
5. **Graph-capture integrity** — after kernel changes, confirm
   `hipGraphLaunch` count per token is unchanged (torch.profiler) and no
   capture-time `cudaStreamIsCapturing` failures; run one session with
   graphs disabled (`EXL3_NO_CUDA_GRAPH` or equivalent) and diff outputs.
6. **Deadlock regression** — any cooperative-launch change: 10-min soak of
   mixed-length requests; the RDNA co-residency deadlock hangs the process,
   so a hang = failure.
7. **Perf gate** — record decode t/s (b1 short), prefill 2k t/s, batch-8
   aggregate in the results log below; a change that doesn't measurably help
   gets reverted.
8. **Load-path race** — a flaky GPU page fault ("Page not present") was seen
   during `model.load()` under rocprofv3 (2/4 runs, ~90 s in). Latent
   host/device race in the deferred-tensor or staging path; any load-path
   change must run ≥5 profiled loads without a fault, and the fault should be
   root-caused before shipping P1 (it may share the async-copy machinery the
   new kernels will use).
9. **rocprofv3 verification** — after each kernel change, re-run the
   kernel-trace + `FETCH_SIZE` counter pass and confirm achieved GB/s per
   kernel class moved toward peak; device-idle % should drop as host stalls
   are removed.

## Results log (to be appended per iteration)

| step | decode t/s (b1, short) | prefill 2k t/s | notes |
|---|---|---|---|
| baseline rocm-port (EXL3_INT8_GEMV default=2) | 6.0 | 431 | |
| EXL3_INT8_GEMV=0 (QTIP GEMV engaged) | 1.7 | — | GEMV worse on RDNA; grid-starved |
| rocm-port + fixes (carveout, atomics, soname) | 6.1 | 430 | correctness pass; no perf delta expected |
| + coop autotuner at concurrency=1 + hw dp4a | 11.9 | 429 | batch8 16.6 → 20.3 t/s (warmed); concurrency>1 still deadlocks on RDNA |
| served endpoint e2e (serve_openai.py, 2026-09-15) | 9.8 | — | HTTP overhead ~18%; 4-way concurrent ≈ 8 t/s aggregate (serialized by _lock) |
| rocprofv3 kernel-trace (2026-09-15, ROCP_TOOL_LIBRARIES recipe) | 6.67 | — | 36% device idle; gemm/mgemm 7–24 GB/s, gemv_int8 96–197 GB/s (FETCH_SIZE counters) |
| + msq kernel (multi-matrix/sliced sq variant, all mul1 m=1 mgemm) | **22.6** | 434 | batch8 20.7 t/s; A/B on same image: 10.5 → 22.6 t/s; `EXL3_INT8_MSQ=0` reverts |
| + rows_per=64 default (sq+msq, gfx1101 sweep: 64 > 32/48/96/128/256/auto) | **26.6** | 436 | batch8 20.7 t/s; msq now bit-exact vs per-matrix int8 path; `EXL3_SQ_ROWS_PER` overrides |
| + generator driver thread (serve_rocm/serve_openai) + msq m>1 (jj-flattened units) + sq m<=4 | 17.0 e2e | — | 4-way concurrent 21.8 t/s aggregate (was ~8 serialized); msq m=4 bit-exact vs per-matrix int8; SQ_COUNTERS_CAP 4096->65536, rows_max 48KB budget on ROCm |
| + async GDN stash/slots + hgemm_recon fp16-out via torch mm + fp32-out via hipBLASLt (timed algo sweep) + serve warmup | 26.6 | **752** (bench) / 604 (fresh-prompt A/B) | prefill 512: 243 (was 204), ctx4k: 609 (was 345); batch8 22.6; VRAM 10.03 GB (+0.15 Lt workspace); gates: golden 7/8 bit-identical + 1 deterministic near-tie transposition, test_msq_ab bit-exact, serving soak PASS |

## Iteration 2 (2026-09-17): leftover decode coop-GEMM + prefill reconstruct path

Baseline re-confirmed on the `gfx1101-msq7` image (tree = 28f29ee): **decode 26.7 t/s**
(b1 short), prefill 438 t/s @2k (204 @512 / 178 @1k / 345 @4k — the sub-2k numbers are
first-use triton JIT compile artifacts, not steady state), batch8 22.6 t/s.
`prefill_4096` in bench_rocm.py is a prompt-cache hit (3927 t/s) — ignore.

Environment finding: the `gfx1101-msq7` image baked a **prebuilt `exllamav3_ext.so` into
site-packages**, which shadows the JIT path — source edits in the bind mount were silently
ignored by every run on that image. `is_precompiled_extension_available()` (exllamav3/ext.py)
finds the stale `.so` first. All 2026-09-17 work uses the new `local/exl3-rocm:gfx1101-base`
image (deps only, current Dockerfile.rocm10), where the extension is always JIT-built into
the `exl3_exl3-cache` volume from the mounted tree.

### Fresh decode profile (torch.profiler, 6 steady-state tokens, no profiler inflation)

~62% of decode device time is still the **cooperative `exl3_gemm_kernel`** — ~50 calls/token
across `<1,f>` `<1,t>` `<2,f>` `<2,t>` `<3,t>` (cb=2 mul1, 1.0–1.3 ms each). The int8 sq path
(hundreds of healthy 60–110 µs calls) covers the rest; GDN recurrent + paged attention remain
fine. These coop calls are the m=1 single-matrix linears (out_proj / o_proj / down_proj /
small-attention projections) that fail an sq/msq gate and fall through to the grid-starved
cooperative kernel. EXL3_DBG_GEMV=1 instrumentation (exl3_gemv_int8.cu) logs every
fall-through reason; census run identifies the exact gates.

Known gate suspects: `size_n % 256` rejects n=1152 tensors (108 in this model: the small
q/k/v/proj/fc2 projections), and the sq kernel map may lack (K, M, c_fp32) instances for
some single-matrix cases that msq already covers.

### Fresh prefill profile (rocprofv3 kernel trace, 2048-token prefill)

64% of prefill device time is **hipBLAS (`Cijk_*` rocBLAS kernels)**, not exl3 kernels:
at m > AUTO_RECONSTRUCT_THRESHOLD (144), `LinearEXL3.forward` takes `reconstruct_hgemm` —
dequantize the tensor (`reconstruct_had_slice`) then run hipBLAS on the fp16 weights,
re-done **every call** (235+211 HSS calls at 2.98 ms avg + HHS MT128x128x16 with MI16x16x16
ops at 1.06 ms). The exl3 GEMM kernel would re-dequantize B tiles once per 16-row m-chunk
(128x redundant at m=2048), so upstream deliberately routes large-m to reconstruct+GEMM.
The lever is the GEMM backend, not the exl3 kernel: try hipBLASLt (TORCH_BLAS_PREFER_HIPBLASLT),
rocBLAS arch tuning, and (longer term) hardware-WMMA in the exl3 GEMM path (EXL3_WMMA=1,
known NaN at M>=3 to bisect) as a hipBLAS alternative.

### Iteration-2 plan (priority order)

1. **P1-d decode: close the int8-path gate gaps.** From the EXL3_DBG_GEMV census, extend
   sq/msq coverage (n%256 handling or route n%256!=0 single matrices through msq, add
   missing sq instances, relax max_k for RDNA3 where msq accepts any K). Verify bit-exact
   vs the coop path per case (test_msq_ab.py pattern), golden parity, deadlock-free.
   Target: 26.7 → 35–45 t/s decode.
2. **P3 prefill: GEMM backend experiments** (env-only, low risk): TORCH_BLAS_PREFER_HIPBLASLT=1,
   then measure prefill 512/2k/4k + decode (hipBLASLt affects decode sampler GEMMs too if any).
   Keep whichever wins; document the env in serve_rocm.py.
3. **P3 prefill: warm the triton JIT at startup** in serve_rocm.py (dummy prefill at a few
   seqlens on a throwaway cache) so the first real request doesn't pay 2–4 s of compile.
4. **P2 host gap re-measure** after 1: sampler readback sync is already batched to one
   torch.cuda.synchronize per iterate (generator.py); quantify the remaining gap and only
   then consider overlapping.
5. **P4 (stretch): WMMA bisect** for the exl3 GEMM path as a reconstruct+hipBLAS alternative
   at large m, and/or TILESIZE_M>16 prefill shapes to amortize B dequant.
6. **Correctness gates** (all changes): pytest suite in-container, EXL3_DBG_GEMV census must
   show zero unexpected misses, golden_rocm.py --check parity vs the saved baseline
   (.profiling/golden_baseline.json), test_msq_ab.py bit-exactness for any new kernel
   routing, batch/sequential coherency, and the perf gate (revert non-winners).

## Iteration 2 results (2026-09-17)

### What moved the numbers

| change | decode b1 | prefill 2k (fresh) | notes |
|---|---|---|---|
| baseline (28f29ee via msq7 image) | 26.7 | 283 t/s | prefill_ab.py, unique prompts, warm process |
| + async GDN state stash + host-side default slots | 26.7 | 283 t/s | removed 102 blocking D2H drains; wall unchanged at this point — see below |
| + hgemm_recon fp16-out via torch mm (hipBLASLt) | 26.7 | ~430 t/s | 348 calls: 41.2 → 3.4 ms each (2048x17408x5120) |
| + hgemm_recon fp32-out via hipBLASLt + timed algo sweep | 26.7 | **604 t/s** | 446 calls: 11.8 → 2.6 ms each; sweep beats the gfx1101 heuristic 3–8x |
| bench_rocm.py (prefix-cache-assisted) | 26.6 | **896 t/s** (was 438) | ctx4k 651 (was 345), 512: 306 (was 204) |

- The prefill D2H finds: `GDNLayerState.stash` ran a blocking `.cpu()` per GDN layer per
  chunked forward (102 hipMemcpyWithStream, 5.7 s of the 7.5 s wall under profiler). Fixed
  with pinned-destination non_blocking copies (stream-ordered, safe against later in-place
  state overwrites). Also `gated_delta_rule.py` built a device arange only to `.tolist()` it
  (one more sync per layer) — replaced with a plain `range(bsz)`.
- The real prefill cost was the GEMM backend: `cublasGemmEx` (hipBLAS→rocBLAS) picks a
  4.4 TF/s kernel for both fp16-out (41.2 ms for 2048x17408x5120) and fp16-in/fp32-out
  (11.8 ms) large-M shapes, while the hipBLASLt path torch.mm uses runs the same shapes at
  3.4 ms. `hgemm_recon` now routes fp16-out through torch mm and fp32-out through a direct
  hipBLASLt call (`hgemm_lt_f32out` in hgemm.cu) with a per-shape cached algo. The gfx1101
  hipBLASLt heuristic's top pick is frequently 3–8x slower than another candidate, so the
  algo is chosen by a one-time timed sweep (m >= 256, capped at 32 sweeps; small-m and
  over-budget shapes take the heuristic's pick so arbitrary prompt remainders can't trigger
  unbounded sweeps). TORCH_BLAS_PREFER_HIPBLASLT makes no difference for torch.mm (already
  hipBLASLt-backed); hipBLASLt handles 16F->32F that torch.mm cannot express.
- Decode (TG): the int8-path census (EXL3_DBG_GEMV=1) shows **zero m=1 gate misses** — all
  ~523 sq + 36 msq calls/token engage; the earlier "62% coop gemm" read was the 11-token
  prefill phase captured in the same profile, not decode. Decode is now 87% sq/msq int8
  kernel time at ~50–100 GB/s effective (432 GB/s part). EXL3_SQ_ROWS_PER re-swept
  (32/48/64/96/128): 64 confirmed optimal. TG kernel efficiency is the next major project;
  no cheap dispatch-level win remains.
- serve_rocm.py: startup warmup (512-token prefill + 8 decode) pays the triton JIT compiles
  and the algo sweeps before the server reports ready.

### Correctness gates

1. **Golden parity** (golden_rocm.py, 8 prompts x 96 greedy tokens, baseline recaptured on
   pristine 28f29ee via git worktree + msq7 image): 7/8 prompts bit-identical. Prompt 6
   diverges at generated token 86 as a two-token transposition ("sentence repeated" ↔
   "repeated sentence") — an argmax near-tie flipped by the GEMM reduction-order change,
   deterministic across repeats (run-to-run identical), text otherwise identical.
2. **pytest**: 431 passed, 1 failed, then a GPU page fault aborted the rest.
   The failure (`test_gated_delta_rule.py[2-1024-...-True]`) is pre-existing: the pristine
   baseline times out (>300 s) on the same param. The page fault matches the documented
   flaky fault (see gate #8 below). Suite rerun after a clean rebuild pending.
3. **Known build hazard found**: torch's in-tree hipify + ninja left a stale hgemm .hip
   after cross-container edits, producing a .so with an undefined `hgemm_lt_f32out`
   (failed only at import-time symbol resolution; decode paths ran fine). Fixed by a clean
   rebuild (`rm -rf $TORCH_EXTENSIONS_BUILD/exllamav3_ext` + stale *.hip). Any C++ edit
   should either bump the version hash expectedly or rebuild clean.
4. Serving soak (batch determinism, mixed-length concurrency, mid-generation disconnect):
   pending on the final build.

## Static decode-path analysis (rocm-port)

Per token, batch 1, ~64 layers → roughly 350–450 kernel launches. CUDA-graph
capture IS active in this generator (~138 `hipGraphLaunch` per token; only
DSV4/MLA special modules capture their own graphs; the CPU launch gap is
~12%, so launches are not the bottleneck).

Key dispatch facts found in code (`exllamav3/exllamav3_ext/quant/`):

1. **1 bpw tensors never take the fast GEMV path.**
   `exl3_gemv_try_launch`: `if (K < 2 || K > 4) return false;`
   This model: 210 tensors at 1 bpw (1.9 GB = ~40% of layer bytes) fall back
   to the generic cooperative GEMM kernel at M=1 (grid capped at
   `(occupancy-1)×SMs` on ROCm). The GEMV kernel supports 2/3/4 bpw only
   (`static_assert`), and the kernel map has no K=1 instantiation.
2. GEMV path requirements otherwise satisfied: model codebook is `mul1`
   (cb=2) so K∈{2,3} pass `K != 4 && cb == 0` check; sizes are multiples
   of 128.
3. There is an experimental `EXL3_INT8_GEMV=1` fused int8-activation GEMV
   for mul1 tensors (not graph-capturable, successive launches, m≤144).
4. GDN (linear-attention) layers: decode uses custom
   `cuda_recurrent_gated_delta_rule` HIP kernel; prefill uses vendored fla
   Triton chunk kernels (works via triton-rocm; unverified perf).
5. Full-attention layers (16 of 64): Triton paged attention path
   (`triton_paged.py`, bc_attn) on ROCm — JIT-compiled per shape at first use
   (explains the first-run 2.5 s "prefill" for 6 tokens).
6. `rocm-optimize` branch (WIP): WMMA/GEMM improvements aimed at prefill
   (M≥3) GEMM path; known NaN issue at M≥3 on RDNA3 left unresolved; also
   fixes hipFuncSetAttribute carveout crash and WMMA operand layouts.
   Decode (M=1) mostly unaffected by that WIP.

## Measured decode hotspots (torch.profiler fallback, 6 steady-state tokens)

Wall 1.49 s → **237 ms/token of device time**, host gap only **12%** — the GPU is
busy; kernels are slow, not the launch loop.

| kernel | total | calls | avg | est. GB/s |
|---|---|---|---|---|
| exl3_mgemm_kernel<1,…> (1 bpw) | 480.7 ms (33.8%) | 161 (27/tok) | 2.985 ms | ~4 |
| exl3_mgemm_kernel<2,…> | 153.6 ms (10.8%) | 49 | 3.135 ms | ~4 |
| exl3_gemm_kernel<1,false,…> | 146.7 ms (10.3%) | 112 | 1.309 ms | ~8 |
| exl3_gemv_int8_sq_kernel<1,1,f,f> | 141.3 ms (9.9%) | 672 (112/tok) | 210 µs | ~53 |
| exl3_gemm_kernel<2,true> | 95.0 ms (6.7%) | 104 | 913 µs | ~10 |
| exl3_gemv_int8_sq_kernel<2,1,t,f> | 76.7 ms (5.4%) | 630 | 122 µs | ~75 |
| exl3_gemm_kernel<1,true> | 64.3 ms (4.5%) | 52 | 1.236 ms | ~8 |
| exl3_gemv_int8_sq_kernel<1,1,t,f> | 54.4 ms (3.8%) | 312 | 174 µs | ~64 |
| everything else (GDN 13.6ms, paged attn 4.3ms, norms ~7ms) | ≈ 30 ms | | | fine |

(GB/s estimated from each kernel's B-matrix bytes at known layer shapes.)

Findings:

- **~75% of decode device time is quant GEMM/GEMV kernels running at 4–75 GB/s**
  against a 432 GB/s part. Per-call times of 1–3 ms for 5–11 MB streams imply a
  tiny effective grid (cooperative co-residency cap on RDNA: grid ≤
  (occupancy−1)×54 SMs, occupancy misreported) — the kernels are latency-bound
  on ~100 blocks instead of bandwidth-bound on thousands.
- The QTIP-style `exl3_gemv_kernel` (the kernel designed for exactly this,
  m=1, 2/3/4 bpw) **never runs**. Two causes found in code:
  - `EXL3_INT8_GEMV` **defaults to 2** (`exl3_gemv_int8.cu`:
    `_exl3_gemv_int8_mode = e ? atoi(e) : 2`), and the int8 path preempts the
    GEMV try_launch for mul1 tensors (all of this model).
  - The GEMV heuristic rejects K=1 outright (`K < 2 → false`), so 1 bpw goes
    to gemm/mgemm even with int8 mode disabled for it.
- `hipGraphLaunch` is used (≈138 launches/token) — graph capture exists in the
  generator; CPU-side launch cost is hidden (12% gap), so graphs are already
  doing their job.
- GDN recurrent kernel (40 µs × 56/tok) and paged attention (32 µs × 19/tok)
  are healthy — linear-attention layers are NOT the problem.
- First-run JIT: triton paged-attn compile lands in the first prefill
  (~2.5 s once per process), irrelevant steady-state.

## Reproducing

```bash
docker build -f Dockerfile.rocm10 -t local/exl3-rocm:gfx1101 .

# In-process benchmark + kernel hotspots (rocprofv3 is broken on this app —
# see above; hotspot_rocm.py uses torch.profiler → same kernel-level data):
docker run --rm --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --ulimit nofile=1048576:1048576 \
  -v /home/ghazni/github/exllamav3-rocm:/src:ro \
  -v /home/ghazni/models/exl3/turboderp/Qwen3.8-27B-EXL3-SC_1.40bpw_H3_V3:/models/qwen38-27b:ro \
  -v exl3_exl3-cache:/root/.cache \
  local/exl3-rocm:gfx1101-unified \
  python3 /src/hotspot_rocm.py        # or /src/bench_rocm.py

# Served endpoint (port 8420):
docker run -d --name qwen38-27b-exl3 --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render --ulimit nofile=1048576:1048576 \
  -p 8420:8420 -e PORT=8420 -e EXL3_MODEL=/models/qwen38-27b \
  -e EXL3_MODEL_NAME=Qwen3.8-27B-EXL3-SC_1.40bpw_H3_V3 -e EXL3_CACHE_TOKENS=32768 \
  -v /home/ghazni/github/exllamav3-rocm/serve_openai.py:/opt/exllamav3/serve_openai.py:ro \
  -v /home/ghazni/models/exl3/turboderp/Qwen3.8-27B-EXL3-SC_1.40bpw_H3_V3:/models/qwen38-27b:ro \
  -v exl3_exl3-cache:/root/.cache \
  local/exl3-rocm:gfx1101-unified python3 /opt/exllamav3/serve_openai.py
curl -s http://localhost:8420/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-27B-EXL3-SC_1.40bpw_H3_V3","prompt":"...","max_tokens":128,"temperature":0}'
```
