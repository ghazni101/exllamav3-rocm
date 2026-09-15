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

## rocprofv3 status (blocking bug, documented)

- Trivial torch app: `rocprofv3 --kernel-trace` works (CSV produced).
- exllamav3 process: SIGSEGV inside rocprofv3 immediately after
  `HSA version 1.21.0 initialized`, in `cfree` via `symbolize_elf.inc`
  (crash handler then fails: "Unable to get high fd", limit=1024).
- Reproduces with import of `exllamav3_ext` alone, with `--mangled-kernels`
  (-M, no demangle) and with `--hip-runtime-trace` → not a demangling issue;
  it is the v3 tool's library-load interception crashing on
  `exllamav3_ext*.so` (very large templated binary: 163 TUs, hundreds of
  kernel instantiations).
- Fixes attempted: ulimit nofile raised at container level (required but not
  sufficient), `-M`, api-only trace. Remaining candidates (not yet tried):
  `rocprof-attach` with a custom rocprofiler-sdk tool library, `-f rocpd`
  streaming output, profiling a stripped/minimal extension build, or
  upstream rocprofv3.
- Workaround used for data: `hotspot_rocm.py` (in-process torch.profiler /
  Kineto→roctracer) captures the same per-kernel device times + counts.

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

## Optimization plan (priority order)

P0 — (done) hotspot table above; next: rocprofv3 PMC on the top kernels once
the v3 crash is worked around, to confirm achieved-bandwidth diagnosis.

P1 — make decode hit DRAM bandwidth (target: 6 → 40–70 t/s). All changes are
kernel-selection and grid-shape work; no algorithm changes needed.
  a. MEASURED: `EXL3_INT8_GEMV=0` (QTIP GEMV engaged for 2/3 bpw) → **1.7 t/s,
     3.5× WORSE than default**. The QTIP GEMV is unusable on RDNA as-built, and
     every M=1 path is slow in the same direction. Primary suspect for all
     paths: the ROCm cooperative co-residency cap (deadlock workaround in
     `exl3_gemv.cu`/`gemm`) shrinks grids to ~(occupancy−1)×54 blocks; M=1
     kernels end up latency-bound on ~100 blocks instead of streaming with
     thousands. Fix = grid/occupancy work, not kernel replacement:
       - raise/eliminate the grid cap for kernels that don't grid.sync()
         per k-tile, or split the two-stage coop kernels into
         non-cooperative launches with atomic-counter chaining;
       - then tune GEMV/int8-GEMV k-split × n-tile for gfx1101 (54 CU),
         verifying with PMC (TCC sector throughput, SQ_WAVES).
  b. Extend decode coverage for K=1 (currently int8-GEMV handles it at
     ~210 µs/call ≈ 53 GB/s; the QTIP path rejects K=1 outright). After the
     grid fix, re-measure which kernel wins per (K, shape).

P2 — host overhead (12% gap; do after P1 changes the mix):
  e. Verify graph capture still covers new paths (int8 GEMV is documented
     as not graph-capturable — it currently coexists with graphs; check
     where the 12% goes once kernels are fast).

P3 — prefill (long prompts / batch serving):
  f. `rocm-optimize` WMMA M≥3 GEMM NaN: bisect with that branch's scripts,
     fix or gate behind arch check, re-bench prefill (expect 2–5×).
  g. GDN prefill fla Triton kernels on gfx1101: benchmark, consider HIP.

P4 — serving-level throughput:
  h. Speculative decoding: no MTP tensors in this checkpoint — external
     Qwen3.5-4B EXL3 draft (~2.3 GB) fits in the remaining ~3 GB if
     context ≤ 8k; measure acceptance-weighted gain at batch 1.
  i. Re-measure batch-8 aggregate (17.8 t/s now) after P1; M=8 GEMM should
     amortize far better than 8× M=1.

## Results log (to be appended per iteration)

| step | decode t/s (b1, short) | prefill 2k t/s | notes |
|---|---|---|---|
| baseline rocm-port (EXL3_INT8_GEMV default=2) | 6.0 | 431 | |
| EXL3_INT8_GEMV=0 (QTIP GEMV engaged) | 1.7 | — | GEMV worse on RDNA; grid-starved |

## Reproducing

```bash
docker build -f Dockerfile.rocm10 -t local/exl3-rocm:gfx1101 .
docker run --rm -it --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render --cap-add=PERFMON \
  --ulimit nofile=1048576:1048576 \
  -v /home/ghazni/models/exl3/turboderp/Qwen3.8-27B-EXL3-SC_1.40bpw_H3_V3:/models/qwen38-27b:ro \
  local/exl3-rocm:gfx1101-v1 \
  python3 /opt/exllamav3/bench_rocm.py
# hotspots: python3 hotspot_rocm.py   (rocprofv3 crashes on this app, see above)
```
