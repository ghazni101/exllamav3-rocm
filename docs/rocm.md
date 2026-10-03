# ROCm port notes (RDNA3 / gfx11, wave32)

Scope: gfx110x (RDNA3). Only wave32 parts are supported — `arch_list.py` /
`exl3_devctx` reject wave64 devices (CDNA) at import.

## What the port consists of

- `exllamav3_ext/hip_compat.cuh` — shims hipify doesn't translate: widened warp-sync
  masks, `__dp4a` via `__builtin_amdgcn_sudot4`, scoped atomics, fragment types.
- `ptx.cuh` — m16n8k16 MMA emulated with `__shfl`-based math on ROCm; `cp_async`,
  `lop3`, and PTX barriers replaced with portable equivalents.
- `cuda_drv.*`, `triton_kernel.*`, `graph.*` — HIP module/graph APIs standing in
  for the CUDA driver API (Triton kernels load hsaco instead of cubin; warp size
  is read from the device).
- `exl3_devctx.cu` — `CC_RDNA3` device class via `gcnArchName`.
- `quantize.cu`, `exl3_kernel_map.cu`, `exl3_gemm_inner.cuh` — 64 KB dynamic
  shared-memory opt-in limit: oversized kernel shapes are not instantiated and
  are skipped at dispatch.
- `coop_autotune.cu`, `exl3_gemv.cu` — RDNA over-reports cooperative-launch
  co-residency; concurrency is clamped to 1 and occupancy is reduced by one
  block/SM to avoid `grid.sync()` deadlocks.
- `cpu/moe_handoff.cu` — `hipStreamWriteValue32`/`WaitValue32` memops for the CPU
  MoE offload handshake.
- Python: pinned/non-blocking H2D staging in the generator (`job.py`,
  `pagetable.py`, `SeqTensor.pin`, `Embedding` double-buffered staging, `stash()`
  paths) — pageable copies serialize the stream and stall the host ~10 ms per
  chunk boundary.

## Fractional (half-integer) trellis bitrates

`port/rocm-frac-trellis` merges upstream dev's fractional-trellis support
(half-integer bitrates `K = N + 0.5`, `mul1` codebook only: positions alternate
N / N+1-bit steps) onto the port. Dedicated instances exist for **1.5 / 2.5 / 3.5
bpw** (`exl3_comp_unit_h{,1..3}.cu`, `quantize_tiles_frac_inst.cu`,
`exl3_gemv_half_inst.cu`, `exl3_moe{,_coop}_inst_h{,1..3}*.cu`); other half rates
raise the extension's explicit "No kernel for half-integer GEMM bitrate" check.

Verified on gfx1100 (2026-10-03): half-rate `exl3_gemm` agrees with a
`reconstruct_had_slice`-based dequantize-then-matmul reference at ~1.2e-3 relative
RMS for K = 1.5 / 2.5 / 3.5 (same as the integer rates), and `quantize_tiles_frac`
reproduces the expected rate/error curve (tile MSE 0.053 / 0.014 / 0.0034).

**The int8-activation GEMV path carries half rates.** On ROCm the port's fast decode
path is fused int8 GEMV (`EXL3_INT8_GEMV`, default 2). Its `sq` and `coop` kernels are
now templated on `HALF` as well (`exl3_gemv_int8_kernel.cuh`: the two-group extraction of
`dq8_half`, `gemv_int8_twords<bits,HALF>` = `16 * K + 8` uint16 per tile, `ext8w_half`),
with dedicated `_h1/_h2/_h3` instances for 1.5 / 2.5 / 3.5 bpw selected inside
`exl3_gemv_int8.cu`. The port's `sq` grid-multiplier / slice-height tuning and the `msq`
sliced variant are kept; `msq` has no half instances, so half-rate tensors take the
`sq`/`coop` path.

Measured on gfx1100 (2026-10-03, OrcaSAQ-2-27B-EXL3-3.21bpw — 409 tensors, 120 of them
at 3.5 bpw, Q8 KV):

- decode 128 tokens greedy: **8.5 → 34.8 tok/s** with the half-rate int8 path in place
  (1.7 tok/s with `EXL3_INT8_GEMV=0`, i.e. the fp16 path only — the int8 GEMV is the
  whole ballgame on this port);
- int8 half-rate output vs the fp16 reconstruct reference: ~0.8% relative RMS for
  K = 1.5 / 2.5 / 3.5 at m = 1 / 2 / 4 — the same deviation as the integer rates
  (0.12% with the int8 path off).

Dev-only kernels that do not hipify are intentionally not part of this branch
(`dflash2`, `det_gemm`/`hc_mix_tiled`/`routing_gemm`); the corresponding
architectures/bindings are not registered.

## Performance changes (all measured on gfx1100, Qwen3.8-27B-EXL3-3.5bpw)

Decode baseline → final: ~22 → ~41.7 tok/s (4096 ctx / 256 gen, greedy).

Kept:
- `msq` int8 GEMV path: one regular launch per MGEMM call (K ≤ 8 incl.), replaces
  the cooperative mgemm kernel that underfills RDNA. Biggest single win.
- int8 GEMV max K = 6 on RDNA3 (Hopper/Blackwell rule extended): +38% e2e.
- `-mcumode` build (`EXL3_CUMODE=1`, default in `Dockerfile.rocm`): each GEMV
  block scheduled on one CU; pairs with `EXL3_SQ_GRID_MULT=2` and
  `EXL3_SQ_ROWS_PER=40` compile-time defaults (+7.6% combined).
- DPP/xor-swizzle Hadamard (`had_xmask`) and `exl3_row_ror1` rotate in the sq
  kernels (DPP stays on the VALU instead of ds_bpermute): +1.9%/+1.2%.
- BC decode attention (`EXL3_BC_ATTN`) defaults **off** on ROCm — the captured
  block replays ~3.5% slower than eager dispatch on gfx1100.
- `EXL3_HGEMM_F16OUT` on by default on ROCm: hipBLAS fp32-output GEMM is ~5x
  slower than fp16-out + widen on RDNA3; +60-73% on 2k prefill.
- GDN: `ba` GEMV vectorized (10 ms → 0.2 ms/token worth), chunk-kernel gate
  lowered to `EXL3_GDN_CHUNK_MIN=8`, BC-graph q_len capped at 8.

Measured and removed (rejects): whole-step CUDA graph (`EXL3_STEP_GRAPH`),
hardware WMMA MMA (`EXL3_WMMA`, NaN bug on M≥3), nontemporal/buffer K=4 loads
(`EXL3_SQ_STREAM`, `EXL3_SQ_BUFFER_LOAD`), `EXL3_SQ_STAGE_SMEM` /
`EXL3_SQ_ROWS_PER_NARROWN` overrides, attention splits/warps knobs, deeper GEMV
row pipelines.

## Batch-size-dependent decode numerics

Decode output is **not bitwise identical across batch sizes** on this port, by
design of the dispatch:

- `m <= 4` rows take the int8 GEMV path (per-row quantized activations,
  ~0.8% RMS quantization error) — the decode engine.
- `m >= 5` rows take the fp16 reconstruct + GEMM path (~0.06% error) — the
  int8 msq kernels decline above 4 rows on RDNA3.

A near-tie logit can therefore flip one greedy token between an m=1 and an m=8
run of the same prompt and fork the continuation; both branches remain
individually coherent text (observed on prompt batch_c: "Blackwood Island" vs
"Blackwood Point"). The fork position varies with the GEMM autotuner's
per-session config choice (one-token flip in some sessions, a story-level fork
in others — the autotuner times candidates at first use, so the winning config
depends on GPU conditions at process start). CUDA has the same sq/msq split
with different crossovers, so this is not ROCm-specific — but exact
batch-invariance of greedy output is **not a contract** here. The `batch` gate
encodes the policy instead: equal lengths, first fork no earlier than
`GATE_BATCH_MINSTEP` generated tokens (default 4; earlier means prefill or
first-step numerics differ materially). Totals beyond the fork are reported
but not gated - two coherent continuations drift in and out of coincidental
token agreement; the numeric gap itself is numcheck's job.
`GATE_BATCH_STRICT=1` restores
exact equality for A/B runs whose numerics should be batch-independent.
Within a dispatch family, outputs are batch-size *tolerant*, not bitwise
identical (measured 2026-10-03, qwen38-27b 4.0bpw build git-584dd44f: the
int8 family's lm_head row output differs between the m=1/2 and m=3/4
sub-paths by up to 0.02 abs — consistent with sub-path quantization
granularity — and the fp16 reconstruct family differs above m~512 by up to
0.004 abs, the hipBLAS tiling changing with m; an earlier claim of bitwise
batch-independence for m=5..2048 did not reproduce). Same-m reruns are
bitwise deterministic. `fuzz_exl3_adversarial.py`'s per-family row
consistency check (0.05) is the enforcement of this tolerance.

Prefill numerics additionally differ from exact fp32 GEMM output when
`EXL3_HGEMM_F16OUT=1` (default on ROCm): each fp32-output reconstruct GEMM
result is rounded once to fp16, the precision the residual stream carries
anyway. Measured worst model-level KLD for that rounding is 1.9e-3
(vs 1e-3 gate default) — when A/B-ing across that default change, set
`NUMCHECK_MAXDIV` and `NUMCHECK_KLD` accordingly, or compare like with like.

## Known ROCm-open issue: transient wrong triton prefill output (full-suite only)

2026-10-01, updated after the upstream-dev merge: full `pytest tests/` runs
intermittently (~3 of 12 observed) fail 26-36 tests in
`test_triton_paged_hdpad.py` with rel err ~1.0 — prefill, nocache and
quantized-cache variants, while every decode variant passes. Pre-dates the
upstream merge (first seen on the cac783b build). Characterization so far:

- NOT reproduced by: the file alone (warm or cold), any in-order subset of the
  suite, 6000-iteration randomized stress across shapes/head dims/windows
  (`bench/stress_triton_prefill.py`), or in-process GPU memory pressure up to
  20 GB hauled.
- Only full-suite processes trigger it, probabilistically; the same command
  passes 5+ times in a row between occurrences.
- With `EXL3_ATTN_CANARY=2` the failures convert to RuntimeErrors at the
  prefill exit, confirming the wrapper returns wrong output rather than the
  tests mis-comparing. (The canary itself no longer materializes a float copy
  of the output — on the >2^31-offset overflow test that copy OOMs.)
- Suspected, unproven: interference from a co-resident GPU process (observed
  while a standing inference server was parked on the same card); lock
  history shows no concurrent lock holder during failing runs, but the parked
  server's residency was not recorded per run.

`EXL3_ATTN_CANARY=1` (warn) / `2` (raise) instruments decode
and prefill exits for detection in CI or a server until this is pinned. If it
reproduces, capture triton-rocm version + the canary message and file upstream
against triton-rocm.

### Bounded paged indexing — NOT on this branch (verified 2026-10-03)

The kernel-side bounds guards (`num_cache_pages` plumbing + `phys` clamps in the
C++/HIP paged kernels, the masked `_paged_kv_update_kernel` store and the
decode/prefill `phys`/`total_k_len` clamps in `triton_paged.py`) live on the
`port/upstream-paged-bounds` branch (PR in preparation) and are **not merged
into `port/rocm`**. An earlier revision of this section described them as
present here; that was wrong.

Verified empirically on this branch's build (image git-584dd44f,
`repro_436_portable.py` from `issue-436-repro` at 526e94b0, provenance-asserted
against the image's own `triton_paged.py`):

- an unmasked store through a `block_table` entry of `pool+2` lands exactly two
  pages past the cache pool (3/3 repeats, sentinel guard region);
- attention reads through garbage `phys` values (2^20 .. 2^31-1, -1) complete
  silently with finite output;
- a torn pinned staging upload still makes attention launched with table A
  return table B's output exactly (3/3 repeats, silent).

Until that PR merges, `EXL3_ATTN_CANARY` remains the only tripwire on this
branch, and the upload-ordering rule in `util/tensor.py` ("must not refill them
until a sync point") is the only guard against the staging tear.

## Testing

`bench/` holds the A/B harness:

- `bench/bench.py` — decode/prefill/batch benchmark (model via `EXL3_MODEL`).
- `bench/correctness_gate.py` — `baseline`/`compare` golden greedy-token parity,
  `batch` sequential-vs-concurrent equivalence (divergence policy above),
  `numcheck save|compare` logits KLD (includes the long/prefill prompts by
  default so `EXL3_HGEMM_F16OUT`-class changes are actually measured).
- `bench/check_so_wgp_mode.py` — ELF `.workgroup_processor_mode` gate for
  `-mcumode` builds (used by Dockerfile.rocm).
- `bench/fuzz_exl3_adversarial.py`, `bench/fuzz_hgemm_adversarial.py`,
  `bench/stress_triton_prefill.py` — kernel-vs-reference fuzz and the triton
  repro hammer (added by the 2026-10-01 review).

Both run inside the image; mount a model and pass `-e EXL3_MODEL=…`.
