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
Within a dispatch family the output *is* bitwise batch-independent (verified
m=5..2048 on model tensors).

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
- Suspected, unproven: interference from a co-resident GPU process (this
  machine runs a standing serve that parks/unparks around lock windows); lock
  history shows no concurrent lock holder during failing runs, but the parked
  serve's residency was not recorded per run.

`EXL3_ATTN_CANARY=1` (warn; serve default) / `2` (raise) instruments decode
and prefill exits for detection in serve/CI until this is pinned. If it
reproduces, capture triton-rocm version + the canary message and file upstream
against triton-rocm.

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
