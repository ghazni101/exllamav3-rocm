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

## Testing

`bench/` holds the A/B harness:

- `bench/bench.py` — decode/prefill/batch benchmark (model via `EXL3_MODEL`).
- `bench/correctness_gate.py` — `baseline`/`compare` golden greedy-token parity,
  `batch` sequential-vs-concurrent equivalence, `numcheck save|compare` logits KLD.
- `bench/check_so_wgp_mode.py` — ELF `.workgroup_processor_mode` gate for
  `-mcumode` builds (used by Dockerfile.rocm).

Both run inside the image; mount a model and pass `-e EXL3_MODEL=…`.
