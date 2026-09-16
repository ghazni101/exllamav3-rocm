# ROCm profiling harness (gfx1100 / RX 7900 XTX)

Companion to `docs/rocm-perf-gfx1100-tg-pp-plan.md` and `docs/rocprofv3-blockers.md`. Two modes:

- **Sibling container** (model in a throwaway container, standing serve stopped): full-VRAM
  traces and benches. Scripts: `run_b_flagiso.sh`, `run_cli_trace.sh`, `run_cli_pmc.sh`,
  `bench_lean.py`, `sweep.sh`.
- **Live serve**: the standing serve stays up; profile it through attach mode, or measure it
  over HTTP. Scripts: `run_attach_live.sh`, `run_d_attach.sh`, `ttft_probe.py`, `bench_ab.py`.

**All GPU work must run under `~/gpu-coord/gpu-ctl run` / `acquire`**; full-VRAM runs need the
standing serve stopped first (the model needs ~18 GB of the 24 GB card); attach/HTTP work is
`[svcon]`-tagged so the serve stays up.

## Why these scripts look the way they do

- **CLI tool mode**: `rocprofv3 -- <app>` works only with `profiling/shims/ld_scope_shim.so`
  LD_PRELOADed — the tool's C++ ABI symbols otherwise collide with triton's bundled libstdc++
  and the process dies in `cfree` (root cause + fix: `docs/rocprofv3-blockers.md`, Blocker 1).
  Keep `--ulimit nofile=65536:65536` on any rocprofv3 container.
- **Attach mode**: `rocprofv3 --attach` is broken on this SDK (wrapper hangs). Use
  `rocprof-attach` **directly**, on a **quiescent** target, with the tool config in the client's
  env — full recipe and 2×2 matrix: `docs/rocprofv3-blockers.md` Blocker 2,
  `docs/rocprofv3-findings-log.md` §9. Target-side env: `profiling/attach_override.yml`.
- **Counters**: only `SQ_WAVES`/`SQ_BUSY_CYCLES` return data on gfx1100/SDK 1.3.5; every
  memory-system counter is rejected (error 38) or always-0 (census: findings-log §8.1).

## Scripts

| script | what it measures |
|---|---|
| `bench_lean.py` | load once → decode b1 (3×192 tok, median), cold prefill 2 k, batch-4 and batch-8 aggregate. One config per process (the RV knobs are read once and cached in statics). |
| `sweep.sh` | runs `bench_lean.py` for baseline / `EXL3_SQ_ROWS_PER` 32/128/256 / `EXL3_INT8_GEMV=1` / `EXL3_INT8_MSQ=0` under one lock hold. |
| `run_b_flagiso.sh` | the session-2 primary: full-model CLI kernel trace (`trace_main`), flag-isolation runs, `--hip-graph-trace` coverage, SQ counter census (`sqmap`). |
| `run_cli_trace.sh` / `run_cli_pmc.sh` | older single-purpose CLI trace / counter runners (superseded by `run_b_flagiso.sh` for attribution work). |
| `run_counter.sh`, `run_prefill_trace.sh` | register-env-mode runners kept for reference (pre-shim). |
| `run_ctr_probe.sh` + `counter_probe.py` | known-traffic memory probe → the counter census (findings-log §8.1). |
| `run_attach_live.sh` | attach-profile the LIVE serve: recreate with `attach_override.yml`, attach `rocprof-attach` directly while idle, drive `ttft_probe.py` inside the window. |
| `run_d_attach.sh` | earlier attach driver (TTFT probe + quiescent attach test). |
| `ttft_probe.py` | HTTP streaming TTFT: warmup, short decode, cold/warm prefix, 1.5 k, 2 concurrent decodes. Writes `/tmp/ttft_<label>.json`. |
| `bench_ab.py` | non-streaming HTTP decode/prefill A/B (repo root). |
| `analyze_trace.py` / `phase_stats.py` | post-process a `1_kernel_trace.csv`: class shares, grid census, per-segment busy-union vs wall. |
| `bw_probe.py` | practical DRAM peak (copy + read-only). |
| `gemm_probe.py` | hipBLASLt fp16 TFLOP/s on the model's own prefill shapes. |

Trace outputs land in the mounted output dirs (hundreds of MB — they are regenerable, do not
commit them).

## Knobs worth knowing

- `EXL3_INT8_GEMV`: 2 = plain int8 GEMV (default, and the fast one), 1 = residual mode (measured
  regression), 0 = fp16 QTIP GEMV.
- `EXL3_INT8_MSQ=0` disables the multi-matrix/sliced regular-launch kernel: large regression, it
  is the path that made batched decode viable. Use it only as an A/B control.
- `EXL3_SQ_ROWS_PER`: slice height, pinned per-arch default (64 for all of ROCm today; gfx1100
  may prefer 32 — see the plan, the difference is inside noise until re-measured with reps).
- `EXL3_INT8_GEMV_MAX_K`: raise to 6 to cover the four K=6 tensors in this model.
