# LSSO CUDA/MathDx backend

The maintained CUDA extension is deliberately narrow. It accelerates the
default Trace-normalized LSSO/RRLSSO computation and retains only primitives
that win on an active training path. Historical `(mu, gamma)`, token-RMS, and
architecture-probe kernels live under `archive/` or in repository history.

## Maintained native surface

| operator | purpose |
| --- | --- |
| `solve_spd` | FP32 bucketed cuSolverDx solve, ranks 1--64 |
| `masked_trace_stats_solve_readout` | masked or null-mask Trace statistics, Cholesky, and Woodbury readout |
| `masked_stats_solve_readout` | effective-strength adjoint solve/readout used by Trace backward |
| `dual_backward_statistics_tensorcore` | joint `Y.T@U` and `P.T@U` statistics |
| `dual_grad_u_tensorcore` | Tensor-Core `grad_U` plus Trace radial epilogue |
| `rank_rotary` | Rank Rotary and its inverse |

The Trace forward accepts FP16/BF16 inputs, accumulates Gram and cross
statistics in FP32, performs FP32 POTRF/POTRS, rounds the compact solution once
to the activation dtype, and writes

```text
Y = gain * (C - effective_alpha * U @ K)
```

directly. Mask predicates are evaluated before U/C loads, so poisoned padding
cannot enter compact statistics or output. Ranks 16/32/48/64 use the fused
path; other ranks up to 64 use padded FP32 solve buckets after PyTorch compact
statistics.

Trace backward does not use the retired scalar all-in-one kernel. It performs
an adjoint solve, selects dual WMMA or cuBLAS compact statistics by shape, then
uses the dual Tensor-Core `grad_U` epilogue where eligible. The epilogue also
adds the derivative of Trace normalization before the only activation-sized
write. A differentiable PyTorch recomputation handles `create_graph=True`.

The fused readout accepts a strided logical `[B,H,N,width]` right-hand side
directly from the token-major `w_uc` projection and stores its logical output
in token-major physical order. Consequently `output.transpose(1,2)` feeds
`w_o` without a full activation copy. The dual-statistics and `grad_U` kernels
also consume strided token-major `Y/P`; only portable fallbacks materialize a
head-major contiguous layout. When the radial epilogue is active, its BF16/FP16
`grad_U` is returned directly without a BF16/FP16-to-FP32 round trip or a
second padding-mask pass.

CUDA low-precision training intentionally saves only tensors required by the
analytic first-order backward. Saving the strided `C` view solely for a
possible double backward would retain the complete joint `UC` projection
storage. CPU/reference execution continues to support `create_graph=True` and
is covered by gradgrad tests. A CUDA experiment that explicitly needs double
backward can opt into the additional saved activation with
`LSSO_CUDA_HIGHER_ORDER=1`.

Ultra-long masked inputs use split-N statistics. The dispatcher chooses a CTA
path or split-N from sequence length, padding hint, system count, and GPU
family. Override only for architecture measurements:

```text
LSSO_MASKED_TRACE_SPLIT_N=cta|split_n|auto
LSSO_MASKED_TRACE_SPLIT_CHUNK=<positive integer>
```

Path counters (`LSSO_PATH_COUNTERS=1`) and solver-info checks
(`LSSO_MATHDX_DEBUG_INFO=1`) are debug-only; the latter synchronizes the host.

## Deliberately retired

- 2/4 systems-per-CTA Cholesky (`solve_spd_bpb`): slower than one CTA/system;
- non-Trace fused statistics/readout: only served the historical formulation;
- masked non-Trace stats-only solve;
- scalar all-in-one backward: blocked Tensor Cores and lost at training shapes;
- rotary-on-load Trace fusion: repeated rotation and was slower than the
  separate strided Rank Rotary kernel;
- native token-RMS preparation: retained only as a PyTorch ablation.

See `archive/retired_cuda_compat/README.md` and
`archive/retired_cuda_token_rms/README.md`. Retired numerical behavior remains
available through PyTorch compact statistics plus `solve_spd`; retired native
operators are not part of the public extension ABI.

## Build and verify

The maintained native operator contract is **CUDA backend ABI 1**, introduced
in LSSO 0.2.0. Operator schemas, logical tensor layouts, dtype behavior, mask
semantics, and fallback behavior are frozen for this ABI. Kernel-internal
scheduling may change without incrementing it. The loader verifies the ABI
reported by the shared library before dispatching an operator.

```bash
cd /root/LSSO
bash tools/build_mathdx_backend.sh

LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/x86_64-linux/lib \
  .venv/bin/python -m pytest -q tests/mathdx_backend_test.py
```

The development build targets the installed GPU. Set
`LSSO_MATHDX_RELEASE=1` for the configured Ampere/Ada/Hopper/Blackwell release
architecture set. CUDA 12 and CUDA 13 require matching MathDx packages and
device-LTO objects; the build script rejects mixed major-version artifacts.

Official release assets provide separate Linux x86-64 runtime wheels for
PyTorch 2.11 with CUDA 12.8 and CUDA 13.0. The loader validates the exact
PyTorch and CUDA build before using a packaged runtime; incompatible or absent
packages retain the portable PyTorch fallback. Source builds remain available
for other environments.

The CUDA 12.8 wheel uses MathDx 25.12.1 and is the default recommendation. The
CUDA 13.0 wheel uses MathDx 26.06. Both contain native SASS for SM80, SM86,
SM87, SM89, SM90, SM100, and SM120, carry no build-host RUNPATH, and report
backend ABI 1. Runtime wheels are release assets rather than Git-tracked
binaries.
