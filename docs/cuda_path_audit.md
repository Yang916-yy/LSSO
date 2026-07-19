# CUDA path audit

Audit date: 2026-07-19. Scope: native LSSO/RRLSSO forward, backward, compact
solve, masking, Rank Rotary, dispatch, Python exposure, tests, and build output.

## Retention rule

A native path stays in the shipping extension only when it is used by the
default Trace-normalized operator and has a measured reason to exist. CPU,
unsupported-rank/dtype, graph-compilation, and second-order support remain
PyTorch fallbacks rather than parallel CUDA implementations.

## Resulting call graph

```text
Trace forward
  -> masked/null-mask fused statistics + POTRF/POTRS + readout (r=16/32/48/64)
  -> split-N compact statistics + bucket solve for ultra-long sequences
  -> PyTorch compact statistics + bucket solve for other ranks/layouts

Trace backward
  -> effective-strength adjoint readout
  -> dual WMMA or cuBLAS compact statistics
  -> dual Tensor-Core grad_U + radial epilogue
  -> differentiable PyTorch recomputation for create_graph=True

RRLSSO
  -> standalone strided Rank Rotary
  -> the same Trace paths as LSSO
```

The maintained native path uses logical `[B,H,N,*]` tensors with explicit
strides. `C`, forward output, and backward `Y/P` remain token-major in physical
memory through the surrounding projections; only the fallback requests a
head-major contiguous copy.

The public extension contains six operators: `solve_spd`,
`masked_stats_solve_readout`, `masked_trace_stats_solve_readout`,
`dual_backward_statistics_tensorcore`, `dual_grad_u_tensorcore`, and
`rank_rotary`.

## Removed from the shipping extension

| path | audit finding |
| --- | --- |
| 2/4 systems per Cholesky CTA | slower than one independent CTA per system |
| legacy unmasked stats/readout kernel | served non-Trace `(mu, gamma)` only |
| masked stats-only kernel | redundant after effective-strength readout |
| scalar all-in-one backward | FP32 scalar FMA prevented Tensor-Core use |
| rotary-on-load Trace fusion | repeated rotation and lost to materialize-once |
| unmasked TMA staging kernel | belonged to the retired legacy stats kernel |
| FP32 fused Trace GEMM variants | not the mixed-precision training fast path |

The historical benchmark/test/document snapshots are in
`archive/retired_cuda_compat/`. Token-RMS CUDA preparation has its own archive.
No retired native schema remains registered.

## Measured cleanup

- CUDA translation unit: about 2,560 to 1,410 lines (about 45% fewer);
- local SM120 `.so`: 14,196,712 to 4,551,888 bytes (about 68% smaller);
- focused Trace/mask/backend tests: 117 passed;
- complete current test suite: 220 passed;
- SM120 BF16 `B=16,H=12,N=197,r=32,d_h=64`: absorbed Trace forward+backward
  0.8135 ms and 0.054 GiB peak versus explicit normalization 1.4515 ms and
  0.059 GiB.

These are local implementation measurements, not cross-GPU performance
claims. Release fatbins still require CI/build verification on each supported
architecture family.

## Remaining intentional fallbacks

- ranks 1--64 outside 16/32/48/64: PyTorch compact statistics plus bucketed
  FP32 MathDx solve;
- rank above 64, CPU, unsupported dtype/layout, or compilation mode: pure
  PyTorch solve path;
- second-order gradients: differentiable PyTorch Trace recomputation;
- token-RMS and historical `(mu, gamma)` behavior: PyTorch ablation only.

These fallbacks preserve research usability without multiplying native ABI or
kernel maintenance burden.
