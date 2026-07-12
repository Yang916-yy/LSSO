# LSSO MathDx backend

This optional CUDA backend removes the synchronization-heavy host-library path
for LSSO's small SPD systems. It targets rank 16/32 and treats relation groups
as an ordinary batch dimension, so the same kernels cover LSSO, RRLSSO, and
Grouped LSSO/RRLSSO for any relation-group count.

## Implemented paths

- `solve_spd`: one CTA per system; cuSolverDx POTRF is executed once and the
  factor is reused by 32-column POTRS tiles. RHS width is unrestricted.
- `stats_solve_spd`: one CTA computes FP32 `U.T @ U` with cuBLASDx, constructs
  `I + alpha * U.T @ U`, factors it with cuSolverDx, then loops over RHS tiles,
  computing `U.T @ C` and solving without writing Gram/RHS to global memory.
- `lsso.modules` uses the autograd-aware `solve_spd` path automatically.
  Missing libraries, CPU execution, unsupported
  ranks, and non-FP32 solve tensors fall back to `torch.linalg.solve_ex`.
- `RRLSSO` uses a fused CUDA rank-rotary kernel for fixed sequential positions.
  Its custom backward applies the inverse rotation in one kernel. Explicit
  `position_ids`, missing native libraries, and graph compilation retain the
  exact PyTorch implementation as a fallback.
- The standard unmasked encoder path fuses per-token RMS normalization,
  effective-length scaling, and optional rank rotation into one CUDA pass.
  Its backward fuses inverse rotation with the RMS-normalization gradient.
- Woodbury readout scales the compact rank-space solution and uses a GEMM
  epilogue to write `Y` directly, avoiding an output-sized `U @ K` temporary.
- CUDA BF16/FP16 Woodbury statistics use `bmm` with direct FP32 output inside
  the custom LSSO path. This removes reduced-precision statistic writes and
  follow-up conversion kernels. General autograd and higher-order derivative
  paths retain the ordinary differentiable `bmm` implementation.

The fully fused one-CTA path is selected conservatively for CV-sized token
grids when `N <= 512`, systems (`B * G`) are at most 64, and RHS width is at
most 64. Wider grouped-relation RHS tiles and large system batches favor two
parallel GEMMs followed by the device-side solve. Long sequences also need more
parallelism across N. This measured hybrid policy is deliberately the default
rather than forcing the fully fused kernel at every shape.

## Build in WSL

The local development build targets the installed GPU:

```bash
cd /root/LSSO
bash tools/build_mathdx_backend.sh
```

The release build produces a fat binary for Ampere (SM80/86/87), Ada (SM89),
Hopper (SM90), and Blackwell (SM100/120):

```bash
LSSO_MATHDX_RELEASE=1 \
LSSO_MATHDX_BUILD_DIR="$HOME/.cache/lsso-mathdx-release" \
LSSO_MATHDX_RECONFIGURE=1 \
bash tools/build_mathdx_backend.sh
```

Override `MATHDX_ROOT`, `CUDA_HOME`, `PYTHON_BIN`, or the architecture lists for
another installation. MathDx descriptors are architecture-specific, so both
`LSSO_CUDA_ARCHITECTURES` and `LSSO_MATHDX_LTO_ARCHITECTURES` must include any
additional target.

The checked release artifact was built successfully at
`build/mathdx-release/lib/lsso_mathdx.so` (about 6.2 MB). `cuobjdump --list-elf`
confirmed embedded cubins for SM80, 86, 87, 89, 90, 100, and 120.

## Verification

```bash
LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/x86_64-linux/lib \
  .venv/bin/python -m pytest -q tests/mathdx_backend_test.py

LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/x86_64-linux/lib \
  .venv/bin/python benchmarks/benchmark_mathdx_backend.py
```

On the development RTX 5070 Ti (SM120), rank 16/32 numerical tests covered RHS
widths 1--192 and non-aligned N=65/196. Maximum fused-path absolute error versus
`torch.linalg.solve` was below `5e-6`. Measured solve-only speedups were
`3.8--12.8x`. These numbers are local measurements, not cross-GPU claims.
With the conservative selector, a main-LSSO case with 64
systems, N=196, rank 32, and RHS 64 was `1.58x` faster than the already
accelerated parallel-stats + MathDx-solve hybrid; RHS 192 correctly selected
the hybrid path.

## Real ViT training A/B

On CIFAR-100 ViT-B/4 with BF16 autocast and batch 128, steady epoch timing over
120 complete training steps gave:

| mixer | G | backend off | backend on | throughput gain |
| --- | ---: | ---: | ---: | ---: |
| RRLSSO | 12 | 13.6656 s | 12.3433 s | 1.107x |
| Grouped-RRLSSO | 4 | 12.8183 s | 11.4041 s | 1.124x |

The G=1 MathDx run completed at 10.9753 seconds. The corresponding PyTorch
`solve_ex` run failed reproducibly in MAGMA with an illegal memory access for
RHS 768 at batch sizes 128 and 64, so it has no valid speed ratio. Treat these
as local development measurements and re-run
`benchmarks/benchmark_mathdx_backend.py` on the target GPU before reporting
hardware claims.

## RTX 5070 Ti rotary optimization signal

For the CIFAR ViT token shape `B=128, N=65, D=768, H=12, r=32`, the combined
rotary, basis-preparation, and readout work reduced eager per-layer RRLSSO
forward-plus-backward from about `4.06 ms` to `2.87 ms`; forward fell from
about `1.45 ms` to `0.94 ms`. LSSO reached `2.86 ms` forward-plus-backward and
`0.91 ms` forward. Per-layer peak allocation fell from about `251 MiB` to
`233 MiB`.

In a 12-layer ViT-B/4 forward-plus-backward measurement at batch 64, RRLSSO
fell from `56.39 ms` to `46.77 ms`, with peak allocation falling from about
`1.84 GiB` to `1.73 GiB`. The matched MHA measurement was `45.70 ms` and
`1.86 GiB`. These are local optimization measurements, not cross-GPU claims.

An attempted reuse of forward Cholesky factors in backward did not improve
wall time and increased per-layer peak memory by roughly 66 MiB at this shape,
so that path is deliberately not retained.

A standalone fused split/transpose and merge/transpose CUDA path was also
tested. It was about 1.5--2% slower than PyTorch's optimized contiguous
transposes and did not lower peak memory, so it is not retained. Future layout
work should place the transform inside a projection or readout GEMM epilogue
rather than launch another memory-only kernel.
