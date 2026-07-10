# Bidirectional Flash LSSO Prototype

This folder is isolated from the causal `flashlsso_prototype/` line. It targets
non-causal vision workloads where LSSO/RRLSSO can use full-sequence statistics.

## Implemented

- `flash_bidir_lsso.py`
  - Triton forward for the large sequence reductions:
    - `S = U^T U`
    - `P = U^T C`
    - `Y = C / mu - gamma / mu^2 * U @ solve(I + gamma / mu * S, P)`
  - Fused forward statistics kernel for `S` and `P`.
  - PyTorch `torch.linalg.solve_ex(..., check_errors=False)` for `rank x rank`
    systems. Profiling showed this is faster than `solve`, `cholesky_solve`,
    and `inv @ P` on the local RTX 5070 Ti.
  - Specialized no-autotune forward kernels for the common
    `rank=32, head_dim=64` path.
  - Analytic backward:
    - Triton reduction for `dK = U^T dY`
    - PyTorch small-matrix solve for `dP`
    - Triton token kernel for `dU` and `dC`
    - PyTorch reductions for per-head `mu/gamma` gradients
- `bidir_lsso_flash_layer.py`
  - `BidirLSSOFlashPrototype`
  - `BidirRRLSSOFlashPrototype`
- `test_flash_bidir_lsso_numerical.py`
  - Core and layer-level forward/backward checks against existing PyTorch LSSO/RRLSSO.
- `benchmark_flash_bidir_lsso.py`
  - Forward-only timing baseline for CV-like sequence lengths.
- `benchmark_flash_bidir_cv_block.py`
  - ViT-style block timing with LayerNorm, mixer, MLP, residuals, and backward.

## Current Status

Correctness is verified against the existing PyTorch implementation. Forward
performance is not yet compelling for short and medium sequence lengths because
PyTorch's batched GEMMs are already strong for this small-rank workload and the
prototype still pays multiple kernel launches plus a PyTorch small-matrix solve.

Profiling confirms the small `rank x rank` solve is the main bottleneck:

```text
RTX 5070 Ti, B=2,H=8,R=32,DH=64
N=3136 bf16: stats=0.053ms solve=0.252ms solve_ex=0.154ms output=0.022ms
N=8192 bf16: stats=0.130ms solve=0.270ms solve_ex=0.188ms output=0.031ms
```

RTX 5070 Ti, `B=2,H=8,R=32,DH=64` forward-only:

```text
N=8192 bf16: torch=0.551ms flash=0.506ms speedup=1.09x
N=4096 bf16: torch=0.339ms flash=0.485ms speedup=0.70x
N=196  bf16: torch=0.238ms flash=0.491ms speedup=0.48x
```

RTX 5070 Ti, ViT-style block with backward, `B=2,D=512,H=8,R=32,bf16`:

```text
LSSO   N=3136: torch_block=3.415ms flash_block=3.827ms speedup=0.89x
RRLSSO N=3136: torch_block=3.898ms flash_block=4.193ms speedup=0.93x
LSSO   N= 784: torch_block=2.220ms flash_block=2.973ms speedup=0.75x
RRLSSO N= 784: torch_block=2.530ms flash_block=3.138ms speedup=0.81x
```

The next useful optimization pass is:

1. Fuse output projection with the solve readout or move more of the small-rank
   solve path into a persistent/specialized kernel.
2. Consider a Triton-only exact `32x32` SPD solve only if profiling in real CV
   models still shows `solve_ex` dominating.
3. Benchmark inside actual ViT/SegFormer/Swin integrations after replacing a
   full attention block, not only the synthetic block here.

## Reference Skeleton

The closest borrowed skeleton is
`external/flash-bidirectional-linear-attention/flash_bla/ops/simple_la/fused.py`,
which implements bidirectional linear attention as:

```text
state = K^T V
output = Q state
```

Bidirectional LSSO follows the same broad scheduling shape but adds one
small-rank SPD solve between the global state and token output.
