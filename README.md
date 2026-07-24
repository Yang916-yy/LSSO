# LSSO: Exact Inner Learning for Token Mixing

[![CI](https://github.com/Yang916-yy/LSSO/actions/workflows/ci.yml/badge.svg)](https://github.com/Yang916-yy/LSSO/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

**What if a token mixer could learn from the current sample without unrolling
an inner optimizer?**

The **Learnable Structured Solve Operator (LSSO)** is a global token mixer that
turns per-sample inner learning into an exact low-rank solve. The outer network
constructs a strongly convex learning problem from the current input; LSSO
computes its unique optimum directly instead of approximating it with a small
number of gradient updates.

```text
construct a sample-specific inner problem
                  ↓
       solve its exact equilibrium
                  ↓
   read out temporary fast weights
```

## From unrolled adaptation to an exact solution

Attention can be viewed as a context-instantiated dynamic network, while
test-time training (TTT) makes the learning interpretation explicit: tokens
form a temporary dataset, an inner model adapts to that dataset, and its
adapted state processes the same context. In practice, inner learning is often
limited to a few unrolled steps—more inner depth increases optimization and
memory cost.

LSSO takes a different route. For every sample and every head, it predicts a
low-rank relation basis `U` and content target `C`, then solves the token-space
equilibrium

```text
Y = g (I + alpha U U^T)^-1 C .
```

The same computation has an exact rank-space learning interpretation:

```text
Z* = argmin_Z  1/2 ||Z||_F^2 + alpha/2 ||UZ - C||_F^2,
Y  = g (C - UZ*).
```

`Z*` is a set of **sample-conditioned temporary fast weights**. The
token-space equilibrium and rank-space learner are an exact primal–dual pair.
The closed-form solve equals the infinite-step limit of any convergent gradient
inner loop, eliminating finite-step truncation and unrolling for this
structured learner.

Through Woodbury, the implementation solves only an `r × r` symmetric
positive-definite system and never forms an `N × N` token matrix. The mixer is
therefore linear in sequence length for fixed rank.

## RRLSSO: rotating the geometry of inner learning

**Rank-Rotary LSSO (RRLSSO)** applies token-dependent orthogonal rotations to
the low-rank features before the solve. Rank rotary is more than a positional
tag: it reshapes the covariance, conditioning, effective rank, and adaptation
directions of the sample-specific regression problem while preserving row
norms, total relation energy, and positive-semidefinite Gram structure.

The maintained implementation uses ordinary one-dimensional Rank Rotary over
the token order. The retired axial 2-D construction is not part of the current
operator or experimental baseline.

## Why LSSO?

- **Exact per-sample learning** — computes the unique optimum of a strongly
  convex inner problem instead of choosing an arbitrary number of updates.
- **Global mixing without dense attention** — uses low-rank sufficient
  statistics rather than an explicit token-token score matrix.
- **Fast-weight interpretation** — exposes the rank-space state learned from
  each sample and its regularized residual readout.
- **Controlled dynamics** — positive definiteness gives a unique solution and
  a stable rational spectral response.
- **Linear token complexity** — `O(N r (r + d_h))` mixer arithmetic for fixed
  rank, with an `r × r` solve.
- **Optimized execution** — PyTorch fallback plus fused CUDA/MathDx statistics,
  solve, readout, masked, long-sequence, and backward paths.

## Current evidence

On eight **GenomicBenchmarks** tasks, a compact RRLSSO encoder beats a matched
MHA encoder on seven tasks without pretraining:

| Mixer | Macro accuracy | Mixer MAC reduction | Longest-task time |
|---|---:|---:|---:|
| MHA | 85.85% | 1.00× | 3.18 s/epoch |
| RRLSSO | **86.54%** | **1.64×–12.08×** | **1.73 s/epoch** |

On the longest Mouse Enhancers task, mixer MACs fall from `25.16G` to `2.08G`,
yielding a measured `1.83×` end-to-end training speedup. Large-scale vision
classification, detection, and segmentation experiments are in progress; the
repository does not claim those results before they are complete.

The maintained ImageNet program registers RRLSSO versions of the standard
DeiT-III Small, Base, and Large backbones. It reproduces Meta's size-specific
DeiT-III recipe, including constant stochastic depth, LayerScale `1e-4`,
official class-token initialization, virtual-device grouped Mixup/CutMix,
distributed-style repeated augmentation, BCE, EMA, and Apex FusedLAMB
pretraining, plus the published 224/384 refinement stages. See
[`docs/imagenet_deit3_rrlsso.md`](docs/imagenet_deit3_rrlsso.md) for model
names, parameter counts, exact recipes, virtual augmentation semantics, and the
unbounded log-strength and reciprocal-Woodbury parameterization.

## Install

```bash
git clone https://github.com/Yang916-yy/LSSO.git
cd LSSO
python -m pip install -e .
```

The core package requires PyTorch. The native CUDA/MathDx backend is optional;
unsupported devices and shapes retain the portable PyTorch path. See
[`docs/mathdx_backend.md`](docs/mathdx_backend.md) for build and dispatch
details. The retained/retired kernel rationale and verification record are in
[`docs/cuda_path_audit.md`](docs/cuda_path_audit.md).

The current main branch uses CUDA backend ABI 2. It adds the compact forward
state required by the fused backward and the unbounded log-alpha/reciprocal
Woodbury contract. ABI 1 wheels from LSSO 0.2.0 are deliberately rejected.
Install the v0.3.0 runtime matching `torch.version.cuda` to skip local NVCC and
MathDx compilation:

```bash
# PyTorch 2.11 + CUDA 12.8 (recommended)
pip install 'https://github.com/Yang916-yy/LSSO/releases/download/v0.3.0/lsso_mathdx_runtime-0.3.0%2Btorch2110cu128-py3-none-linux_x86_64.whl'

# PyTorch 2.11 + CUDA 13.0
pip install 'https://github.com/Yang916-yy/LSSO/releases/download/v0.3.0/lsso_mathdx_runtime-0.3.0%2Btorch2110cu130-py3-none-linux_x86_64.whl'

# Other PyTorch/CUDA combinations can build from source.
bash tools/build_mathdx_backend.sh
```
Experiment checkpoints are disposable local artifacts; the retention and
directory rules are in
[`docs/experiment_outputs.md`](docs/experiment_outputs.md).

## Minimal use

```python
import torch
from lsso import RRLSSO

x = torch.randn(2, 197, 768)
mixer = RRLSSO(dim=768, num_heads=12, rank=32)
y = mixer(x)

assert y.shape == x.shape
```

The supported constructors intentionally do not expose an alpha initialization
knob. Learnable `theta_alpha` starts from the dimensionless reference
`alpha=1` and is restored exactly from checkpoints, preventing accidental
extreme initialization without restricting the learned solve strength.

Drop it into the token-mixing slot of a pre-norm encoder block:

```text
X = X + RRLSSO(LN(X))
X = X + MLP(LN(X))
```

The public package surface is exposed through `lsso`, `lsso.ops`, `lsso.nn`,
and `lsso.backends`. Maintained examples use standard encoder backbones so the
operator can be evaluated without inheriting a bespoke vision architecture.

## Repository map

```text
lsso/          installable operators, modules, and backend facades
csrc/mathdx/   fused CUDA/MathDx implementation
examples/      model integration references
experiments/   reproducible vision and sequence programs
benchmarks/    performance and complexity measurement
tests/         numerical, dispatch, integration, and regression checks
docs/          architecture, recipes, and research notes
paper/         manuscript sources and compact experimental tables
archive/       unsupported historical prototypes, excluded from the package
```

See [`docs/repository_architecture.md`](docs/repository_architecture.md) for
layer boundaries and [`CONTRIBUTING.md`](CONTRIBUTING.md) for contribution,
compatibility, and artifact policy.

## Research status

LSSO is an active research codebase. The mathematical operator, RRLSSO geometry,
portable implementation, optimized CUDA paths, model adapters, and genomic
evaluation are implemented. The paper and formal computer-vision evaluation
are still evolving; APIs covered by the public namespaces are kept compatible
while internal modules are progressively reorganized.
