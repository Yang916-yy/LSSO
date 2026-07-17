# LSSO

LSSO (Learnable Structured Solve Operator) is a bidirectional token mixer for
encoder models. The current project is focused on computer vision: replace
self-attention in a standard ViT block while keeping the patch embedding, MLP,
residual structure, and training recipe fixed.

For a learned low-rank relation basis `U` and content `C`, the mixer solves

```text
(mu I + gamma U U^T) Y = C
```

through the Woodbury identity. The global relation is therefore represented by
a rank-sized SPD system rather than a dense token-token matrix.

## Active model and research program

`RRLSSO` (Rank-Rotary LSSO) is the primary model. `GroupedRRLSSO` is retained
as an optional relation-sharing variant; its group count changes the number of
relation fields, not the encoder width.

The active work has three linked tracks:

1. exact CUDA acceleration of the small SPD systems using cuBLASDx and
   cuSOLVERDx, for Ampere and newer GPUs;
2. effective-length normalization and a robust initialization interval for the
   global correction; and
3. controlled CV experiments, beginning with standard ViT on CIFAR-100 and
   then classification, retrieval, and segmentation.

The current default is `gamma_max=1.2`, `theta_gamma_init=0.5`,
`length_normalize=True`, and `length_reference=1.0`. This gives an initial
`gamma / mu` near 1.08, inside the controlled CIFAR-100 selection interval
`0.85..1.15`. Details are in
[`docs/global_correction.md`](docs/global_correction.md).

## Install

```bash
python -m pip install -e .
```

The core package requires PyTorch. The optional native MathDx backend is
documented in [`docs/mathdx_backend.md`](docs/mathdx_backend.md).

## Minimal use

```python
import torch
from lsso import RRLSSO

x = torch.randn(2, 197, 768)
mixer = RRLSSO(dim=768, num_heads=12, rank=32)
y = mixer(x)
```

Use the module in the attention slot of an encoder block:

```text
X = X + RRLSSO(LN(X))
X = X + MLP(LN(X))
```

## Repository layout

The installable framework exposes stable interfaces through `lsso`,
`lsso.ops`, `lsso.nn`, and `lsso.backends`. Legacy implementation imports are
kept compatible while internals are split incrementally. See
[`docs/repository_architecture.md`](docs/repository_architecture.md) for layer
boundaries and [`CONTRIBUTING.md`](CONTRIBUTING.md) for change and artifact
policy.

```text
lsso/          installable operators, modules, and backend facades
csrc/mathdx/   fused CUDA/MathDx implementation
examples/      compact model integration references
experiments/   reproducible vision and sequence task programs
integrations/  timm, MMDetection, and MMSegmentation adapters
benchmarks/    performance and complexity measurement
tests/         numerical, dispatch, integration, and regression checks
docs/          architecture, recipes, and active research notes
paper/         manuscript sources and compact experimental tables
tools/         build, data, and repository operations
```

Historical causal, retrieval, diffusion, and paper artifacts are deliberately
not part of the current working tree. They remain reachable through Git
history and existing release tags.
