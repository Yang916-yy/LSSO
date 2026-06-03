# LSSO Operator

LSSO = Learnable Sylvester Solve Operator.

LSSO is a PyTorch token mixer for bidirectional encoder-style models. It
learns low-rank global relation features and solves a positive-shifted linear
system instead of routing values with pairwise attention.

```text
LSSO(X) = Y W_o
(mu I + gamma U(X) U(X)^T) Y = C(X)
```

The implementation uses the Woodbury form, so the solve works on a small
`rank x rank` system rather than an `N x N` inverse.

## Install

For local development:

```bash
python -m pip install -e .
```

If pip tries to create an isolated build environment and your package mirror is
offline, reuse the current environment's build tools:

```bash
python -m pip install -e . --no-build-isolation
```

The core package only depends on PyTorch.

## Use As A Token Mixer

```python
import torch
from lsso import LSSO

x = torch.randn(2, 128, 256)
mixer = LSSO(dim=256, num_heads=8, rank=32)
y = mixer(x)
```

Drop it into an encoder block where bidirectional self-attention would normally
live:

```text
X = X + LSSO(LN(X))
X = X + FFN(LN(X))
```

Everything else stays aligned with the baseline Transformer encoder.

## Core Package Scope

The package intentionally exposes only the core bidirectional token mixer:

```python
from lsso import LSSO, lsso
```

`LSSO` is meant to be dropped into existing encoder-style systems in the same
slot where a bidirectional token mixer would live. Repository model wrappers
for ViT-style classifiers, BERT-style retrieval encoders, DiT experiments, and
baseline comparisons live under `examples/models/`. They are examples for
integration and experiments, not part of the installed core API.

## Functional API

The functional entry point is analogous to an attention kernel:

```python
from lsso import lsso

# U: [B, H, N, r], C: [B, H, N, dh]
# mu/gamma: [H] or broadcastable to [B, H, 1, 1]
Y = lsso(U, C, mu, gamma)
```

Useful module arguments:

```text
rank: low-rank solve size, usually 16 or 32
gamma_max: maximum global correction strength
theta_gamma_init: gamma initialization, default -6.0
normalize_u: RMS-normalize U for stability
no_global: ablation path with gamma = 0
```

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Smoke Test

```bash
python -m tests.smoke_test
```

## GitHub Release Checklist

This repository is set up so the first public GitHub push can keep the package
small:

```bash
git init
git add .gitignore LICENSE NOTICE MANIFEST.in README.md pyproject.toml lsso tests examples requirements.txt
git commit -m "Release LSSO operator package"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

Large local experiment folders and scripts such as `data/`, `runs/`,
`runs_archive/`, `scripts/`, and `train_*.py` are ignored by default. They can
live in a research workspace without becoming part of the installable operator
package.
