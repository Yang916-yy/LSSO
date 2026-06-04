# LSSO

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
gamma_max: maximum global correction strength, default 0.3
theta_gamma_init: gamma initialization, default -4.0
normalize_u: RMS-normalize U for stability
no_global: ablation path with gamma = 0
```

## Solve Scale Defaults

The recommended LSSO initialization is:

```python
LSSO(dim=dim, num_heads=heads, rank=rank, gamma_max=0.3, theta_gamma_init=-4.0)
```

These two values control how much global solve correction is available at the
start of training. With `theta_mu = 0`, this gives roughly:

```text
mu ~= softplus(0) ~= 0.693
gamma ~= 0.3 * sigmoid(-4) ~= 0.0054
gamma / mu ~= 0.0078
```

This is intentionally not tiny. If `gamma_max` is too small or
`theta_gamma_init` is too negative, LSSO can behave almost like the local
`mu^-1 C(X)` projection early in training, which weakens the global solve term.
For example, `gamma_max=0.1, theta_gamma_init=-6.0` starts around
`gamma / mu ~= 3.6e-4`, which is usually too conservative for main experiments.

## Paper Experiment Results

Completed paper experiments are organized under [`paper_results/`](paper_results/).
The repository tracks lightweight artifacts only: summary tables, manifests,
source scripts, and JSONL logs. Model checkpoints are not tracked in git; they
are available from the GitHub Release:

```text
https://github.com/Yang916-yy/LSSO/releases/tag/paper-results-v0
```

See [`paper_results/release_assets.tsv`](paper_results/release_assets.tsv) for
release asset names, sizes, SHA256 checksums, and contents.

### Retrieval Main Table

Random-initialized BERT-style retrieval encoders, 3 seeds, `dim=256`,
`depth=8`, `heads=8`, `max_doc_len=512`, mean pooling. MACs are mixer-only
document-side MACs.

| Dataset | Model | Params (M) | Mixer MACs (G) | Save vs MHA | R@10 | MRR@10 | Samples/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FIQA | MHA | 14.33 | 2.147 | 0.0% | 0.2145±0.0135 | 0.0987±0.0032 | 597.1 |
| FIQA | Nystromformer | 14.33 | 1.301 | 39.4% | 0.2428±0.0039 | 0.1190±0.0071 | 375.2 |
| FIQA | LSSO-r16 | 13.54 | 0.713 | 66.8% | 0.2258±0.0125 | 0.1126±0.0061 | 676.0 |
| FIQA | LSSO-r32 | 13.80 | 0.908 | 57.7% | 0.2387±0.0183 | 0.1150±0.0035 | 605.7 |
| NFCorpus | MHA | 14.33 | 2.147 | 0.0% | 0.5439±0.0187 | 0.3809±0.0145 | 599.3 |
| NFCorpus | Nystromformer | 14.33 | 1.301 | 39.4% | 0.5841±0.0125 | 0.4323±0.0132 | 376.1 |
| NFCorpus | LSSO-r16 | 13.54 | 0.713 | 66.8% | 0.5686±0.0125 | 0.4121±0.0036 | 673.2 |
| NFCorpus | LSSO-r32 | 13.80 | 0.908 | 57.7% | 0.5841±0.0036 | 0.4333±0.0220 | 599.9 |
| SciFact | MHA | 14.33 | 2.147 | 0.0% | 0.6789±0.0190 | 0.5994±0.0138 | 581.7 |
| SciFact | Nystromformer | 14.33 | 1.301 | 39.4% | 0.7267±0.0133 | 0.6313±0.0033 | 366.3 |
| SciFact | LSSO-r16 | 13.54 | 0.713 | 66.8% | 0.7022±0.0107 | 0.6214±0.0119 | 654.3 |
| SciFact | LSSO-r32 | 13.80 | 0.908 | 57.7% | 0.7089±0.0117 | 0.6247±0.0080 | 586.6 |

Full table: [`paper_results/retrieval_main/summary.tsv`](paper_results/retrieval_main/summary.tsv).

### Retrieval Ablations

The main ablations use FIQA and SciFact with the same retrieval setup. The
`no-global` variant fixes `gamma=0`, removing the global solve correction.

| Dataset | Variant | R@10 | MRR@10 | gamma/mu | Correction ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| FIQA | no-global r32 | 0.2052±0.0137 | 0.0911±0.0019 | 0.0000 | 0.0000 |
| FIQA | fixed mu/gamma r32 | 0.2428±0.0126 | 0.1136±0.0028 | 0.0078 | 0.2130 |
| FIQA | no U RMS norm r32 | 0.2160±0.0101 | 0.1007±0.0008 | 0.0085 | 0.2050 |
| FIQA | full r8 | 0.2088±0.0208 | 0.0923±0.0139 | 0.0100 | 0.1059 |
| FIQA | full r4 | 0.2042±0.0045 | 0.0979±0.0036 | 0.0109 | 0.0742 |
| SciFact | no-global r32 | 0.6767±0.0120 | 0.5811±0.0058 | 0.0000 | 0.0000 |
| SciFact | fixed mu/gamma r32 | 0.7089±0.0102 | 0.6088±0.0066 | 0.0078 | 0.2631 |
| SciFact | no U RMS norm r32 | 0.6933±0.0133 | 0.5972±0.0150 | 0.0078 | 0.0471 |
| SciFact | full r8 | 0.7067±0.0173 | 0.6118±0.0064 | 0.0078 | 0.1149 |
| SciFact | full r4 | 0.7022±0.0069 | 0.6146±0.0107 | 0.0078 | 0.0766 |

Full table: [`paper_results/retrieval_ablation/summary.tsv`](paper_results/retrieval_ablation/summary.tsv).

### CIFAR-100 CV Main Table

Patch-2 ViT-style encoder on CIFAR-100, 3 seeds, `dim=96`, `depth=3`,
`heads=6`, CLS pooling, RandAugment(2,9), Mixup=0.2, CutMix=0.5. MACs are
mixer-only MACs.

| Model | Params (M) | Mixer MACs (G) | Save vs MHA | Top-1 | Epoch sec |
| --- | ---: | ---: | ---: | ---: | ---: |
| MHA | 0.3714 | 0.0665 | 0.0% | 0.5540±0.0027 | 3.85 |
| Nystromformer | 0.3726 | 0.0389 | 41.5% | 0.5998±0.0058 | 6.80 |
| LSSO-r16 | 0.3427 | 0.0249 | 62.5% | 0.5479±0.0014 | 4.03 |
| LSSO-r32 | 0.3703 | 0.0385 | 42.1% | 0.5538±0.0018 | 4.99 |

Full table: [`paper_results/cifar100_cv_main/summary.tsv`](paper_results/cifar100_cv_main/summary.tsv).

### Rank Pruning

Inference-time rank pruning is evaluated on trained LSSO-r32 retrieval
checkpoints. Keeping rank 16 is close to lossless in the current retrieval
setting while reducing compact mixer MAC ratio to about 0.79; keeping rank 8 is
mostly usable but begins to trade accuracy for compression.

Full table: [`paper_results/rank_pruning/summary.tsv`](paper_results/rank_pruning/summary.tsv).

### Checkpoints

Download checkpoints from the release and extract the needed archive:

```bash
tar -xf retrieval_main_fiqa_checkpoints.tar
sha256sum -c SHA256SUMS
```

Release assets are split by experiment group and dataset:

```text
retrieval_main_fiqa_checkpoints.tar
retrieval_main_nfcorpus_checkpoints.tar
retrieval_main_scifact_checkpoints.tar
retrieval_ablation_fiqa_checkpoints.tar
retrieval_ablation_scifact_checkpoints.tar
cifar100_cv_main_checkpoints.tar
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
