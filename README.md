# LSSO

LSSO is the **Learnable Sylvester Solve Operator**, a PyTorch token mixer for
bidirectional encoder-style models.

Instead of routing values with pairwise attention, LSSO learns low-rank global
relation features and solves a positive-shifted linear system:

```text
LSSO(X) = Y W_o
(mu I + gamma U(X) U(X)^T) Y = C(X)
```

The implementation uses the Woodbury form, so the solve is over a small
`rank x rank` system instead of an `N x N` inverse. This makes LSSO a drop-in
candidate for non-causal encoder token mixing, especially when sequence length
or token count makes dense attention expensive.

## Highlights

- **Encoder-oriented token mixer:** designed to replace bidirectional
  self-attention in Transformer-style encoder blocks.
- **Global solve, not value routing:** constructs a learned low-rank global
  relation field `U U^T`, then solves for token states consistent with that
  field.
- **Stable positive-shifted system:** `mu > 0`, `gamma >= 0`, and `U U^T`
  keep the solve well behaved.
- **Low-rank Woodbury solve:** the expensive part scales with rank `r`, not
  the full token-token matrix.
- **Small core API:** the installed package exposes only `LSSO` and the
  functional `lsso` operator. Model wrappers and experiments live in
  `examples/` and `paper_results/`.

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

## Quick Start

```python
import torch
from lsso import LSSO

x = torch.randn(2, 128, 256)  # [batch, tokens, dim]
mixer = LSSO(dim=256, num_heads=8, rank=32)
y = mixer(x)
```

Use it in the same slot as a bidirectional token mixer:

```text
X = X + LSSO(LN(X))
X = X + FFN(LN(X))
```

For clean comparisons, keep the rest of the encoder block unchanged:
LayerNorm, FFN, residual path, classification head, and positional encoding.

## Core API

The package intentionally exposes only the core operator:

```python
from lsso import LSSO, lsso
```

`LSSO` is the module form. The functional API is useful when integrating LSSO
into custom model code or kernel experiments:

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

Repository model wrappers for ViT-style classifiers, BERT-style retrieval
encoders, DiT experiments, and baseline comparisons live under
[`examples/models/`](examples/models/). They demonstrate integration patterns
but are not part of the installed core API.

## Recommended Solve Scale

The recommended initialization is:

```python
LSSO(
    dim=dim,
    num_heads=heads,
    rank=rank,
    gamma_max=0.3,
    theta_gamma_init=-4.0,
)
```

With `theta_mu = 0`, this gives roughly:

```text
mu ~= softplus(0) ~= 0.693
gamma ~= 0.3 * sigmoid(-4) ~= 0.0054
gamma / mu ~= 0.0078
```

This is intentionally not tiny. If `gamma_max` is too small or
`theta_gamma_init` is too negative, LSSO can behave almost like the local
`mu^-1 C(X)` projection early in training, weakening the global solve term.
For example, `gamma_max=0.1, theta_gamma_init=-6.0` starts around
`gamma / mu ~= 3.6e-4`, which is usually too conservative for main
experiments.

## Complexity Notes

For an input with sequence length `N`, model dimension `D`, head count `H`, and
rank `r`, the LSSO mixer uses:

```text
U/C projection:  N * D * (H*r + D)
output projection: N * D * D
low-rank solve/correction: H*N*r^2 + 2*N*r*D + H*r^3
```

This is most attractive when `r << N`. In the current paper tables, MACs are
reported as **mixer-only MACs** so the effect of replacing the token mixer is
visible without being diluted by FFN cost.

## Paper Experiment Results

Completed experiments are organized under [`paper_results/`](paper_results/).
The repository tracks lightweight artifacts only: summary tables, manifests,
source notebooks/scripts, and JSONL logs. Model checkpoints are not tracked in
git; they are available from one GitHub Release:

```text
https://github.com/Yang916-yy/LSSO/releases/tag/paper-results-v1
```

See [`paper_results/release_assets.tsv`](paper_results/release_assets.tsv) for
release asset names, sizes, SHA256 checksums, and contents.

### Retrieval Main Table

Random-initialized BERT-style retrieval encoders, 3 seeds, `dim=256`,
`depth=8`, `heads=8`, `max_doc_len=512`, mean pooling. MACs are mixer-only
document-side MACs.

| Dataset | Model | Params (M) | Mixer MACs (G) | Save vs MHA | R@10 | MRR@10 | Samples/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FIQA | MHA | 14.33 | 2.147 | 0.0% | 0.2145+/-0.0135 | 0.0987+/-0.0032 | 597.1 |
| FIQA | Nystromformer | 14.33 | 1.301 | 39.4% | 0.2428+/-0.0039 | 0.1190+/-0.0071 | 375.2 |
| FIQA | LSSO-r16 | 13.54 | 0.713 | 66.8% | 0.2258+/-0.0125 | 0.1126+/-0.0061 | 676.0 |
| FIQA | LSSO-r32 | 13.80 | 0.908 | 57.7% | 0.2387+/-0.0183 | 0.1150+/-0.0035 | 605.7 |
| NFCorpus | MHA | 14.33 | 2.147 | 0.0% | 0.5439+/-0.0187 | 0.3809+/-0.0145 | 599.3 |
| NFCorpus | Nystromformer | 14.33 | 1.301 | 39.4% | 0.5841+/-0.0125 | 0.4323+/-0.0132 | 376.1 |
| NFCorpus | LSSO-r16 | 13.54 | 0.713 | 66.8% | 0.5686+/-0.0125 | 0.4121+/-0.0036 | 673.2 |
| NFCorpus | LSSO-r32 | 13.80 | 0.908 | 57.7% | 0.5841+/-0.0036 | 0.4333+/-0.0220 | 599.9 |
| SciFact | MHA | 14.33 | 2.147 | 0.0% | 0.6789+/-0.0190 | 0.5994+/-0.0138 | 581.7 |
| SciFact | Nystromformer | 14.33 | 1.301 | 39.4% | 0.7267+/-0.0133 | 0.6313+/-0.0033 | 366.3 |
| SciFact | LSSO-r16 | 13.54 | 0.713 | 66.8% | 0.7022+/-0.0107 | 0.6214+/-0.0119 | 654.3 |
| SciFact | LSSO-r32 | 13.80 | 0.908 | 57.7% | 0.7089+/-0.0117 | 0.6247+/-0.0080 | 586.6 |

Full table: [`paper_results/retrieval_main/summary.tsv`](paper_results/retrieval_main/summary.tsv).

### MS MARCO -> BEIR Transfer

Random-initialized BERT-style retrieval encoders are pretrained on MS MARCO and
evaluated zero-shot on BEIR-style datasets across 3 seeds. The evaluated BEIR
sets are FIQA, NFCorpus, SciFact, ArguAna, and TREC-COVID. MACs are analytic
document-side mixer MACs at `doc_len=512`.

| Dataset | Model | Params (M) | Mixer MACs (G) | Save vs MHA | nDCG@10 | Recall@100 | MRR@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| macro avg | MHA | 14.33 | 2.147 | 0.0% | 0.22945 | 0.36061 | 0.30225 |
| macro avg | Nystromformer | 14.32 | 1.270 | 40.9% | 0.20523 | 0.33456 | 0.27115 |
| macro avg | LSSO-r16 | 13.53 | 0.713 | 66.8% | 0.22973 | 0.35889 | 0.29382 |
| macro avg | LSSO-r32 | 13.80 | 0.908 | 57.7% | 0.22940 | 0.36367 | 0.29858 |

Full table: [`paper_results/msmarco_beir_transfer/summary.tsv`](paper_results/msmarco_beir_transfer/summary.tsv).

### Retrieval Ablations

The main ablations use FIQA and SciFact with the same retrieval setup. The
`no-global` variant fixes `gamma=0`, removing the global solve correction.

| Dataset | Variant | R@10 | MRR@10 | gamma/mu | Correction ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| FIQA | no-global r32 | 0.2052+/-0.0137 | 0.0911+/-0.0019 | 0.0000 | 0.0000 |
| FIQA | fixed mu/gamma r32 | 0.2428+/-0.0126 | 0.1136+/-0.0028 | 0.0078 | 0.2130 |
| FIQA | no U RMS norm r32 | 0.2160+/-0.0101 | 0.1007+/-0.0008 | 0.0085 | 0.2050 |
| FIQA | full r8 | 0.2088+/-0.0208 | 0.0923+/-0.0139 | 0.0100 | 0.1059 |
| FIQA | full r4 | 0.2042+/-0.0045 | 0.0979+/-0.0036 | 0.0109 | 0.0742 |
| SciFact | no-global r32 | 0.6767+/-0.0120 | 0.5811+/-0.0058 | 0.0000 | 0.0000 |
| SciFact | fixed mu/gamma r32 | 0.7089+/-0.0102 | 0.6088+/-0.0066 | 0.0078 | 0.2631 |
| SciFact | no U RMS norm r32 | 0.6933+/-0.0133 | 0.5972+/-0.0150 | 0.0078 | 0.0471 |
| SciFact | full r8 | 0.7067+/-0.0173 | 0.6118+/-0.0064 | 0.0078 | 0.1149 |
| SciFact | full r4 | 0.7022+/-0.0069 | 0.6146+/-0.0107 | 0.0078 | 0.0766 |

Full table: [`paper_results/retrieval_ablation/summary.tsv`](paper_results/retrieval_ablation/summary.tsv).

### CV Encoder Tables

CIFAR-100 uses a patch-2 ViT-style encoder, 3 seeds, `dim=96`, `depth=3`,
`heads=6`, CLS pooling, RandAugment(2,9), Mixup=0.2, CutMix=0.5.

| Dataset | Model | Params (M) | Mixer MACs (G) | Save vs MHA | Top-1 |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | MHA | 0.3714 | 0.0665 | 0.0% | 0.5540+/-0.0027 |
| CIFAR-100 | Nystromformer | 0.3726 | 0.0389 | 41.5% | 0.5998+/-0.0058 |
| CIFAR-100 | LSSO-r16 | 0.3427 | 0.0249 | 62.5% | 0.5479+/-0.0014 |
| CIFAR-100 | LSSO-r32 | 0.3703 | 0.0385 | 42.1% | 0.5538+/-0.0018 |

ImageNet-100 uses one seed with image size 224, patch size 8, `dim=256`,
`depth=8`, `heads=8`, RandAugment, Mixup=0.8, CutMix=1.0, label smoothing
0.1, and bf16 AMP.

| Dataset | Model | Params (M) | Mixer MACs (G) | Best epoch | Top-1 | Top-5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ImageNet-100 | MHA | 6.59 | 0.520 | 119 | 82.26 | 95.54 |
| ImageNet-100 | LSSO-r16 | 5.80 | 0.137 | 118 | 80.54 | 94.88 |
| ImageNet-100 | LSSO-r32 | 6.06 | 0.174 | 119 | 78.30 | 92.86 |

Full tables:
[`paper_results/cifar100_cv_main/summary.tsv`](paper_results/cifar100_cv_main/summary.tsv) and
[`paper_results/imagenet100_cv_main/summary.tsv`](paper_results/imagenet100_cv_main/summary.tsv).

### Rank Pruning

Inference-time rank pruning is evaluated on trained LSSO-r32 retrieval
checkpoints. The table reports theoretical mixer MAC savings for a compact
exported rank after pruning, not the temporary dynamic-mask implementation used
to score the checkpoints. Keeping rank 16 is close to lossless in the current
retrieval setting while reducing theoretical mixer MACs by about 21%; keeping
rank 8 saves about 31% and begins to trade accuracy for compression.

Full table: [`paper_results/rank_pruning/summary.tsv`](paper_results/rank_pruning/summary.tsv).

### Checkpoints

Download checkpoints from the relevant release and extract the needed archive:

```bash
sha256sum -c SHA256SUMS
tar -xf retrieval_main_fiqa_checkpoints.tar
```

Release assets are grouped by experiment and dataset in the same release:

```text
retrieval_main_fiqa_checkpoints.tar
retrieval_main_nfcorpus_checkpoints.tar
retrieval_main_scifact_checkpoints.tar
retrieval_ablation_fiqa_checkpoints.tar
retrieval_ablation_scifact_checkpoints.tar
cifar100_cv_main_checkpoints.tar
msmarco_beir_transfer_checkpoints.tar
imagenet100_cv_main_checkpoints.tar
```

## Development

Run the smoke test:

```bash
python -m tests.smoke_test
```

Large local experiment folders such as `data/`, `runs/`, `runs_archive/`,
`release_assets/`, and checkpoint files are ignored by default. This keeps the
installable operator package small while allowing heavier paper artifacts to be
distributed through GitHub Releases.

## License

Apache License 2.0. See [LICENSE](LICENSE).
