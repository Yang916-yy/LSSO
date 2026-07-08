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

For the experimental causal Triton forward path on Linux/WSL CUDA
environments:

```bash
python -m pip install -e ".[triton]" --no-build-isolation
```

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
from lsso import LSSO, RoPELSSO, lsso
```

`LSSO` is the module form. The functional API is useful when integrating LSSO
into custom model code or kernel experiments:

```python
from lsso import lsso

# U: [B, H, N, r], C: [B, H, N, dh]
# mu/gamma: [H] or broadcastable to [B, H, 1, 1]
Y = lsso(U, C, mu, gamma)
```

`RoPELSSO` is the v2 module. It applies rotary position phases to the low-rank
solve basis `U` before calling the same LSSO solve:

```python
from lsso import RoPELSSO

# Bidirectional Rank-RoPE LSSO
mixer = RoPELSSO(dim=256, num_heads=8, rank=32)

# Causal prefix Rank-RoPE LSSO
mixer = RoPELSSO(dim=256, num_heads=8, rank=32, causal=True, causal_chunk_size=256)
```

The rank must be even because RoPE rotates rank channels in pairs. Optional
`position_ids` can be passed at forward time for offset or packed-sequence
experiments:

```python
y = mixer(x, position_ids=position_ids)
```

For simple solve-state cache experiments, use the generic S/P cache helpers:

```python
from lsso import (
    apply_rank_rope,
    update_solve_state,
    read_solve_state,
)

# For RoPE-LSSO, rotate U first. Plain LSSO can pass U directly.
U_tilde = apply_rank_rope(U, position_ids)
cache = update_solve_state(None, U_tilde[:, :, :1], C[:, :, :1])
cache = update_solve_state(cache, U_tilde[:, :, 1:2], C[:, :, 1:2])
y_t = read_solve_state(U_tilde[:, :, 1:2], C[:, :, 1:2], cache, mu, gamma)
```

The cache stores only the two low-rank statistics needed by the solve:

```text
S = sum_i U_i^T U_i
P = sum_i U_i^T C_i
```

For RoPE-LSSO, the same cache is used after the rank basis has been rotated,
so `S = sum_i U_tilde_i^T U_tilde_i` and `P = sum_i U_tilde_i^T C_i`.

Useful module arguments:

```text
rank: low-rank solve size, usually 16 or 32
gamma_max: maximum global correction strength, default 0.3
theta_gamma_init: gamma initialization, default -4.0
normalize_u: RMS-normalize U for stability
no_global: ablation path with gamma = 0
causal: prefix low-rank causal mode for experiments
causal_exclusive: use tokens < i instead of tokens <= i in causal correction
causal_chunk_size: optional chunk size for FlashAttention-style prefix scans
causal_backend: "torch" or experimental causal-only "triton"
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
low-rank solve/correction: H*N*r^2 + 2*N*r*D + D*r^2 + H*r^3
```

The `D*r^2` term accounts for applying the small Cholesky factors to all
per-head right-hand sides; `H*r^3` is a conservative MAC-equivalent estimate
for the factorizations.

This is most attractive when `r << N`. In the current paper tables, MACs are
reported as **mixer-only MACs** so the effect of replacing the token mixer is
visible without being diluted by FFN cost.

## Experimental Causal Prefix Mode

The core module also exposes an experimental causal path:

```python
mixer = LSSO(dim=256, num_heads=8, rank=16, causal=True)

# Memory-friendlier prototype: scan chunks and carry only S/P prefix states
# between chunks instead of materializing full-sequence prefix tensors.
mixer = LSSO(dim=256, num_heads=8, rank=16, causal=True, causal_chunk_size=128)

# Optional Triton forward path for causal prefix experiments.
mixer = LSSO(
    dim=256,
    num_heads=8,
    rank=16,
    causal=True,
    causal_chunk_size=256,
    causal_backend="triton",
)
```

This does not apply an explicit `N x N` triangular mask. Instead, token `i`
uses prefix low-rank statistics:

```text
S_i = sum_{j<=i} u_j^T u_j
P_i = sum_{j<=i} u_j^T c_j
y_i = (c_i - gamma/mu * u_i (I + gamma/mu * S_i)^-1 P_i) / mu
```

Training/prefill can therefore be written as prefix sums plus batched small
`rank x rank` solves. Autoregressive decoding can cache `S_t` and `P_t`.

The optional `causal_chunk_size` path is a PyTorch prototype of a
FlashAttention-style implementation: each chunk forms local prefix statistics,
adds the running low-rank state from previous chunks, solves the small systems,
and writes the output chunk. It reduces the explicit prefix-tensor footprint.

The optional `causal_backend="triton"` path is causal-only and experimental.
Its forward kernel maintains the inverse prefix state with the
Sherman-Morrison update, avoiding a fresh small solve at every token. In
forward-only tests this substantially reduces the prefix-state memory footprint
and can be faster than the materialized PyTorch prefix path for long sequences.
Training still uses a PyTorch recomputation fallback in backward, so the Triton
path should be treated as a forward/prefill kernel prototype rather than a
complete fused training kernel. The causal path is intended for kernel and
causal-model experiments; the paper's main claims still target bidirectional
encoders.

## Paper Experiment Results

Completed experiments are organized under [`paper_results/`](paper_results/).
The versioned theory-and-experiments preprint is available under
[`paper/`](paper/), with the compiled PDF at
[`paper/LSSO_Learnable_Low-Rank_Sylvester_Solves_for_Efficient_Bidirectional_Encoders_v1.pdf`](paper/LSSO_Learnable_Low-Rank_Sylvester_Solves_for_Efficient_Bidirectional_Encoders_v1.pdf).
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
| FIQA | LSSO-r16 | 13.54 | 0.714 | 66.8% | 0.2258+/-0.0125 | 0.1126+/-0.0061 | 676.0 |
| FIQA | LSSO-r32 | 13.80 | 0.910 | 57.6% | 0.2387+/-0.0183 | 0.1150+/-0.0035 | 605.7 |
| NFCorpus | MHA | 14.33 | 2.147 | 0.0% | 0.5439+/-0.0187 | 0.3809+/-0.0145 | 599.3 |
| NFCorpus | Nystromformer | 14.33 | 1.301 | 39.4% | 0.5841+/-0.0125 | 0.4323+/-0.0132 | 376.1 |
| NFCorpus | LSSO-r16 | 13.54 | 0.714 | 66.8% | 0.5686+/-0.0125 | 0.4121+/-0.0036 | 673.2 |
| NFCorpus | LSSO-r32 | 13.80 | 0.910 | 57.6% | 0.5841+/-0.0036 | 0.4333+/-0.0220 | 599.9 |
| SciFact | MHA | 14.33 | 2.147 | 0.0% | 0.6789+/-0.0190 | 0.5994+/-0.0138 | 581.7 |
| SciFact | Nystromformer | 14.33 | 1.301 | 39.4% | 0.7267+/-0.0133 | 0.6313+/-0.0033 | 366.3 |
| SciFact | LSSO-r16 | 13.54 | 0.714 | 66.8% | 0.7022+/-0.0107 | 0.6214+/-0.0119 | 654.3 |
| SciFact | LSSO-r32 | 13.80 | 0.910 | 57.6% | 0.7089+/-0.0117 | 0.6247+/-0.0080 | 586.6 |

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
| macro avg | LSSO-r16 | 13.53 | 0.714 | 66.8% | 0.22973 | 0.35889 | 0.29382 |
| macro avg | LSSO-r32 | 13.80 | 0.910 | 57.6% | 0.22940 | 0.36367 | 0.29858 |

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
| CIFAR-100 | LSSO-r16 | 0.3427 | 0.0250 | 62.4% | 0.5479+/-0.0014 |
| CIFAR-100 | LSSO-r32 | 0.3703 | 0.0388 | 41.7% | 0.5538+/-0.0018 |

ImageNet-100 uses one seed with image size 224, patch size 8, `dim=256`,
`depth=8`, `heads=8`, RandAugment, Mixup=0.8, CutMix=1.0, label smoothing
0.1, and bf16 AMP. As in the retrieval and CIFAR-100 tables, mixer MACs sum
over every encoder layer rather than reporting a single block.

| Dataset | Model | Params (M) | Mixer MACs (G) | Best epoch | Top-1 | Top-5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ImageNet-100 | MHA | 6.59 | 4.162 | 119 | 82.26 | 95.54 |
| ImageNet-100 | LSSO-r16 | 5.80 | 1.093 | 118 | 80.54 | 94.88 |
| ImageNet-100 | LSSO-r32 | 6.06 | 1.391 | 120 | 80.58 | 94.76 |

Full tables:
[`paper_results/cifar100_cv_main/summary.tsv`](paper_results/cifar100_cv_main/summary.tsv) and
[`paper_results/imagenet100_cv_main/summary.tsv`](paper_results/imagenet100_cv_main/summary.tsv).

The LSSO-r32 ImageNet-100 result above is the corrected controlled run. The
superseded run accidentally used Mixup=0 and CutMix=0 while MHA and LSSO-r16
used Mixup=0.8 and CutMix=1.0.

### Latent Diffusion Boundary Experiment

The one-seed ImageNet-100 latent diffusion experiment uses cached VAE latent
means, 784 tokens, `dim=384`, `depth=8`, `heads=8`, and 50 training epochs.
The table reports noise-prediction validation MSE, not FID or perceptual sample
quality.

| Model | Params (M) | Mixer MACs (G) | Save vs MHA | Best epoch | Best val MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| MHA | 22.16 | 7.476 | 0.0% | 23 | 0.148047 |
| Nystromformer | 22.17 | 3.997 | 46.5% | 23 | 0.144175 |
| LSSO-r16 | 20.19 | 2.249 | 69.9% | 23 | 0.159607 |
| LSSO-r32 | 20.58 | 2.677 | 64.2% | 23 | 0.158541 |

Nystromformer has the lowest validation MSE here, but takes about twice the
wall time per epoch of MHA. LSSO retains most of the MHA optimization quality
with substantially lower theoretical mixer MACs. This is treated as a boundary
result until FID/KID and controlled sample grids are available.

Full table:
[`paper_results/diffusion_imagenet100/summary.tsv`](paper_results/diffusion_imagenet100/summary.tsv).

### Rank Pruning

Inference-time rank pruning is evaluated on trained LSSO-r32 retrieval
checkpoints. The table reports theoretical mixer MAC savings for a compact
exported rank after pruning, not the temporary dynamic-mask implementation used
to score the checkpoints. Keeping rank 16 is close to lossless in the current
retrieval setting while reducing theoretical mixer MACs by about 21%; keeping
rank 8 saves about 31% and begins to trade accuracy for compression.

Full table: [`paper_results/rank_pruning/summary.tsv`](paper_results/rank_pruning/summary.tsv).

### Sequence Scaling

A single-layer bf16 benchmark compares MHA, official Nystromformer,
LSSO-r16, and LSSO-r32 at sequence lengths from 128 to 2048 with
`batch=8`, `dim=256`, and `heads=8`.

At `N=2048`, LSSO-r16 uses 86.7% fewer theoretical mixer MACs than MHA and
reduces forward+backward time from 2.61 ms to 1.88 ms. LSSO-r32 uses 83.1%
fewer MACs and takes 1.95 ms. Optimized PyTorch MHA uses Flash/SDPA, so peak
allocated memory remains close; this benchmark supports the arithmetic and
long-sequence latency claim rather than claiming universal memory savings.

Figure and raw data:
[`paper_results/sequence_scaling/`](paper_results/sequence_scaling/).

### Operator Diagnostics

Trained LSSO-r32 checkpoints were evaluated layer by layer on FIQA, NFCorpus,
SciFact, and CIFAR-100. All tasks retain a nonzero global correction, and the
learned effective rank is task dependent: the mean is about 3.77 on FIQA,
4.36 on NFCorpus, 20.81 on SciFact, and 2.49 on CIFAR-100.

Combined figure and layer-level tables:
[`paper_results/operator_diagnostics/`](paper_results/operator_diagnostics/).

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
diffusion_imagenet100_checkpoints.tar
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
