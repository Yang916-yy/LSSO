# Global correction scale

Bidirectional LSSO now uses per-sample, per-relation-system Gram-trace
normalization by default. For a raw basis `Z [B,G,N,r]`, it sets

```text
tau = r * length_reference
s       = tau / (||Z||_F^2 + eps * r * valid_length)
alpha_e = alpha * s
```

over valid tokens. This fixes the total Gram energy while preserving relative
token radii. The implementation neither materializes a normalized basis nor
performs a separate energy pass: it obtains `||Z||_F^2` from
`trace(Z.T @ Z)` after the required Woodbury Gram statistic has been formed.
The old row-wise RMS basis remains available as the PyTorch-only
`basis_normalization="token_rms"` ablation; its CUDA path is retired.

With `eps=0`, global rescaling of `Z` leaves the layer invariant and the exact
backward is orthogonal to the single global radial direction. The custom
backward includes the derivative of `s(Z)`; omitting it breaks that invariant.

The canonical learnable scalars are per-head output gain `g` and solve strength
`alpha`. The frozen default is the deliberately simple `g=1`, `alpha=1.2`,
with `alpha_max=3.0`. Controlled ten-epoch CIFAR-100 screens found a broad
operating plateau: `alpha_init` from roughly 1.08 to 1.60 and `g_init` from
roughly 0.75 to 1.44 trained comparably. A separate normalization comparison
at matched initialization reached 50.46% validation accuracy for trace
normalization and 48.39% for token RMS. Increasing the ceiling from 2 to 3
changed ten-epoch accuracy by only 0.18 percentage points while keeping the
initial sigmoid coordinate away from saturation and leaving room for long-run
head specialization.

This is a robust starting interval, not a universal optimum. New image
resolutions, ranks, or downstream tasks should first use this default and then
run a small controlled bracket around it. The maintained initialization search
is `experiments/search/sweep_trace_alpha_init_cifar100.py`; the superseded
RMS/length-normalization diagnostic is preserved under
`archive/retired_auxiliary_benchmarks/experiments/`.
