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
`alpha`. Both use log coordinates:

```text
g     = exp(theta_gain)
alpha = exp(theta_alpha)
beta  = 1 / alpha
```

The default `theta_gain=theta_alpha=0` gives `g=alpha=beta=1`. Ignoring
epsilon, trace normalization fixes `trace(U.T @ U)=r`, so the mean compact
eigenvalue is one and the initial resolvent response there is exactly one half.
There is no learned-strength ceiling or dedicated scalar regularizer: positive
alpha already preserves the SPD resolvent, while its unbounded range spans the
continuous path from identity propagation to relation-subspace projection.

The maintained Woodbury implementation uses the reciprocal form

```text
Y = g * (C - U @ solve(beta * I + U.T @ U, U.T @ C))
```

which avoids forming a large `alpha * Gram` term and naturally approaches the
projection limit as `beta` tends to zero.

New image resolutions, ranks, or downstream tasks should first use this
dimensionless default and then inspect the learned per-head alpha distribution.
The maintained initialization search
is `experiments/search/sweep_trace_alpha_init_cifar100.py`; the superseded
RMS/length-normalization diagnostic is preserved under
`archive/retired_auxiliary_benchmarks/experiments/`.
