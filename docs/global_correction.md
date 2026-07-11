# Global correction scale

Bidirectional LSSO normalizes the solve basis with effective-length means.
This makes `U^T U` and `U^T C` invariant to padding and repeated-token length,
so the global correction does not drift simply because the sequence is longer.

The controlled CIFAR-100 short-run selection study supports an initialization
region of `0.85 <= gamma / mu <= 1.15` for both full RRLSSO and grouped RRLSSO.
The active default is `gamma_max=1.2` and `theta_gamma_init=0.5`; with
`theta_mu=0`, it initializes near `gamma / mu = 1.08`.

This is a robust starting interval, not a universal optimum. New image
resolutions, ranks, or downstream tasks should first use this default and then
run a small controlled bracket around it. The diagnostic script is
`experiments/diagnose_length_normalization.py`; the sweep driver is
`experiments/sweep_gamma_strength_cifar100.py`.
