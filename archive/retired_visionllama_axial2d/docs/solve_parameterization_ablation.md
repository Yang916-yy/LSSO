# Solve scalar parameterization ablation

## Question

The historical per-head coefficients

\[
\mu_h=\operatorname{softplus}(\theta_{\mu,h})+\varepsilon,
\qquad
\gamma_h=\gamma_{\max}\sigma(\theta_{\gamma,h})
\]

couple the output gain and relative solve strength because

\[
g_h=\mu_h^{-1},\qquad \alpha_h=\gamma_h/\mu_h.
\]

The alternative implementation learns these two quantities independently:

\[
g_h=\exp(\theta_{g,h}),\qquad
\alpha_h=\alpha_{\max}\sigma(\theta_{\alpha,h}),
\]

and passes the algebraically equivalent coefficients
\(\mu_h=1/g_h\) and \(\gamma_h=\alpha_h/g_h\) to the existing solve kernel.
Its initialization is matched exactly to the historical operator.

## CIFAR-100 screening experiment

- Backbone: VisionLLaMA-S
- Mixer: RRLSSO, rank 32
- Dataset and recipe: existing CIFAR-100 recipe
- Epochs: 10
- Seed: 1234
- Shared settings: batch 128, BF16, AdamW, identical augmentation and schedule
- Direct parameterization bound: \(\alpha_{\max}=2\)

| Epoch | legacy \((\mu,\gamma)\) | direct \((g,\alpha)\) | fixed \(g=1\) |
|---:|---:|---:|---:|
| 1 | 14.46 | 14.42 | 14.45 |
| 2 | 23.43 | 23.45 | 23.13 |
| 3 | 30.64 | 30.66 | 30.05 |
| 4 | 35.90 | 35.92 | 34.84 |
| 5 | 40.94 | 41.53 | 40.87 |
| 6 | 44.15 | 44.18 | 43.64 |
| 7 | 48.46 | 48.30 | 48.02 |
| 8 | 50.98 | 50.65 | 49.96 |
| 9 | 51.85 | 51.71 | 51.45 |
| 10 | **52.29** | **52.24** | **51.86** |

Mean validation accuracy over the ten checkpoints is 39.310%, 39.306%, and
38.827%, respectively. The fixed-gain run absorbs the matched initial gain into
the corresponding input columns of \(W_O\), so all three models implement the
same function at step zero. It removes only 72 scalars from the 12-layer,
6-head model.

The final learned mean relative strengths are 1.1757, 1.1541, and 1.1535. Peak
allocated memory is the same (2.22 GiB), and total measured epoch time differs
by about 1.6%, within ordinary run-to-run systems noise. Parameter counts are
18,681,076 for both learned-gain models and 18,681,004 for fixed gain.

## Decision

This one-seed, ten-epoch screen finds no measurable accuracy advantage or
penalty from the direct learned coordinates. Fixing \(g=1\) is consistently
weaker in the later part of the run and finishes 0.43 percentage points below
the legacy model, despite exact function matching at initialization. This is a
screening result rather than a multi-seed significance claim, but it provides
no reason to remove the learnable gain. Keep `mu_gamma` as the default for
checkpoint compatibility, expose `gain_alpha` as an opt-in research setting,
and retain a learnable per-head gain in the proposed model.

Raw runs:

- `runs/cifar100_solve_parameterization_10ep_mu_gamma/rrlsso`
- `runs/cifar100_solve_parameterization_10ep_gain_alpha/rrlsso`
- `runs/cifar100_solve_parameterization_10ep_fixed_gain_alpha/rrlsso`
