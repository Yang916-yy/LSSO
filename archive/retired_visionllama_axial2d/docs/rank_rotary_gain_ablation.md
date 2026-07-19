# Rank Rotary × head-gain CIFAR-100 ablation

## Question

Does the learnable per-head gain primarily compensate for weak or degenerate
coverage in axial 2-D Rank Rotary?

## Protocol

- VisionLLaMA-S with RRLSSO rank 32
- CIFAR-100, 10 epochs, seed 1234
- BF16, batch 128, identical augmentation and optimizer schedule
- trace normalization with `length_reference=1`
- identical \(g_0=1.4426742275\), \(\alpha_0=1.0776072417\)
- fixed-gain cells absorb \(g_0\) into the corresponding columns of \(W_O\),
  so learned and fixed models are function-identical at step zero

## Results

| Rank Rotary | Gain | Final val. acc. | 10-checkpoint mean | Final train acc. | Val. loss |
|---|---|---:|---:|---:|---:|
| 1-D | learnable | **52.55** | **39.436** | **50.28** | **1.8015** |
| 1-D | fixed | 52.04 | 38.943 | 49.81 | 1.8096 |
| axial 2-D | learnable | 52.24 | 39.306 | 49.74 | 1.8068 |
| axial 2-D | fixed | 51.86 | 38.827 | 48.96 | 1.8234 |

Effects at epoch 10:

- learnable-gain benefit under 1-D: +0.51 percentage points;
- learnable-gain benefit under 2-D: +0.38 percentage points;
- 1-D benefit with learned gain: +0.31 percentage points;
- 1-D benefit with fixed gain: +0.18 percentage points;
- interaction (difference of differences): -0.13 percentage points.

## Interpretation

The screen does not support the claim that learnable gain is specifically
repairing axial 2-D Rank Rotary. Its benefit is at least as large under ordinary
1-D Rank Rotary. The more defensible interpretation is that the redundant
per-head scalar supplies a short optimization and regularization path that
would otherwise have to be reproduced through many entries of \(W_O\).

Axial 2-D remains mildly weaker in both gain settings, consistent with its poor
frequency coverage on an 8x8 patch grid. Because this is a one-seed, short-run
screen, the 0.18--0.31 point rotary differences should be treated as directional
rather than statistically conclusive. Retain learnable gain and use ordinary
Rank Rotary as the current default; keep axial 2-D as an ablation until a
grid-calibrated directional variant is tested.

Raw metrics:

- `runs/cifar100_rank_rotary_gain_ablation_1d_learned/rrlsso/metrics.csv`
- `runs/cifar100_rank_rotary_gain_ablation_1d_fixed/rrlsso/metrics.csv`
- `runs/cifar100_solve_parameterization_10ep_gain_alpha/rrlsso/metrics.csv`
- `runs/cifar100_solve_parameterization_10ep_fixed_gain_alpha/rrlsso/metrics.csv`
