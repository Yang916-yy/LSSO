# Ablations

The public configuration exposes exactly two ablation axes.

| Ablation | Configuration | Compact-core behavior |
| --- | --- | --- |
| sample-conditioned coordinates | core_mode=DYNAMIC | R = R0 + Z W_drive / sqrt(n_valid) |
| sample-independent coordinates | core_mode=STATIC | R = R0 |
| zero compact core | core_mode=ZERO | M = 0, with no compact parameters |
| remove Rank-Rotary | rank_rotary=False | relation frame has no rank-space phase rotation |

ZERO changes only the compact core. It retains the relation frame P, content C,
learned complement eta, and the same token lift. No other model variants are
supported.

## Backend and attribution

Native CUDA implements only DYNAMIC with Rank-Rotary enabled. STATIC, ZERO,
and Rank-Rotary-off experiments must request the reference implementation;
there is no implicit fallback. Hold precision, data order, optimizer, training
schedule and model shell fixed when comparing ablations, and report backend
changes separately from accuracy effects.

Rank-Rotary rotates relation features in rank space. Its placement is compatible
with positional mechanisms elsewhere in a model; that compatibility does not
by itself establish a benefit. Current formal complete-model results do not
isolate either ablation axis. See [published-result provenance](../results/README.md).
