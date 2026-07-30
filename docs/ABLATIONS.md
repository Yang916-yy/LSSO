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
