# Length-normalized global-strength sweep

## Outcome

For strict effective-length normalization (`length_reference=1`), the robust
initial global-strength interval is:

```text
0.85 <= gamma / mu <= 1.15
```

The default is therefore `gamma_max=1.2, theta_gamma_init=0.5`. With
`theta_mu=0`, this starts at `gamma/mu=1.0776`; after the short runs it settled
near `1.10`. This is a robust initialization region, not a claim that one
decimal value is universally optimal.

The former default (`gamma_max=0.3, theta_gamma_init=-4`) starts at
`gamma/mu=0.00778`. Once `U^T U` uses effective-length means, that value is a
near-local ablation. On the G=4 seed-1234 run it reached validation loss
`4.4096` and accuracy `2.49%`, versus `3.5622` and `13.98%` for the new default
under the same short-run protocol.

## Protocol

- CIFAR-100, ViT-B/4: 12 layers, 768 hidden channels, 12 heads, rank 16.
- G=4 Grouped-RRLSSO and full G=12 RRLSSO.
- BF16, MathDx backend, batch size 128.
- Four epochs of 160 training steps (640 updates), then full 10,000-image
  validation each epoch.
- Seeds 1234 and 2025 for the final high-strength bracket.
- `length_normalize=True`, `length_reference=1.0`.

These are controlled optimization short runs for selecting an initialization
region, not converged CIFAR-100 accuracy claims.

## Two-seed aggregate

| Mixer | Initial gamma/mu | Final gamma/mu | Val loss mean (range) | Val accuracy mean (range) |
|---|---:|---:|---:|---:|
| G=4 | 0.866 | 0.904 | 3.5544 (3.5467-3.5622) | 14.49% (14.25-14.72) |
| G=4 | 1.078 | 1.103 | 3.5560 (3.5497-3.5622) | 14.67% (13.98-15.36) |
| G=4 | 1.266 | 1.285 | 3.5721 (3.5558-3.5883) | 13.98% (13.47-14.48) |
| G=12 | 0.654 | 0.693 | 3.6065 (3.6005-3.6125) | 13.90% (13.08-14.72) |
| G=12 | 0.866 | 0.899 | 3.5517 (3.5434-3.5600) | 14.73% (14.65-14.81) |
| G=12 | 1.078 | 1.103 | **3.5257** (3.4784-3.5729) | **14.97%** (14.93-15.00) |
| G=12 | 1.266 | 1.280 | 3.5482 (3.5364-3.5600) | 13.96% (13.81-14.11) |

The G=4 `0.90` and `1.10` loss means differ by only `0.0015`, and their
ordering flips between seeds. G=12 leans toward `1.10`, while `>=1.25` loses
accuracy in both relation configurations. That supports reporting a plateau
of `0.85-1.15` and choosing the directly tested `1.078` initialization rather
than overfitting a single short-run optimum.

The exact midpoint experiment (`theta_gamma_init=0.25`, initial
`gamma/mu=0.973`) was deliberately retained. It scored `3.5838/13.45%` for
G=4 and `3.5049/15.24%` for G=12 at seed 1234, illustrating why interpolation
from a noisy four-epoch curve should not be presented as a precise optimum.

## Wall-clock check

Changing the strength does not change any projection, statistic, factorization,
or solve shape. Averaging epochs 2-4 of matched seed-1234 runs measured
`16.9243 s` at the old G=4 near-local setting and `17.1140 s` at the new
default-strength setting; G=12 measured `18.4883 s` at a low-strength point and
`18.6890 s` at the new default. The roughly 1.1% spread is run-to-run timing
variation, not a new computational term. The separate MathDx training A/B
continues to show the previously measured 10-12% end-to-end acceleration.

## Reproduction

```bash
export LSSO_MATHDX_LIBRARY=/mnt/d/LSSO/build/mathdx-release/lib/lsso_mathdx.so

python experiments/sweep_gamma_strength_cifar100.py \
  --model grouped-rrlsso --relation-groups 4 \
  --gamma-max 1.2 --theta -0.5 0 0.5 1 \
  --epochs 4 --steps-per-epoch 160 --batch-size 128 \
  --warmup-epochs 0.25 --seed 1234 \
  --out-dir runs/gamma_strength_sweep/g4_reproduction

python experiments/sweep_gamma_strength_cifar100.py \
  --model rrlsso --relation-groups 12 \
  --gamma-max 1.2 --theta -0.5 0 0.5 1 \
  --epochs 4 --steps-per-epoch 160 --batch-size 128 \
  --warmup-epochs 0.25 --seed 1234 \
  --out-dir runs/gamma_strength_sweep/g12_reproduction
```

Machine-readable two-seed aggregates are in `aggregate.tsv`; raw epoch metrics
remain under `runs/gamma_strength_sweep/`.

## Length invariance check

`experiments/diagnose_length_normalization.py` repeats one fixed 32-token
sequence up to 1,024 tokens. With the new default strength, mean-normalized
statistics keep both the correction/local norm ratio (`0.34640145`) and the
largest covariance eigenvalue (`2.23532626`) constant. The legacy sum form
moves the ratio from `0.67247397` to `0.70635864`, while its largest eigenvalue
grows exactly 32x (`71.53` to `2288.97`). The full output is in
`length_invariance.tsv`.
