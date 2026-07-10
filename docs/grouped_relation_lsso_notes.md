# Grouped-Relation LSSO/RRLSSO Prototype

This prototype reduces the number of independent relation fields and small SPD
systems without reducing the model's content width.

For `num_heads=H` and `num_relation_groups=G`, each relation group owns one
`U_g: [B, N, r]`, while its assigned content heads are concatenated into
`C_g: [B, N, D/G]`. The group computes

```text
S_g = U_g^T U_g
P_g = U_g^T C_g
K_g = solve(I + gamma_g / mu_g * S_g, P_g)
Y_g = C_g / mu_g - gamma_g / mu_g^2 * U_g K_g
```

`GroupedRRLSSO` applies the same rank-space rotary transform as `RRLSSO` to
`U_g` before building the statistics.

## Compatibility

- `num_relation_groups == num_heads` is exactly the original per-head
  parameterization and can load `LSSO` or `RRLSSO` state dictionaries.
- `num_relation_groups < num_heads` changes the parameter shapes of `w_uc`,
  `theta_mu`, and `theta_gamma`; it is intended for training or conversion
  experiments rather than strict checkpoint loading.
- This first implementation is bidirectional-only and intentionally reuses the
  existing PyTorch `lsso` operator. It validates model quality and tensor
  layout before a fused CUDA implementation.

## Expected Cost Changes

Relative to per-head LSSO, relation-basis projection, Gram construction, and
small factorization costs change from `H` copies to `G` copies. The total
`U^T C` and `U K` work remains approximately `O(N r D)` because all content
channels are retained.

## Benchmark

```powershell
python -m benchmarks.benchmark_grouped_lsso `
  --batch 8 --tokens 512 --dim 256 --heads 8 --rank 32 `
  --groups 1 2 4 --dtype bf16
```

The PyTorch benchmark is a structural signal, not the final kernel target. A
hardware-oriented implementation should fuse statistics, device-side SPD
solve, and readout, and should reuse the forward Cholesky factor in backward.

## Initial RTX 5070 Ti Signal

The first BF16 PyTorch measurements use CUDA events and the existing exact
`lsso` implementation. They do not include a fused CUDA kernel.

```text
shape                             model                G    forward    fwd+bwd
B=8, N=512, D=256, H=8, r=32      RRLSSO              8    0.970 ms   2.932 ms
B=8, N=512, D=256, H=8, r=32      GroupedRRLSSO       1    0.861 ms   2.896 ms
B=2, N=3136, D=512, H=8, r=32     RRLSSO              8    0.805 ms   2.796 ms
B=2, N=3136, D=512, H=8, r=32     GroupedRRLSSO       1    0.722 ms   2.693 ms
B=128, N=65, D=768, H=12, r=32    RRLSSO             12    1.834 ms   4.531 ms
B=128, N=65, D=768, H=12, r=32    GroupedRRLSSO       1    1.257 ms   3.562 ms
B=128, N=65, D=768, H=12, r=32    GroupedRRLSSO       4    1.513 ms   4.031 ms
```

The strongest raw speed signal is currently `G=1`. Intermediate group counts
are still useful quality/throughput trade-off candidates, but their eager
PyTorch timings are not monotonic because fewer systems also create wider
right-hand sides for `torch.linalg.solve_ex`. Accuracy experiments must decide
how much relation-field sharing is acceptable before kernel specialization.

## Initial Five-Epoch Quality Check

ViT-B/4 is trained from scratch on CIFAR-100 with the same seed and schedule as
the existing five-epoch ERF run. These short runs measure early optimization,
not converged accuracy.

```text
model                    G    params    val acc @ 5    steady epoch    peak memory
MHA                      -    85.22 M   17.62%         38.83 s         3.89 GiB
RRLSSO-r32              12    74.59 M   33.91%         48.28 s         3.83 GiB
GroupedRRLSSO-r32        4    72.23 M   33.02%         43.77 s         3.66 GiB
GroupedRRLSSO-r32        1    71.34 M   29.73%         not comparable  3.60 GiB
```

`G=4` retains substantially more of the early-learning quality than `G=1`
while reducing model size and steady eager epoch time by about 9.3% relative
to per-head RRLSSO. It is the current balanced candidate. It is still slower
than optimized MHA at this short `N=65` sequence length. The `G=1` run used
`num_workers=0` after a separate Windows process hit a pinned-allocator
teardown failure; its epoch time is therefore not directly comparable with the
multi-worker runs.
