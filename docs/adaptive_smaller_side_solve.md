# Dimension-adaptive Trace solve

For each sample/head, the Trace-normalized operator now chooses the
mathematically smaller exact system:

- `Nvalid >= rank`: rank-space balanced Woodbury;
- `Nvalid < rank`: token-space balanced primal solve.

With `delta=1, eta=alpha` for `alpha<1`, and
`delta=1/alpha, eta=1` otherwise, the token system is

```text
(delta I + eta U U^T) Y = delta C.
```

It is algebraically identical to `(I + alpha U U^T)^-1 C`.  Consequently,
the SPD energy interpretation, uniqueness, firm non-expansiveness, implicit
fixed-point interpretation, and Trace-normalization theory are unchanged.
The smaller-side choice removes artificial `delta` eigenvalues caused only
by forming a rank-space Gram wider than the valid token matrix.

The maintained module interface passes `theta_alpha` directly.  The fused
Trace kernel selects the balanced form using
`theta_alpha + log(scale_squared)` and computes only the bounded coefficient
on that side.  Its analytic backward returns `dL/d theta_alpha` directly.
This keeps nonzero finite gradients through `theta_alpha=80` in BF16 tests
without first materializing `exp(theta_alpha)`.

## Validation

On the local SM120 GPU:

- 253 tests pass;
- FP64 direct-resolvent equality covers `N<r`, `N=r`, and `N>r`;
- `gradcheck` and `gradgradcheck` pass on the primal/log-alpha path;
- mixed masked batches with NaN-poisoned padding have zero forward and
  backward leakage;
- FP16/BF16 ranks 16/32/48/64 retain the existing explicit-normalization
  accuracy contract;
- CUDA 13 builds complete for SM80, SM89, SM90, and SM120.  SM80/89/90 are
  compile validation; runtime validation was performed on SM120.

Run the reproducible benchmark with:

```bash
python benchmarks/benchmark_adaptive_trace.py --comprehensive
```

The portable token-side path eliminates `U.T @ C` and the Woodbury readout,
but its several CUDA launches remain slower than the single-CTA rank kernel:
on an RTX 5070 Ti, short-side forward latency is currently 2.4--8.9x the
forced fused-rank comparator and training latency is 1.2--1.6x.  Replacing
`torch.linalg.solve` with the no-check SPD backend reduced representative
forward latency from about 0.56 ms to 0.24 ms.  A future single-CTA primal
kernel is therefore a kernel-fusion optimization, not a mathematical or
correctness prerequisite.  Normal `N>=r` workloads retain the fused path and
show no material regression in the benchmark.
