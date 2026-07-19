# Retired CUDA compatibility paths

This directory records CUDA paths removed from the maintained extension after
the Trace-normalized operator became the default.

Retired paths:

- `solve_spd_bpb`: experimental 2/4-systems-per-CTA cuSolverDx schedules;
- legacy non-Trace `stats_solve_spd` and `stats_solve_readout` fusion;
- legacy masked non-Trace statistics/solve fusion;
- scalar all-in-one LSSO backward;
- rotary-on-load Trace fusion.

These paths were either slower than the maintained dispatcher, disabled by
default, or only served the historical `(mu, gamma)` / token-RMS formulation.
Their numerical behavior remains available through PyTorch compact statistics
plus the maintained bucketed MathDx solve. The archived benchmarks and
pre-audit test/document snapshots preserve the old experimental entry points
for historical comparison; they are not expected to run against the slim
current extension without restoring the corresponding operators from
repository history. Retired CUDA templates are intentionally absent from the
shipping translation unit.

The maintained native surface is intentionally limited to:

1. bucketed FP32 SPD solve for arbitrary ranks through 64;
2. masked/null-mask Trace statistics + solve + compact readout;
3. effective-strength adjoint readout;
4. dual Tensor-Core backward statistics and `grad_U` radial epilogue;
5. Rank Rotary and its inverse.
