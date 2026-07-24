# Retired solve-parameterization ablations

These paths were removed from the supported module constructors when the
Trace/log-alpha interface was frozen.

- `token_rms_reference.py` preserves the former per-token RMS basis
  normalization used only as a PyTorch ablation.
- `fixed_gain_reference.py` preserves the former fixed-gain/output-folding
  experiment.
- `solve_parameterization_test.py` records the corresponding historical test
  contract.
- `sweep_trace_alpha_init_cifar100.py` is the retired initialization search.
- `checkpoint_regularization_state.md` records the removed gain-reference and
  alpha-saturation checkpoint contract and distinguishes it from the retained
  general refinement/resume state machine.

The maintained operator uses sample/head Trace normalization, learnable
log-gain and learnable unbounded log-alpha. Alpha initialization is fixed
internally to one and is intentionally absent from the public constructor.
Existing checkpoints remain compatible because they store `theta_alpha`.

Nothing in this directory is imported, packaged, or exercised by CI.
