# Operator Diagnostics

Layer-wise diagnostics from trained LSSO-r32 checkpoints on FIQA, NFCorpus,
SciFact, and CIFAR-100. Each value is averaged over 8 evaluation batches of 32
samples. Retrieval documents are tokenized to the trained length of 512;
CIFAR-100 uses the patch-2 encoder checkpoint.

The plotted quantities are:

- `gamma_over_mu`: learned global solve strength.
- `correction_ratio`: norm of the global correction divided by the local
  `mu^-1 C` term.
- `effective_rank`: participation-ratio rank of `U^T U`.

| Task | Layers | Mean gamma/mu | Mean correction ratio | Mean effective rank |
| --- | ---: | ---: | ---: | ---: |
| FIQA | 8 | 0.00873 | 0.3256 | 3.77 |
| NFCorpus | 8 | 0.01173 | 0.4946 | 4.36 |
| SciFact | 8 | 0.00779 | 0.4304 | 20.81 |
| CIFAR-100 | 3 | 0.02111 | 0.8014 | 2.49 |

The nonzero correction ratios confirm that trained models use the global solve
rather than collapsing to the local projection. Effective rank is strongly
task dependent: SciFact uses much more of the available rank-32 subspace than
FIQA or NFCorpus, while CIFAR-100 learns a very compact relation field.

Artifacts:

- `operator_diagnostics_summary.png`: combined paper figure.
- `summary.tsv`: per-task mean, minimum, and maximum.
- `*_layer_diagnostics.tsv`: layer-level source values.
- `*_layer_diagnostics.png`: individual task figures.

Reproduce one task and rebuild the combined figure:

```bash
python benchmarks/plot_operator_diagnostics.py \
  --checkpoint path/to/checkpoint.pt \
  --task fiqa \
  --task-type retrieval

python benchmarks/summarize_operator_diagnostics.py
```
