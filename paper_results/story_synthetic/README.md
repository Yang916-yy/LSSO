# Story Synthetic Experiments

This folder contains a first-pass synthetic experiment meant to test the paper
story that LSSO rank is a controllable global-relation capacity knob rather
than only an efficiency hyperparameter.

## Intrinsic-Rank Solve Task

The task generates labels from a frozen positive-shifted low-rank solve teacher.
Inputs are continuous token sequences. For each intrinsic rank `k`, the teacher
constructs low-rank relation features `U` and target states `C`, solves a
shifted system, and derives binary labels from the solved representation. The
student models are then trained from scratch under the same encoder scaffold.

Current run:

```text
seed = 1
seq_len = 128
dim = 96
depth = 3
heads = 4
steps = 500
train_samples = 4096
val_samples = 2048
intrinsic_rank in {4, 8, 32}
models = MHA, Nystrom-r32, Performer-r32, LSSO-r4/r8/r16/r32/r64
```

The single-seed result is not yet a paper table. The useful signal is the
directional behavior: for intrinsic rank 32, LSSO-r4 is under-capacity, while
r8/r16/r32 recover substantially; for intrinsic rank 4, small rank is already
competitive. The diagnostics also show that learned effective rank increases as
the configured LSSO rank grows.

## Files

```text
summary.tsv                 raw single-seed table
config.json                 exact run configuration
intrinsic_rank_story.png    rank/accuracy and effective-rank plot
```

The MQAR/associative-recall prototype was also tried, but the current setup
overfits training samples and remains near chance on held-out random mappings.
Those runs are kept under `runs/` for debugging and are not treated as paper
evidence.
