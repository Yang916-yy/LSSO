# Sequence Experiments

`python -m experiments.train_transformers` is the single PyTorch entrypoint
for current GenomicBenchmarks and LRA results.  It constructs the same learned
absolute-position encoder, residual blocks, MLP, readout, optimizer policy,
and validation-selection rule for `--mixer mha` and `--mixer lsso`.

The LSSO default is the complete DYNAMIC + Rank-Rotary CUDA contract.  STATIC,
ZERO, and Rank-Rotary-off remain reference-only ablations.  Rank-Rotary is an
internal rank-space coordinate transform, so it is used in addition to, never
instead of, the shared learned absolute position embedding.
The CUDA sequence runner uses FP16 AMP; BF16 is rejected on the LSSO fast path
instead of being silently routed through a different numerical contract.

## Data Protocol

- GenomicBenchmarks uses its official train/test partition.  A deterministic
  stratified validation subset is derived only from train with `--split-seed`.
  `--max-length 0` preserves the longest source sequence, including Mouse
  examples; it does not apply a legacy fixed truncation.
- LRA ListOps uses the provided train/validation/test files.  Text and
  Retrieval use UTF-8 bytes plus EOS.  Text derives validation from its train
  split; Retrieval retains its supplied validation file.
- Pathfinder uses the official TFDS 4.0.1 `hard` record order before its
  80/10/10 slice. TFDS's `BeamWriter` sorts `md5(b"hard" + example_key)`;
  the runner reproduces that ordering as
  `tfds-v4.0.1-hard-md5-order-v1` and records index fingerprints for every
  derived partition.

The data decisions follow the archived official
[Long Range Arena repository](https://github.com/google-research/long-range-arena)
and the official
[GenomicBenchmarks repository](https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks).
The former is archived, so each run records local source metadata and cached
dataset fingerprints rather than relying on a mutable download endpoint.

## Run Contract

Every run writes its expanded configuration, data provenance, split
fingerprints, runtime information, model parameter count, and configuration
digest to its output directory.  Checkpoints refuse a configuration mismatch on
explicit `--resume`; a prior run is never resumed implicitly.  `best.pt` is
chosen from validation accuracy, with validation loss breaking exact ties, and
the benchmark test split is evaluated only once after that choice. A completed
run records that final result in `last.pt`; an explicit resume returns it
without another training or test pass. Early-stop deltas only decide whether a
run is stale; they never suppress a strictly better validation checkpoint.

LRA defaults preserve the requested rates: `5e-4` for ListOps, Text, and
Retrieval, `2e-4` for Pathfinder.  Early stopping cannot trigger before 75% of
configured epochs.  Use `--validation-only` for pilots and `--formal` only
from a clean committed revision; `--formal` also rejects pilot sample and batch
limits.  `experiments/configs/lra.toml` is a shared operator configuration;
the selected `--task` determines its schedule, batch sizes, sequence length,
and patience so task overrides cannot inherit ListOps values.

Formal runs hash each local source file tree or cached Arrow split into recorded
`content_sha256` metadata, so the split fingerprints identify concrete source
contents rather than an absolute path or timestamp. Downloads instead require a
full immutable commit SHA in `--data-revision`; mutable names and tags are
rejected. Packed token caches serialize the first build of each prefix, so
independent seeds may safely share a cache root.
