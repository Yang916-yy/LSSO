# Sequence Experiments

`python -m experiments.train_transformers` is the single PyTorch entrypoint
for current GenomicBenchmarks and LRA results.  It constructs the same learned
absolute-position encoder, residual blocks, MLP, readout, optimizer policy,
and validation-selection rule for `--mixer mha` and `--mixer lsso`.

The LSSO default is the complete DYNAMIC + Rank-Rotary CUDA contract.  STATIC,
ZERO, and Rank-Rotary-off remain reference-only ablations.  Rank-Rotary is an
internal rank-space coordinate transform, so it is used in addition to, never
instead of, the shared learned absolute position embedding.
The shared learned position table is initialized once from `Normal(0, 0.02)`;
it is not left at PyTorch's default embedding scale.
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
- Generic `--task pathfinder` accepts `--pathfinder-resolution` (32 by
  default). `--task pathx` is the explicit Path-X identity, fixed to
  Pathfinder-128 and recorded as `task: pathx`, `resolution: 128`. Both use
  the official TFDS 4.0.1 `hard` record order before their 80/10/10 slice.
  TFDS's `BeamWriter` sorts `md5(b"hard" + example_key)`; the runner
  reproduces that ordering as `tfds-v4.0.1-hard-md5-order-v1` and records
  index fingerprints for every derived partition. As in the archived LRA
  Transformer input pipeline, zero-valued background pixels are excluded by
  the attention/readout mask. Their original row-major raster positions are
  retained for the remaining foreground pixels.

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

LRA uses one task-specific shared shell for the current formal panel.  MHA and
LSSO receive the same input representation, position encoding, readout, width,
depth, heads, MLP, optimizer, schedule, effective batch, checkpoint selection,
and data split.  Only the mixer changes: LSSO is DYNAMIC + Rank-Rotary CUDA;
MHA is PyTorch MHA.  The frozen recipes are:

| Task | Shared shell | Physical x accumulation | Effective batch | LR / epochs |
| --- | --- | ---: | ---: | --- |
| ListOps | `D128/L6/H8`, MLP 4, dropout .1, learned absolute PE + mean | `25 x 2` | 50 | `5e-4` / 40 |
| Text | `D256/L6/H8`, MLP 4, dropout .1, learned absolute PE + mean | `16 x 2` | 32 | `5e-4` / 32 |
| Retrieval | `D128/L6/H8`, MLP 4, dropout .1, learned absolute PE + mean | `16 x 4` | 64 | `5e-4` / 20 |
| Pathfinder-32 | `D256/L6/H4`, MLP 2, dropout 0, factorized row/column learned absolute PE (no PEG) + meanmax | `64 x 2` | 128 | `2e-4` / 200 |

All four use AdamW, weight decay `.01`, 5% linear warmup followed by cosine
decay, and gradient clip `1.0`. Pathfinder uses its official nonzero pixel
mask after the row/column position features are added; LSSO then adds its flat
rank-space Rank-Rotary coordinate transform. Pathfinder's earliest possible stop is epoch 150 with
patience 10; the remaining tasks keep their task defaults with no stop before
75% of the declared budget.  The runner picks the best checkpoint on validation
accuracy (then validation loss), evaluates the held-out test exactly once, and
records the result.

This is the `LSSO shared-shell protocol`, not the official LRA
apples-to-apples Transformer setting.  Official LRA fixes its own Transformer
shape and permits a new model at no more than 10% extra parameters; its
Transformer also uses a different PE and readout.  Published S4 and official
LRA values are therefore external references, not numbers to pool with this
panel.  `experiments/configs/lra.toml` holds only the shared LSSO operator
selection; the chosen `--task` resolves its full frozen recipe without a
generic config silently overriding it.

Path-X remains a fixed data identity but is outside this formal panel.  Its
current defaults are only a local-probe configuration until a separately
declared model search is complete.

Formal runs hash each local source file tree or cached Arrow split into recorded
`content_sha256` metadata, so the split fingerprints identify concrete source
contents rather than an absolute path or timestamp. Downloads instead require a
full immutable commit SHA in `--data-revision`; mutable names and tags are
rejected. Packed token caches serialize the first build of each prefix, so
independent seeds may safely share a cache root.
