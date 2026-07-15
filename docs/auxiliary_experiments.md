# Auxiliary experiments

The formal auxiliary suite is deliberately based on public benchmarks with
stable splits and dense published baseline tables. Its purpose is to test
whether RRLSSO transfers beyond vision without requiring the project to
reproduce every competing operator.

The previous MS MARCO/BEIR retrieval and FLIP AAV protein experiments are no
longer part of the paper plan. Their implementations remain in `experiments/`
as archived prototypes; they must not be included in the formal experiment
matrix or GPU budget unless the plan is explicitly revised.

## Formal suite

### 1. GenomicBenchmarks (8 tasks)

Run all eight official sequence-classification datasets and retain their
provided train/test split:

- `dummy_mouse_enhancers_ensembl`
- `demo_coding_vs_intergenomic_seqs`
- `demo_human_or_worm`
- `human_enhancers_cohn`
- `human_enhancers_ensembl`
- `human_ensembl_regulatory`
- `human_nontata_promoters`
- `human_ocr_ensembl`

The benchmark has no official validation split. Following Caduceus, construct
a deterministic stratified 90/10 train/validation split from the official
training data and leave the test split untouched. Use a single-nucleotide
vocabulary, learned absolute position embeddings, masked pooling, and the same
small bidirectional encoder family for every dataset. The primary metric is
test accuracy; also retain macro F1 and Matthews correlation coefficient for
imbalanced datasets. Report the mean
and standard deviation over three fixed seeds and the mean rank across all
eight tasks.

The main external references are the supervised CNN/Transformer results and
the published HyenaDNA, Caduceus, and later genomic-model tables. Results from
models with genomic pretraining must be placed in a separate **pretrained**
block; a randomly initialized RRLSSO must not be described as an
apples-to-apples comparison with those models.

### 2. Long Range Arena (4 tasks)

Use the official LRA data splits and fixed model configurations for:

- ListOps
- byte-level Text classification
- AAN document Retrieval
- Pathfinder

Sequential CIFAR-10 is omitted because the main paper already contains a much
stronger vision suite. Path-X is omitted because failure-heavy results do not
justify its additional cost.

The implemented default is a modern wide-and-shallow PyTorch protocol: width
256, depth 2, 8 heads, RRLSSO rank 32, AdamW, learned absolute position
embeddings, and the long sequence
lengths commonly used by later PyTorch LRA repositories (2048/4096/4000/1024).
It is deliberately labelled `native-pytorch-reported-baseline`, not an exact
reproduction of Google's archived JAX/Flax recipe. The original recipe varies
width and schedule by task (for example, ListOps uses width 512 while AAN uses
width 128) and uses a different optimizer schedule. The runners expose every
model and optimizer argument, so an exact-width controlled rerun can be made,
but numbers from the compact default must not be presented as belonging to the
official LRA apples-to-apples track.

When sample limits are used for smoke tests or scale probes, all three runners
take deterministic stratified subsets rather than the first rows in dataset
order. The requested limits, actual split sizes, batch sizes, and worker count
are persisted in `config.json`.

Published Transformer and efficient-attention results from the official LRA
table, together with S4/S5 and MEGA results, may be cited rather than rerun only
when the official configuration is matched exactly. LRA is retained as a
standardized operator comparison, not treated as conclusive proof of genuine
long-range reasoning.

### 3. UEA-30 multivariate time-series classification

Run the complete 30-dataset UEA multivariate archive using the official
training and test splits. Do not select datasets after observing results.
Each input channel is projected to the shared model width; padding masks must
exclude missing/padded timesteps from both the structured solve and pooling.

Use accuracy per dataset and report:

- mean rank over all 30 datasets;
- average accuracy as a secondary summary;
- wins/ties/losses against each reference method;
- a critical-difference diagram when the number of complete methods permits;
- peak memory and throughput on representative short, medium, and long series.

The reference table should include standard published results for DTW,
ROCKET-family methods, InceptionTime, ConvTran, and a Transformer baseline.
Published numbers using the same archive version, official split, and metric
may appear in a clearly labelled **reported baselines** block even when their
method-specific preprocessing differs. They are not controlled reproductions.
Results based on resampled or altered splits are contextual references only.

## Minimum self-run policy

The main CV experiments remain the controlled MHA/LSSO/RRLSSO comparison.
The auxiliary suite minimizes new training as follows:

1. Run RRLSSO on every task in the three formal suites.
2. Use published baselines wherever the protocol is exactly compatible.
3. Run one internal MHA anchor per suite only if needed to validate that the
   local data pipeline reproduces the published protocol. Do not run the full
   three-mixer grid by default.
4. Run LSSO auxiliary baselines only when a reviewer-facing question cannot be
   answered by the main CV ablation.

This policy supports a claim of cross-domain applicability. It does not by
itself support a claim that RRLSSO is state of the art in genomics, language,
or time-series classification.

## Reporting and citation rules

- Mark locally reproduced results normally and imported results with a dagger
  (`†`) plus the source paper in the table caption.
- Separate locally controlled results, reported standard-split baselines, and
  pretrained baselines into distinct table blocks.
- Never mix pretrained and from-scratch models in one rank calculation.
- Record dataset version, split, preprocessing, model parameters, seed,
  training steps, selected checkpoint, wall-clock time, and GPU-hours.
- Select checkpoints using validation data only; never tune on the official
  test split.
- Report all planned tasks and seeds, including failures and OOMs.
- The paper's generality statement is based on coverage across discrete DNA,
  byte-level language/retrieval/reasoning, and continuous multivariate signals,
  not on cherry-picked individual wins.

## Installation

From the repository root:

```bash
pip install -e ".[sequence]"
```

`datasets` loads the GenomicBenchmarks author mirrors and IMDB, `aeon`
downloads the official UEA train/test archives, and Pillow reads Pathfinder.
The optional `genomic-benchmarks` package is only a fallback and is not needed
when the Hugging Face mirrors are available.

## Runnable entry points

Each task runner saves `last.pt` and `best.pt` atomically, resumes by default,
uses only a validation split for checkpoint selection, and writes
`config.json`, `metrics.jsonl`, and `test_metrics.json`.

### RRLSSO-DNA competitive-model program

`run_rrlsso_dna_program.py` enforces a validation-gated four-stage protocol:

1. screen reverse-complement augmentation/evaluation and mean, max, or
   mean-max pooling on Human-vs-Worm, Non-TATA, OCR, and Cohn;
2. freeze the selected recipe and compare Tiny (128x2, rank 16), Small
   (192x2, rank 32), and Base (256x2, rank 32) on validation data only;
3. compare Base against a 256x4 candidate and Base with a residual depthwise
   kernel-7 local motif stem, using seed 0 and validation data only;
4. write an immutable `frozen_config.json`, then evaluate the selected
   RRLSSO-DNA architecture on
   all eight test splits for seeds 0/1/2.

The next GenomicBenchmarks validation gates are deliberately single-seed:

1. complete the existing Tiny/Small/Base validation scale curves with seed 0
   only;
2. using seed 0 only, evaluate a 256x4 RRLSSO candidate and a lightweight local
   motif stem (token embedding followed by a residual depthwise 1-D convolution
   and pointwise projection);
3. select and freeze the architecture using validation results only. Do not
   inspect the official test split and do not add repeated seeds during these
   two exploratory gates;
4. spend the multi-seed budget only after the final model and recipe have been
   frozen for the eight-task formal run.

The local motif stem is an architecture candidate, not an RRLSSO replacement.
If retained, controlled mixer comparisons must give the identical stem to MHA,
LSSO, and RRLSSO.

After selecting Base+motif-7, run the low-budget HyenaDNA-flavored recipe
transfer with:

```bash
python experiments/run_genomic_recipe_probe.py
```

It changes only architecture-agnostic training choices on Non-TATA and OCR:
100 epochs, batch 128, LR 6e-4, 1% warmup, a 0.1 cosine floor, embedding
dropout 0.1, and zero residual dropout. RRLSSO keeps weight decay 0.01 instead
of copying Hyena-specific layer decay.

The first two stages pass `--validation-only`; those run directories contain
`validation_metrics.json` and cannot contain `test_metrics.json`. The launcher
runs one child at a time and resumes completed stages automatically:

```bash
python experiments/run_rrlsso_dna_program.py \
  --output-root runs/rrlsso_dna_program \
  --scale-seeds 0 --formal-seeds 0 1 2 --workers 4
```

Checkpoint state includes Python, NumPy, PyTorch, CUDA, DataLoader-generator,
and length-bucket sampler state. Pathfinder uses a fixed `--split-seed`
(default 0) independent of the model `--seed`, so repeated model seeds always
share the same train/validation/test partition. Use `--max-parameters` to make a
paper protocol fail early when a model exceeds its declared parameter budget.

Smoke one dataset from each suite:

```bash
python experiments/genomic_benchmarks.py \
  --dataset dummy_mouse_enhancers_ensembl --epochs 1 \
  --max-train-samples 64 --max-eval-samples 64 --max-train-batches 2

python experiments/lra_benchmark.py --task listops --epochs 1 \
  --max-train-samples 64 --max-eval-samples 64 --max-train-batches 2

python experiments/uea_benchmark.py --dataset BasicMotions --epochs 1 \
  --max-train-batches 2 --workers 0
```

Run the complete RRLSSO matrix sequentially for seeds 0/1/2:

```bash
python experiments/run_sequence_benchmarks.py \
  --suite all --mixer rrlsso --seeds 0 1 2 --workers 4
```

The launcher never runs two jobs simultaneously. A directory containing
`test_metrics.json` is skipped; an incomplete directory resumes from
`last.pt`. A failed child writes `failed.json` and the remaining matrix
continues; add `--fail-fast` for interactive debugging. Add arguments after
`--` to forward them to every child, for
example `-- --dim 192 --depth 6`.

To preview commands without training:

```bash
python experiments/run_sequence_benchmarks.py --suite genomic --dry-run
```

## LRA data layout

The original Google Cloud `lra_release.gz` link currently returns HTTP 403,
so the code does not silently substitute a different dataset. Put the official
or S4-compatible extracted data under `data/lra`:

```text
data/lra/
  listops/basic_train.tsv
  listops/basic_val.tsv
  listops/basic_test.tsv
  aan/new_aan_pairs.train.tsv
  aan/new_aan_pairs.eval.tsv
  aan/new_aan_pairs.test.tsv
  pathfinder/pathfinder32/curv_contour_length_14/
    metadata/*.npy
    imgs/**.png
```

IMDB is downloaded automatically. AAN may be fetched from the OpenNLPLab
mirror with `--download-aan`; ListOps and Pathfinder remain explicit local
inputs so their provenance stays auditable. Tokenized ListOps, IMDB, and AAN
are written once as memory-mapped `uint16` caches under `data/lra_cache`.
Every cache has a manifest containing its source identity, split, vocabulary
fingerprint, preprocessing schema, and maximum length. A mismatch invalidates
and rebuilds the cache instead of silently reusing stale token IDs.

## Aggregation and reported baselines

```bash
python experiments/summarize_sequence_benchmarks.py \
  --root runs/sequence --output runs/sequence/summary
```

This produces seed-level `runs.csv`, dataset-level `datasets.csv`, and
`mean_ranks.csv`, plus seed completeness, wins/ties/losses, and Nemenyi
critical-difference figures. Runtime, peak memory, throughput, and parameter
counts are retained in the CSV outputs. An optional reported-baseline CSV can be supplied with
`--reported-baselines`; use
`experiments/sequence_benchmarks/reported_baselines.example.csv` as the schema.
Ranks use only the intersection of datasets completed by every included model,
so missing results cannot improve a model's rank.

UEA defaults to a rank-16 factorized learned absolute position table. This is
still a trainable absolute position embedding, but its parameter cost is
`length * 16 + 16 * model_dim` instead of `length * model_dim`. Passing
`--position-rank 0` restores the full learned table.
