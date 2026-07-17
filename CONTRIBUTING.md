# Contributing

## Scope a change

Use short-lived branches from `main`.  A reviewable change should belong to one
of these scopes:

- `core`: mathematical or module semantics;
- `backend`: CUDA, MathDx, dispatch, or fallback behavior;
- `model`: reusable backbone or adapter;
- `experiment`: training/evaluation recipe;
- `integration`: external framework support;
- `paper`: manuscript only;
- `infra`: packaging, tests, CI, or repository tooling.

Do not combine kernel changes, a new experiment, and manuscript edits in one
commit.  A paper update may follow the tested experiment in a separate commit.

Suggested commit subjects are short and imperative, for example:

```text
backend: add split-N masked backward dispatch
experiment: benchmark genomic mixer training cost
paper: add GenomicBenchmarks auxiliary results
```

## Compatibility

The supported imports are `lsso`, `lsso.ops`, `lsso.nn`, and the public portion
of `lsso.backends`.  When moving an existing implementation, retain a forwarding
import at its old path until a versioned deprecation is complete.

Checkpoint-compatible refactors must preserve parameter names and tensor
shapes.  If that is impossible, include an explicit state-dict migration and a
test for it.

## Validation

Run the smallest relevant numerical tests while iterating, then before merging:

```bash
python tools/check_repository_hygiene.py
python -m compileall -q lsso experiments benchmarks integrations tools
pytest -q
python -m build
```

GPU-specific changes also require reference comparisons, backward-gradient
checks, mask-leakage tests, and fallback coverage on unsupported shapes.

## Experimental artifacts

Do not commit datasets, checkpoints, complete run directories, generated PDFs,
extension binaries, or caches.  Commit a compact result summary only when it
records the model/configuration, seed, dataset split, metric definition, source
commit, and backend used.
