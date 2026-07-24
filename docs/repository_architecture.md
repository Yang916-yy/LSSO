# Repository architecture

LSSO is a research framework, not a collection of independent training
scripts.  The repository is a monorepo so operator changes, integrations,
reproducible experiments, and the paper can evolve together.  The installable
wheel remains deliberately smaller than the repository.

## Dependency direction

Dependencies must point downward through these layers:

```text
experiments / integrations / benchmarks
                 |
              lsso.nn
                 |
              lsso.ops
                 |
           lsso.backends
                 |
       PyTorch reference / CUDA extension
```

The paper may consume checked experimental summaries, but runtime code must
never import from `experiments`, `benchmarks`, `integrations`, or `paper`.

## Directory contracts

- `lsso/`: the only package included in the wheel.
  - `lsso.api`: stable public surface used by `import lsso`.
  - `lsso.ops`: supported functional operators and solve-state operations.
  - `lsso.nn`: reusable `torch.nn.Module` components.
  - `lsso.backends`: optional backend discovery and loading facades.
- `csrc/`: compiled backend implementation.  Kernels implement backend
  contracts; they do not define model semantics.
- `experiments/`: reproducible task training and evaluation programs.
- `benchmarks/`: performance and complexity measurement, never training
  results used as implicit configuration.
- `integrations/`: adapters for external frameworks such as timm, MMDetection,
  and MMSegmentation.
- `tests/`: numerical, dispatch, integration, and regression coverage.
- `tools/`: repository operations, data conversion, and backend build tools.
- `paper/`: manuscript sources and small, reviewable tables only.

## Public and compatibility APIs

New user code should import from `lsso`, `lsso.ops`, `lsso.nn`, or the narrow
`lsso.backends` facade.  Existing imports from `lsso.modules` and
`lsso.mathdx_backend` remain supported during the staged refactor.

An implementation symbol becomes public only after it has:

1. a documented shape and dtype contract;
2. CPU/reference behavior;
3. numerical tests;
4. stable error and fallback behavior.

CUDA fast paths may reject unsupported inputs, but must fall back without
changing public semantics.

## Artifact policy

Git stores source, configuration, tests, and compact summaries.  It does not
store datasets, environments, build trees, checkpoints, complete caches, or
raw experiment directories.  A result cited by the paper should have a small
machine-readable summary plus enough configuration and provenance to recreate
it; the checkpoint itself belongs in external artifact storage.

## Frozen operator interface

Supported constructors use Trace normalization with learnable log-gain and
learnable unbounded log-alpha. Alpha starts internally at one and is restored
from `theta_alpha` in checkpoints; initialization and retired
parameterization ablations are not public constructor options.

Future internal moves should remain incremental:

1. move solve-state and reference functions behind `lsso.ops`;
2. move backend loading and dispatch behind `lsso.backends`;
3. split CUDA statistics, solve, readout, backward, and bindings translation
   units without changing Python calls;
4. remove a compatibility module only in a versioned release after downstream
   imports have migrated.
