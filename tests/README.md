# Test Map

The test tree is organized by the public boundary it protects. Run the smallest
matching scope during normal work; each file is a focused, Git-tracked contract
rather than a local experiment or benchmark.

| Scope | Command | Covers |
| --- | --- | --- |
| `core` | `python -m pytest tests/core -m "not cuda"` | Operator mathematics, model/configuration behavior, and public API. |
| `cuda` | `python -m pytest tests/cuda` | Native extension ABI, dispatch, and CUDA oracle comparisons. |
| `integrations` | `python -m pytest tests/integrations -m "not cuda"` | timm and OpenMMLab adapters. |
| `experiments` | `python -m pytest tests/experiments -m "not cuda"` | Data protocols and training entrypoints. |
| `repository` | `python -m pytest tests/repository` | Repository shape and ownership contract. |

Use `python -m pytest -m cuda` to select all CUDA-marked checks across scopes.
Run `python -m pytest -q` only for a release, a broad refactor, or a change
that crosses multiple scopes.
