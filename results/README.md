# Published Results

This directory contains the compact, machine-readable evidence behind the
results reported in the README and paper. It publishes final per-seed test
metrics and enough provenance to identify the code, data, model shell, and
runtime used by each formal panel.

| Result | Files | Scope |
| --- | --- | --- |
| GenomicBenchmarks | [`genomic/per_seed.csv`](genomic/per_seed.csv), [`genomic/provenance.csv`](genomic/provenance.csv), [`genomic/metadata.json`](genomic/metadata.json) | Matched MHA/LSSO shell, eight tasks, three seeds |
| Long Range Arena | [`lra/per_seed.csv`](lra/per_seed.csv), [`lra/provenance.csv`](lra/provenance.csv), [`lra/metadata.json`](lra/metadata.json) | LSSO ListOps, Text, Retrieval, and Pathfinder-32, three seeds |
| Contraction certificates | [`certificates/summary.csv`](certificates/summary.csv), [`certificates/metadata.json`](certificates/metadata.json) | Layerwise 1K/2K/4K/8K operator stress test |
| CUDA wall clock | [`cuda/wall_clock.csv`](cuda/wall_clock.csv), [`cuda/metadata.json`](cuda/metadata.json) | Complete mixer forward and forward-backward timing |

Accuracies in the CSV files are percentages. `selected_epoch` is the epoch of
the validation-selected checkpoint evaluated once on the held-out test split.
Each `config_digest` is the SHA-256 digest stored by the formal runner for the
fully resolved run configuration. Dataset content hashes and recorded protocol
metadata are included in the corresponding provenance table; split
fingerprints are included where the formal artifact records them.

The repository intentionally omits checkpoints, caches, and verbose training
logs. Those files are large and are not needed to audit the reported numbers.
The LRA CSV contains only results trained by this repository; published
comparison values in the paper remain attributed to their original sources.
