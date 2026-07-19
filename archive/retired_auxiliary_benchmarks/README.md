# Retired auxiliary benchmark bundle

This directory preserves experiment entry points that are no longer part of
the paper's supported benchmark matrix.  It contains the retired BEIR/MS MARCO,
FLIP-AAV, UEA-30, legacy length-normalization, and Hugging Face ImageNet
streaming paths.

The files are historical snapshots, are excluded from the installed package
and the main test suite, and are not maintained as supported launchers.  Their
data dependencies and output formats may no longer match the current public
API.  The supported non-vision experiments are GenomicBenchmarks and the four
LRA tasks documented in `docs/auxiliary_experiments.md`.

The obsolete `tools/run_food101_vit_b16_80ep.sh` wrapper was deleted rather
than archived because it passed removed `mu/gamma` command-line arguments and
could not launch the current trainer.
