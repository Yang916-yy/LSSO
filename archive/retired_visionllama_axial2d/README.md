# Retired VisionLLaMA and axial 2-D rotary code

This directory is a historical snapshot, not a supported part of LSSO.

The code was retired because the inherited axial 2-D rotary construction uses
language-scale frequencies on small visual grids. On rank 32, an 8x8 grid left
8 of 16 rotary pairs nearly inactive over the complete grid; a 14x14 grid left
6 of 16 nearly inactive. Axis-zero rows and columns additionally create exact
identity subspaces. Controlled CIFAR-100 runs found ordinary Rank Rotary at
least as strong as the axial 2-D variant.

Consequences:

- nothing under this directory is imported by the active package;
- it is excluded from the maintained test and experiment surfaces;
- its VisionLLaMA, timm, OpenMMLab, ImageNet, and axial-rotary paths must not be
  presented as current baselines;
- files remain only to preserve provenance for old checkpoints and experiments.

Do not build new work on this directory. Git history is the authoritative
record for any further archaeology.
