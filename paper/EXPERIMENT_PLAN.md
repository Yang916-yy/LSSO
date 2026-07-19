# Maintained formal experiment plan

## Active architectural scope

- RRLSSO uses ordinary Rank Rotary.
- The solve scalars are learnable per-head gain and strength, \((g_h,\alpha_h)\).
- Relation bases use trace normalization and length-normalized statistics.
- VisionLLaMA and axial 2-D rotary are retired and must not appear as active
  baselines, implementations, or recommended recipes.

## Main computer-vision experiments

Use maintained recent backbones with minimal architectural modification:

1. ImageNet-1K classification: DeiT-III Base with only the attention mixer
   replaced by RRLSSO; compare against the unchanged official DeiT-III result
   and a locally reproduced MHA control when protocol differences require it.
2. COCO detection: select a maintained MMDetection backbone and replace only
   its global token-mixing blocks. Do not reuse the retired Pyramid
   VisionLLaMA integration.
3. ADE20K segmentation: use the matching maintained MMSegmentation backbone
   and the same mixer replacement policy as detection.

All local comparisons must hold augmentation, optimizer, schedule, seed,
parameter budget, and positional inputs fixed.

## Auxiliary experiments

- GenomicBenchmarks: eight tasks, Motif-7 stem, ordinary Rank Rotary, three
  seeds for the frozen formal recipe.
- LRA: ListOps, Text, Retrieval, and Image/Pathfinder where protocol validity is
  established. Spatial tasks use the ordinary flattened Rank Rotary path.
- UEA-30 remains excluded from the formal paper.

## Required ablations

- rank: 16, 32, 48, 64;
- learnable versus fixed per-head gain;
- trace normalization versus the historical token-RMS path;
- ordinary Rank Rotary versus no rotary transform;
- LSSO versus RRLSSO at matched rank and training recipe.

The retired axial 2-D variant is not a paper ablation. Its historical evidence
is preserved only under `archive/retired_visionllama_axial2d/`.
