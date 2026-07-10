# LSSO Experiment Results

This directory contains tables, run manifests, source scripts, and logs for completed LSSO experiments. MAC columns are mixer-only MACs unless otherwise noted.

Checkpoints are distributed as GitHub Release assets instead of being tracked in
the git repository, so a default clone stays focused on the core LSSO package.
See `release_assets.tsv` for asset names, sizes, SHA256 checksums, and contents.

## Experiment Groups

- `gamma_strength_sweep/`: active ViT-B/4 strength-selection evidence after
  strict length normalization; G=4/G=12 and two seeds recommend initial
  `gamma/mu=0.85-1.15`.
- `cv_vit_b4_formal/`: active 80-epoch CIFAR-100 ViT-B/4 MHA vs RRLSSO
  mixer-replacement run, including convergence and receptive-field figures.
- `cifar100_cv_main/`: CIFAR-100 bidirectional encoder baseline table,
  patch=2, CLS pooling, medium augmentation, 3 seeds.
- `imagenet100_cv_main/`: ImageNet-100 bidirectional encoder baseline table,
  image size 224, patch=8, 1 seed.
- `sequence_scaling/`: single-layer latency, forward+backward time, allocated
  peak memory, and mixer MAC scaling from 128 to 2048 tokens.
- `operator_diagnostics/`: trained layer-wise `gamma/mu`, global correction
  ratio, and effective-rank visualizations.
- `retrieval_main/`, `retrieval_ablation/`, `msmarco_beir_transfer/`, and
  `rank_pruning/`: historical bidirectional retrieval evidence.
- `diffusion_imagenet100/`: historical one-seed diffusion boundary experiment.

## Archived causal records

`causal_lm_cache/` and `causal_recall/` are preserved as historical logs only.
They are outside the supported package API, the active CV program, and current
empirical claims.

## Notes

- Retrieval models use `max_doc_len=512`, mean pooling, dim=256, depth=8, heads=8.
- CIFAR-100 uses image size 32, patch size 2, dim=96, depth=3, heads=6, RandAugment(2,9), Mixup=0.2, CutMix=0.5.
- The active `gamma_strength_sweep/` uses a separate ViT-B/4 protocol:
  image size 32, patch size 4, dim=768, depth=12, heads=12, rank=16,
  BF16, strict `length_reference=1`, and 640-update selection runs.
- ImageNet-100 uses image size 224, patch size 8, dim=256, depth=8, heads=8, DeiT-style augmentation, and bf16 AMP.
- The ImageNet-100 LSSO-r32 CV entry was corrected on June 6, 2026: the earlier run accidentally disabled Mixup and CutMix while the other models used Mixup=0.8 and CutMix=1.0.
- The diffusion experiment reports noise-prediction validation loss and mixer cost only; FID/KID and controlled samples remain future work.
- Historical paper-result runs use `gamma_max=0.3` and
  `theta_gamma_init=-4.0`. After effective-length mean normalization, new
  bidirectional runs use `gamma_max=1.2` and `theta_gamma_init=0.5`; see the
  dedicated strength sweep before comparing the two regimes.
- Checkpoints are packaged together in release `paper-results-v1`.
