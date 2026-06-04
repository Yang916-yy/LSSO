# LSSO Experiment Results

This directory contains tables, run manifests, source scripts, and logs for completed LSSO experiments. MAC columns are mixer-only MACs unless otherwise noted.

Checkpoints are distributed as GitHub Release assets instead of being tracked in
the git repository, so a default clone stays focused on the core LSSO package.
See `release_assets.tsv` for asset names, sizes, SHA256 checksums, and contents.

## Experiment Groups

- `retrieval_main/`: FIQA, NFCorpus, SciFact retrieval main table with MHA, official Nystromformer, LSSO-r16, and LSSO-r32.
- `retrieval_ablation/`: FIQA and SciFact LSSO ablations: no-global, fixed scale, no U RMS norm, r8, and r4.
- `msmarco_beir_transfer/`: MS MARCO pretraining followed by 3-seed zero-shot BEIR transfer evaluation on FIQA, NFCorpus, SciFact, ArguAna, and TREC-COVID.
- `rank_pruning/`: inference-time rank pruning for trained LSSO-r32 retrieval checkpoints.
- `cifar100_cv_main/`: CIFAR-100 CV main table, patch=2, CLS pooling, medium augmentation, 3 seeds.
- `imagenet100_cv_main/`: ImageNet-100 CV main table, image size 224, patch=8, 1 seed.

## Notes

- Retrieval models use `max_doc_len=512`, mean pooling, dim=256, depth=8, heads=8.
- CIFAR-100 uses image size 32, patch size 2, dim=96, depth=3, heads=6, RandAugment(2,9), Mixup=0.2, CutMix=0.5.
- ImageNet-100 uses image size 224, patch size 8, dim=256, depth=8, heads=8, DeiT-style augmentation, and bf16 AMP.
- LSSO runs use `gamma_max=0.3` and `theta_gamma_init=-4.0`.
- Checkpoints are packaged in releases `paper-results-v0` and `paper-results-v1`.
