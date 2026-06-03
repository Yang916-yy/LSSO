# LSSO Experiment Results

This directory contains tables, run manifests, source scripts, logs, and checkpoints for completed LSSO experiments. MAC columns are mixer-only MACs unless otherwise noted.

## Experiment Groups

- `retrieval_main/`: FIQA, NFCorpus, SciFact retrieval main table with MHA, official Nystromformer, LSSO-r16, and LSSO-r32.
- `retrieval_ablation/`: FIQA and SciFact LSSO ablations: no-global, fixed scale, no U RMS norm, r8, and r4.
- `rank_pruning/`: inference-time rank pruning for trained LSSO-r32 retrieval checkpoints.
- `cifar100_cv_main/`: CIFAR-100 CV main table, patch=2, CLS pooling, medium augmentation, 3 seeds.

## Notes

- Retrieval models use `max_doc_len=512`, mean pooling, dim=256, depth=8, heads=8.
- CIFAR-100 uses image size 32, patch size 2, dim=96, depth=3, heads=6, RandAugment(2,9), Mixup=0.2, CutMix=0.5.
- LSSO runs use `gamma_max=0.3` and `theta_gamma_init=-4.0`.
- Checkpoints are included under each experiment group when available.
