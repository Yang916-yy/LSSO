# Bidirectional CV program

The experimental question is whether RRLSSO can replace the token mixer of a
standard ViT encoder while preserving the surrounding architecture and
training recipe.

## Established working baseline

- torchvision ViT-B/4 on CIFAR-100;
- MHA and RRLSSO differ only in the mixer replacement;
- effective-length normalization and the active global-correction default;
- MathDx enabled when the installed GPU and rank are supported.

The completed single-seed 80-epoch CIFAR-100 run is a development signal, not
a paper aggregate. The next measurement should be a multi-seed comparison
with the same augmentation and optimizer schedule.

## Next experiments

1. multi-seed CIFAR-100: MHA, RRLSSO, and selected grouped RRLSSO;
2. larger image classification with matched ViT capacity;
3. retrieval and semantic segmentation using the same bidirectional mixer;
4. report both model-level wall time and kernel-level MathDx speedups.

`experiments/cv_vit_rrlsso_cifar100.py` is the current reproducible training
entry point. Outputs belong under `runs/`, never in the source tree.
