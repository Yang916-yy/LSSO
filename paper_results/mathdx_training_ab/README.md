# MathDx real-training A/B

Environment: RTX 5070 Ti (SM120), PyTorch 2.12.0+cu130, CUDA 13.0,
MathDx 26.06, torchvision ViT-B/4, CIFAR-100, BF16 autocast, batch 128.

Each successful run used the real training loop in
`experiments/cv_vit_rrlsso_cifar100.py`: forward, cross-entropy, backward,
gradient clipping, AdamW, four-worker augmented CIFAR input, and one validation
batch per epoch. Runs used three epochs with 120 train steps each. Epoch 1 was
treated as warmup; the table reports the mean of epochs 2 and 3. The effective
per-step and image rates divide the complete measured epoch time (including one
validation batch) by 120 steps / 15,360 training images, so they are
conservative training-throughput figures.

## Result

- RRLSSO, G=12: 13.6656 -> 12.3433 seconds, 1.107x throughput, 9.68% less time.
- Grouped-RRLSSO, G=4: 12.8183 -> 11.4041 seconds, 1.124x throughput, 11.03% less time.
- Grouped-RRLSSO, G=1: the MathDx run completed at 10.9753 seconds. The backend-off
  run consistently crashed inside PyTorch/MAGMA `apply_lu_factor_batched_magma`
  with an illegal memory access for rank 32, RHS 768. Reducing batch size to 64
  reproduced the same failure, so no valid speedup ratio is reported.

Peak allocated memory was effectively unchanged for G=12/G=4. Training losses
tracked closely between enabled and disabled runs; these short timing trials
are not accuracy comparisons.

Raw successful metrics are under `runs/mathdx_train_ab/` in the corresponding
`*_on` and `*_off` directories. Exact aggregate values are in `summary.tsv`.
