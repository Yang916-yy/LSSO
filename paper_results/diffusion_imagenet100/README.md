# ImageNet-100 Latent Diffusion

This directory records a one-seed boundary experiment that replaces the
bidirectional token mixer in a DiT-style latent diffusion backbone.

## Shared Configuration

- Dataset: ImageNet-100
- Cached VAE latent means from 224x224 images
- Latent tokens: 784
- Backbone: dim 384, depth 8, heads 8
- Training: 50 epochs, batch size 192, bf16 AMP
- AdamW: learning rate 1.5e-4, weight decay 0.01
- Diffusion steps: 1000
- Seed: 1
- LSSO scale: `gamma_max=0.3`, `theta_gamma_init=-4.0`
- Nystromformer landmarks: 56

`summary.tsv` reports the minimum validation noise-prediction MSE and the final
validation MSE. Mixer MACs are theoretical mixer-only estimates over all eight
blocks at 784 tokens.

This experiment does not include FID, KID, or a controlled sample grid.
Therefore it supports only an optimization/efficiency comparison, not a claim
that lower validation MSE guarantees better perceptual generation quality.

The four final checkpoints are distributed in the GitHub Release asset
`diffusion_imagenet100_checkpoints.tar`.
