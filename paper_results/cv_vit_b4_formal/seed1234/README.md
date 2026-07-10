# CIFAR-100 formal ViT-B/4 mixer replacement, seed 1234

## Protocol

- Standard torchvision ViT-B/4: image size 32, patch size 4, 12 encoder
  blocks, hidden size 768, 12 heads, MLP size 3072.
- Replace only `EncoderBlock.self_attention`; LayerNorm, residuals, FFN,
  classifier, optimizer, batch size, augmentation, and schedule are unchanged.
- 80 full CIFAR-100 epochs, batch size 128, AdamW, BF16, cosine schedule with
  two warmup epochs, `RandomCrop+Flip+RandAugment(2,9)+RandomErasing`.
- RRLSSO uses rank 32, strict effective-length normalization
  (`length_reference=1`), and the retuned initial strength
  `gamma_max=1.2`, `theta_gamma_init=0.5`.

## Result

| Mixer | Parameters | Best epoch | Best validation accuracy | Best validation loss | Steady epoch time | Peak memory |
|---|---:|---:|---:|---:|---:|---:|
| MHA | 85.22M | 78 | 38.88% | 2.4576 | 38.28 s | 3.90 GB |
| RRLSSO-r32 | 74.59M | 80 | **64.56%** | **1.5430** | 38.65 s | 3.83 GB |

RRLSSO improves best validation accuracy by **25.68 percentage points**
(`+66.0%` relative), reduces best validation loss by **37.2%**, uses 12.5%
fewer parameters, and has a 0.84% steady epoch-time difference. It first
exceeds the final MHA accuracy at epoch 8 (41.85% vs MHA's final 38.88%).

This is a single-seed formal convergence run. Treat it as strong evidence for
the next experiment matrix, not a multi-seed paper aggregate.

## Artifacts

- `summary.tsv`: machine-readable final metrics.
- `learning_curves.png`: full 80-epoch validation curves.
- `true_logit_layer0/input_erf_true_logit_paper_grid.png`: class-logit input
  effective receptive field before and after training.
- `center_token_layer0/mixer_layer0_abs.png` and
  `center_token_layer11/mixer_layer11_abs.png`: absolute first/last-layer
  token-mixing operators before and after training.

For the class-logit ERF, RRLSSO's post-training saliency has a higher normalized
entropy (0.9735 vs MHA's 0.8680) and lower top-10% mass (24.31% vs 56.22%),
indicating a substantially broader input receptive field. Operator diagnostics
show RRLSSO preserves a strong local diagonal while adding distributed global
correction: layer-0 diagonal mass is 23.27% after training, compared with
MHA's 2.44%; its CLS-row top-10% mass is 21.21%, compared with MHA's 25.90%.
