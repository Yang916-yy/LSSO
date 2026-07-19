# Official DeiT-III + RRLSSO on ImageNet-1K

The maintained ImageNet experiment changes only the token mixer. Patch
embedding, learned absolute position embedding, class token, MLP, normalization,
LayerScale, stochastic depth, augmentation, loss, optimizer, and schedules
follow Meta's released DeiT-III protocol.

The registered names are resolution-independent:

| Registry name | Width | Depth | Heads | rank-32 parameters |
|---|---:|---:|---:|---:|
| `deit3_small_patch16_rrlsso` | 384 | 12 | 6 | 19,398,520 |
| `deit3_base_patch16_rrlsso` | 768 | 12 | 12 | 75,954,952 |
| `deit3_large_patch16_rrlsso` | 1024 | 24 | 16 | 266,589,928 |

All three use Meta's constant stochastic-depth layout and LayerScale
initialization `1e-4`, not timm's depth-wise drop-path schedule and `1e-6`
default. RRLSSO uses ordinary one-dimensional Rank Rotary.

## Size-specific official recipes

| Size/stage | Resolution | epochs | optimizer | LR | drop path | effective batch |
|---|---:|---:|---|---:|---:|---:|
| Small pretrain | 224 | 800 | Apex FusedLAMB | `4e-3` | 0.05 | 2048 |
| Base pretrain | 192 | 800 | Apex FusedLAMB | `3e-3` | 0.20 | 2048 |
| Large pretrain | 192 | 800 | Apex FusedLAMB | `3e-3` | 0.45 | 2048 |
| Base 224 finetune | 224 | 20 | AdamW | `1e-5` | 0.20 | 512 |
| Large 224 finetune | 224 | 20 | AdamW | `1e-5` | 0.45 | 512 |
| Small/Base/Large 384 finetune | 384 | 20 | AdamW | `1e-5` | 0/0.15/0.40 | 512 |

Pretraining uses FP16 autocast and scaling, EMA 0.99996, 3-Augment,
ColorJitter 0.3, three repeated augmentations, Mixup 0.8, CutMix 1.0,
multi-hot BCE, no random erasing, no label smoothing, weight decay 0.05,
five warmup epochs from `1e-6`, and cosine decay to `1e-5`.

Finetuning uses RandAugment `rand-m9-mstd0.5-inc1`, no repeated augmentation,
Mixup/CutMix, smoothing 0.1, no random erasing, weight decay 0.1, and five
warmup epochs. Validation uses the official crop ratio 1.0.

## RRLSSO-specific safeguards

Two optional regularizers are intentionally zero at initialization:

- `--rrlsso-gain-reg 1e-4` anchors excessive logarithmic output-gain drift;
- `--rrlsso-alpha-reg 1e-4` activates only when `alpha/alpha_max` exceeds 0.8.

They do not weaken the initialized solve and can be disabled with zero weights
for the exact recipe ablation.

## Formal launch

The default Base command chooses physical batch 512 and accumulation 4 to
reproduce effective batch 2048 on one large GPU:

```bash
export HF_TOKEN=...
python experiments/imagenet_wds_train.py \
  --model deit3_base_patch16_rrlsso \
  --stage pretrain \
  --cache-dir /content/imagenet-wds \
  --workers 32 --eval-workers 4 \
  --require-mathdx
```

Formal pretraining fails loudly if NVIDIA Apex FusedLAMB is unavailable. The
portable timm LAMB fallback is available only with `--allow-unfused-lamb` for
local tests and must not be used for reported training.

After Base pretraining:

```bash
python experiments/imagenet_wds_train.py \
  --model deit3_base_patch16_rrlsso \
  --stage finetune224 \
  --init-checkpoint runs/imagenet1k/deit3_base_patch16_rrlsso/pretrain192/best.pt \
  --cache-dir /content/imagenet-wds --require-mathdx --no-resume
```

The position embedding is bicubically resized between patch grids. Each stage
has independent atomic `last.pt` and `best.pt` checkpoints, including model,
EMA, optimizer, scheduler, GradScaler, completed epoch, update count, and RNG
state. Streaming order after a restart remains newly stochastic.
