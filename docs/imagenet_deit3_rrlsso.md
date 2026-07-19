# DeiT-III + RRLSSO on ImageNet-1K

The maintained ImageNet path replaces only the attention mixer in the official
`timm` DeiT-III backbone. Patch embedding, learned absolute position embedding,
class token, LayerScale, MLP, normalization, and classifier head are retained.
RRLSSO uses ordinary one-dimensional Rank Rotary; the retired axial 2D variant
is not part of these models.

## Registered models

| Registry name | Width | Depth | Heads | Parameters |
|---|---:|---:|---:|---:|
| `deit3_small_patch16_192_rrlsso` | 384 | 12 | 6 | 19,378,552 |
| `deit3_base_patch16_192_rrlsso` | 768 | 12 | 12 | 75,915,016 |
| `deit3_large_patch16_192_rrlsso` | 1024 | 24 | 16 | 266,536,680 |

The parameter counts use rank 32 and a 1,000-class head at 192px. All builders
accept a different `img_size` and `rank` through `timm.create_model`.

## Two-stage protocol

Stage 1 trains at 192px for 800 epochs. It retains the established project
recipe: RandAugment, Mixup 0.8, CutMix 1.0, Random Erasing 0.25, AdamW,
cosine decay, BF16, gradient clipping, and seed 0. The peak learning rate is
linearly scaled from `4e-3 @ effective batch 4096` unless explicitly supplied.

Stage 2 refines the trained model at 224px for 20 epochs. It creates a new
AdamW optimizer at learning rate `1e-5`, uses five warmup epochs and `1e-6`
minimum learning rate, and bicubically resizes the learned patch position
embedding from the 12x12 grid to 14x14. Its checkpoints live in a separate
directory and never overwrite the 192px run.

```bash
export HF_TOKEN=...  # gated timm/imagenet-1k-wds access

python experiments/imagenet_wds_train.py \
  --model deit3_base_patch16_192_rrlsso \
  --stage pretrain \
  --cache-dir /local_nvme/imagenet-wds \
  --batch-size 768 --eval-batch-size 768 \
  --workers 32 --eval-workers 4 \
  --require-mathdx

python experiments/imagenet_wds_train.py \
  --model deit3_base_patch16_192_rrlsso \
  --stage finetune \
  --init-checkpoint runs/imagenet1k/deit3_base_patch16_192_rrlsso/pretrain192/best.pt \
  --cache-dir /local_nvme/imagenet-wds \
  --batch-size 512 --eval-batch-size 512 \
  --workers 32 --eval-workers 4 \
  --require-mathdx --no-resume
```

Before a formal run, use one shard and two steps. This exercises model
registration, the fused backend, decoding, forward/backward, validation, and
checkpoint creation.

```bash
python experiments/imagenet_wds_train.py \
  --model deit3_small_patch16_192_rrlsso \
  --epochs 1 --steps-per-epoch 2 --max-val-steps 2 \
  --batch-size 8 --eval-batch-size 8 \
  --workers 0 --eval-workers 0 --shard-limit 1 \
  --output /tmp/deit3_rrlsso_smoke --no-resume
```

`last.pt` stores model and optimizer state, completed epoch, global optimizer
update, best accuracy, arguments, and Python/NumPy/PyTorch RNG state. Automatic
resume is epoch-exact for model, optimizer, and learning-rate progress. A
streaming DataLoader cannot reproduce the exact prior intra-shard cursor after
a process restart, so sample order after a restart remains stochastic.
