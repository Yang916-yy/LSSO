# ImageNet DeiT III

`experiments/train_imagenet.py` launches the shared `experiments/imagenet.py`
workflow, which implements the ImageNet-1K recipes from the
[official DeiT III repository at commit 7e160fe43f0252d17191b71cbb5826254114ea5b](https://github.com/facebookresearch/deit/blob/7e160fe43f0252d17191b71cbb5826254114ea5b/README_revenge.md).
The ImageNet root must be an `ImageFolder` tree with `train/` and `val/`.

The pretraining configurations preserve the public commands: 800 epochs,
cosine decay with five warmup epochs, three-view repeated augmentation,
3-Augment, Mixup `0.8`, CutMix `1.0`, BCE targets, EMA `0.99996`, and Apex
FusedLAMB. The S model uses 224px input; B and L use 192px before the official
20-epoch 224px AdamW fine-tuning phase. Per-GPU batches and learning rates are
not linearly rescaled, matching `--unscale-lr` in the source commands.

The model geometry is:

| Tier | Width | Depth | Heads | LSSO rank | Pretraining input |
| --- | ---: | ---: | ---: | ---: | ---: |
| S | 384 | 12 | 6 | 32 | 224 |
| B | 768 | 12 | 12 | 48 | 192 |
| L | 1024 | 24 | 16 | 64 | 192 |

Install `timm`, `torchvision`, and Apex with FusedLAMB in the experiment
environment. The launcher fails rather than silently changing the published
optimizer. `--allow-lamb-fallback` is available only for explicitly marked
non-fused diagnostic runs.

```bash
torchrun --standalone --nproc_per_node=8 experiments/train_imagenet.py \
  --tier small --phase pretrain --data-root /datasets/imagenet \
  --output runs/imagenet/deit3_small
```

```bash
torchrun --standalone --nproc_per_node=8 experiments/train_imagenet.py \
  --tier base --phase pretrain --data-root /datasets/imagenet \
  --output runs/imagenet/deit3_base_192

torchrun --standalone --nproc_per_node=8 experiments/train_imagenet.py \
  --tier base --phase finetune_224 --data-root /datasets/imagenet \
  --init-checkpoint runs/imagenet/deit3_base_192/checkpoint_best.pt \
  --output runs/imagenet/deit3_base_224
```

The large 192px pretraining command uses 32 GPUs in the official recipe. On
each node, set the usual `torchrun` rendezvous arguments explicitly:

```bash
torchrun --nnodes=4 --node_rank="$NODE_RANK" --nproc_per_node=8 \
  --master_addr="$MASTER_ADDR" --master_port=29500 \
  experiments/train_imagenet.py --tier large --phase pretrain \
  --data-root /datasets/imagenet --output runs/imagenet/deit3_large_192
```

Every output directory contains `metadata.json`, append-only `metrics.jsonl`,
`checkpoint_last.pt`, and `checkpoint_best.pt`. Resume only with the exact
same tier, phase, model contract, and recipe; the runner rejects mismatches.
Each current checkpoint carries a canonical SHA-256 digest of that contract.
Fine-tuning and downstream loading reject a missing, altered, or incompatible
ImageNet contract before loading model tensors.

The shared backbone boundary is intentionally narrow:

```python
integrations.timm.create_lsso_deit3(
    image_size, patch_size, num_classes, embed_dim, depth, num_heads, rank,
    mlp_ratio, core_mode, rank_rotary, bias, implementation, drop_path_rate,
    layer_scale_init_value, norm_eps, no_embed_class=True,
)
```

It must build the current LSSO DeiT III model with a learned 2D patch position
table and no CLS position embedding. The runner owns only data, optimizer,
scheduler, checkpoint, and distributed-training concerns.
