# ImageNet DeiT III

`experiments/train_imagenet.py` launches the shared `experiments/imagenet.py`
workflow, which implements the ImageNet-1K recipes from the
[official DeiT III repository at commit 7e160fe43f0252d17191b71cbb5826254114ea5b](https://github.com/facebookresearch/deit/blob/7e160fe43f0252d17191b71cbb5826254114ea5b/README_revenge.md).
The ImageNet root must be an `ImageFolder` tree with `train/` and `val/`.
For a single-node notebook workflow that validates the environment, installs
the pinned native runtime, launches a run, and exposes a refreshable monitor,
see [`notebooks/imagenet_launcher.ipynb`](../notebooks/imagenet_launcher.ipynb).

The pretraining configurations preserve the public commands: 800 epochs,
cosine decay with five warmup epochs, three-view repeated augmentation,
3-Augment, Mixup `0.8`, CutMix `1.0`, BCE targets, EMA `0.99996`, and Apex
FusedLAMB. The S model uses 224px input; B and L use 192px before the official
20-epoch 224px AdamW fine-tuning phase. That phase retains batch-mode Mixup
`0.8` and CutMix `1.0`, while switching to RandAugment, disabling repeated
augmentation and BCE targets, and using label smoothing `0.1`. Per-GPU batches
and learning rates are not linearly rescaled, matching `--unscale-lr` in the
source commands.

The runner keeps the formal optimizer update independent of the available GPU
count. `batch_size` is a physical per-GPU batch; it must be a multiple of the
configured virtual Mixup group. The launcher derives accumulation so that
`world_size * batch_size * grad_accum` equals the fixed global effective batch.
Each physical batch is split into independent batch-mode Mixup/CutMix groups,
and repeated augmentation replays whole source groups, so no group contains two
views of the same image. Epochs are truncated to whole effective-batch updates.

Data loading uses independent per-rank `train_workers` and `val_workers`
settings. The defaults are 10 train workers and 4 validation workers; override
them with `--train-workers` and `--val-workers`. Workers are recreated at each
epoch boundary so their transform RNG can be replayed after an exact resume;
budget up to `world_size * (train_workers + val_workers)` CPU processes while
either loader is active.

| Phase | Effective global batch | Virtual Mixup group | Updates per ImageNet-1K epoch |
| --- | ---: | ---: | ---: |
| S/B pretraining | 2048 | 256 | 625 |
| L pretraining | 2048 | 64 | 625 |
| B/L 224px fine-tuning | 512 | 64 | 2502 |

`--grad-accum` is optional and only accepted when it exactly matches this
contract. It is useful for asserting a launch plan, not for changing the
effective batch. A larger single-GPU physical batch remains `deit3-derived`
when the resolved plan is exact. For example, one H100 can use 512 images for
S/B pretraining or 128 for L pretraining:

```bash
torchrun --standalone --nproc_per_node=1 experiments/train_imagenet.py \
  --tier small --phase pretrain --data-root /datasets/imagenet \
  --output runs/imagenet/deit3_small_h100 --batch-size 512

torchrun --standalone --nproc_per_node=1 experiments/train_imagenet.py \
  --tier large --phase pretrain --data-root /datasets/imagenet \
  --output runs/imagenet/deit3_large_h100 --batch-size 128
```

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
`checkpoint_last.pt`, and `checkpoint_best.pt`. Checkpoints are written through
a same-directory temporary file, `fsync`, and atomic replace, so a preemption
cannot replace a complete checkpoint with a partial one. Resume only with the
same tier, phase, model contract, recipe, resolved batching plan, device type,
and world size; the runner rejects mismatches, including a changed accumulation
factor. At each saved epoch boundary it records every rank's Python, NumPy,
Torch CPU/CUDA, and train/validation worker-generator states. Workers are
recreated at epoch boundaries, so this replays the next epoch's augmentation
and data-loader random streams in the same environment. It does not promise
bitwise identity across changed GPU, CUDA, PyTorch, or kernel environments.
Each current checkpoint carries a canonical SHA-256 digest of its contract.
This schedule change uses ImageNet checkpoint format 4, so older ImageNet
runner checkpoints are intentionally not resumable. Fine-tuning and downstream
loading continue to validate only the compatible model and operator contract
before loading model tensors. `metadata.json` also records the source commit
and whether the worktree was dirty before the output directory was created; a
dirty marker does not identify the uncommitted patch.

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
