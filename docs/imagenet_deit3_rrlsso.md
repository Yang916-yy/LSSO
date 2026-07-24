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

All three use Meta's constant stochastic-depth layout, LayerScale initialization
`1e-4`, and truncated-normal class-token initialization with standard deviation
`0.02`, rather than current timm defaults. RRLSSO uses ordinary one-dimensional
Rank Rotary.

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

### Virtual-device augmentation groups

Large physical batches do not change the official per-GPU augmentation
semantics. Each physical batch is divided into virtual groups before batch-mode
Mixup/CutMix: 256 samples for Small/Base pretraining, 64 for Large pretraining
and 224px refinement, and 64/32/16 for Small/Base/Large at 384px. Each group
draws its own Mixup/CutMix parameters, while all groups are concatenated for one
large forward pass.

Repeated augmentation is arranged by group rather than by sample. Three views
of one image are emitted into three different virtual groups, so no group
contains duplicate views of the same source image. This reproduces the relevant
semantics of Meta's distributed RASampler without sacrificing single-GPU
throughput.

## RRLSSO solve-scale parameterization

Trace normalization removes the relation basis' arbitrary global energy scale.
Each layer and head then learns an unrestricted positive resolvent strength:

```text
g     = exp(theta_gain)
alpha = exp(theta_alpha)
beta  = exp(-theta_alpha)
```

Both log coordinates start at zero. With epsilon ignored,
`trace(U.T @ U)=rank`, so the average compact spectral scale is one and
`alpha=beta=1` places the initial response at its half-response point.

No RRLSSO-specific loss term or alpha ceiling is applied. The conditional map
in `C` remains an SPD resolvent for every positive alpha. The fused
implementation evaluates the reciprocal Woodbury system
`beta I + U.T @ U`, avoiding both a large `alpha * Gram` matrix and a repeated
alpha multiplication in the readout. The solve scalars remain outside ordinary
weight decay. `metrics.csv` records log-gain spread and the absolute
alpha/beta distribution without altering training.

## Formal launch

The default Base command chooses physical batch 512 and accumulation 4 to
reproduce effective batch 2048 on one large GPU. Every physical batch contains
two independent 256-sample virtual augmentation groups, yielding the official
eight group-level augmentation draws per optimizer update:

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

The position embedding is bicubically resized between patch grids. A refinement
stage initializes a fresh EMA from the resized main model; only an actual resume
restores the stage's saved EMA. Each stage has independent atomic `last.pt` and
`best.pt` checkpoints, including model, EMA, optimizer, scheduler, GradScaler,
completed epoch, update count, and RNG state. Streaming
order after a restart remains newly stochastic.

### Checkpoint state machine

Initialization and resume are separate operations, not two aliases for loading a
checkpoint:

| Intended operation | Required arguments |
|---|---|
| New pretraining | no stage checkpoint exists |
| Resume pretraining | `--resume`, with this stage's `last.pt` |
| New refinement | `--no-resume --init-checkpoint PREVIOUS/best.pt` |
| Resume refinement | `--resume`, with this stage's `last.pt`; no initializer |

The official path deliberately initializes refinement from the raw model rather
than EMA. A fresh refinement EMA is then created from exactly those raw weights,
so its starting state and trained model agree. Checkpoints carry a schema and
stage/model/resolution/rank/alpha metadata; resume fails loudly on incompatible
or legacy state rather than silently changing parameter semantics. Likewise,
`--init-checkpoint` together with `--resume` is rejected. `--overwrite-output`
is required to replace an existing stage checkpoint and is intended only for
deliberate smoke reruns or discarded experiments.
