# Dense Downstream Protocols

The dense experiments share the DeiT III LSSO backbone used by ImageNet. They
use a learned two-dimensional patch position table with no learned CLS
position. Rank-Rotary remains an internal rank-space phase choice, rather than
an image-coordinate embedding.

| Scale | Width | Depth | Heads | LSSO rank | Feature taps |
| --- | ---: | ---: | ---: | ---: | --- |
| Small | 384 | 12 | 6 | 32 | `(3, 5, 7, 11)` |
| Base | 768 | 12 | 12 | 48 | `(3, 5, 7, 11)` |
| Large | 1024 | 24 | 16 | 64 | `(7, 11, 15, 23)` |

The four plain-ViT maps are converted to spatial strides `4, 8, 16, 32` with
the same simple feature pyramid convention used by XCiT: two `2x` transpose
convolutions, one transpose convolution, identity, and one `2x` max pool.
The external FPN or UperNet head then consumes those four maps.

The OpenMMLab wrappers derive an image-validity mask from each sample's
`img_shape`, zero padded pixels before patch embedding, and pass the resulting
token mask to every global LSSO mix. This prevents another image's batch
padding, including padding that partially overlaps an edge patch, from affecting
valid outputs.

## Protocol provenance

ImageNet-1K is the official
[DeiT III recipe](https://github.com/facebookresearch/deit/blob/7e160fe43f0252d17191b71cbb5826254114ea5b/README_revenge.md).
The ImageNet launcher and its exact S/B/L recipes are documented in
`docs/IMAGENET_DEIT3.md`.

COCO is a **downstream standard protocol**, not an official DeiT III detection
recipe. It is the public [XCiT Mask R-CNN + FPN 3x
protocol](https://github.com/facebookresearch/xcit/tree/82f5291f412604970c39a912586e008ec009cdca/detection):
COCO 2017, batch size 2 per GPU, XCiT multi-scale crop augmentation, AdamW
`1e-4` with weight decay `0.05`, 36 epochs, 500-iteration linear warmup, and
milestones at epochs 27 and 33.

ADE20K follows the DeiT III paper's UperNet evaluation and the public
[XCiT UperNet 160k
protocol](https://github.com/facebookresearch/xcit/tree/82f5291f412604970c39a912586e008ec009cdca/semantic_segmentation):
ADE20K 150 classes, batch size 2 per GPU, 512px crop, AdamW `6e-5` with
weight decay `0.01`, 1,500-iteration warmup, and 160,000 iterations. The
small decoder uses 384 working channels, while base and large use 512.

## Environment

Install the vision and OpenMMLab Python packages first:

```bash
pip install -e '.[vision,openmmlab]'
```

Then install a **compiled** `mmcv==2.1.*` build matched to the active PyTorch
and CUDA stack using the [OpenMMLab installation
guide](https://mmcv.readthedocs.io/en/latest/get_started/installation.html).
`mmcv-lite` does not provide the CUDA/C++ operators required by Mask R-CNN.
Install Apex with FusedLAMB as described in `docs/IMAGENET_DEIT3.md` for
canonical DeiT III pretraining.

## Launch

The CUDA extension must have been built for every participating GPU
architecture with `tools/build_cuda.sh`. The launcher loads the strict artifact
before MMEngine constructs the model. New downstream runs require an explicit
ImageNet checkpoint; they never silently train a paper result from scratch.
The backbone verifies the checkpoint's current ImageNet contract and canonical
digest, then checks its tier, LSSO operator, and shared DeiT III geometry before
accepting any pretrained tensor. When a valid ImageNet checkpoint and the
downstream backbone use different learned 2D patch grids, the backbone applies
the same bicubic position-table interpolation used by ImageNet fine-tuning.

```bash
torchrun --standalone --nproc_per_node=8 experiments/train_openmmlab.py \
  experiments/openmmlab/configs/coco_mask_rcnn_lsso_deit3_base_3x.py \
  --data-root /datasets/coco \
  --backbone-checkpoint runs/imagenet/deit3_base_224/checkpoint_best.pt \
  --work-dir runs/coco/lsso_deit3_base_3x --launcher pytorch
```

```bash
torchrun --standalone --nproc_per_node=8 experiments/train_openmmlab.py \
  experiments/openmmlab/configs/ade20k_upernet_lsso_deit3_base_160k.py \
  --data-root /datasets/ADEChallengeData2016 \
  --backbone-checkpoint runs/imagenet/deit3_base_224/checkpoint_best.pt \
  --work-dir runs/ade20k/lsso_deit3_base_160k --launcher pytorch
```

Use `--resume` for a downstream checkpoint, `--resume auto` for the latest
checkpoint in the work directory, and `--test CHECKPOINT` for evaluation.
The six leaf configs are named by task and scale under
`experiments/openmmlab/configs/`.
