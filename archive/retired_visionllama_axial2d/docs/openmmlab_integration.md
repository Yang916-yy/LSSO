# MMDetection and MMSegmentation integration

This integration exposes the same plain dense VisionLLaMA backbone and
SimpleFPN used by the standalone LSSO code through the MMEngine registries.
It does not fork either OpenMMLab framework.

## Environment

Use a dedicated environment whose PyTorch/CUDA combination has an official
MMCV wheel. Detection needs the compiled MMCV operators; `mmcv-lite` is only
sufficient for importing the wrappers and parsing configs.

```bash
python -m pip install -U openmim
mim install "mmcv==2.1.0"
python -m pip install \
  "mmengine==0.10.7" \
  "mmdet==3.3.0" \
  "mmsegmentation==1.2.2"
python -m pip install -e ".[openmmlab]"
```

Run these commands from the repository root so that
`integrations.openmmlab` is importable. Verify the installation with:

```bash
python -c "import mmcv; import mmcv.ops; print(mmcv.__version__)"
```

## COCO instance detection and segmentation

Place COCO under `data/coco/` using the standard `train2017`, `val2017`, and
`annotations` directories. The four Mask R-CNN variants are:

- `mask-rcnn_vllama-b_mha.py`
- `mask-rcnn_vllama-b_lsso-r32.py`
- `mask-rcnn_vllama-b_rrlsso-r32.py`
- `mask-rcnn_vllama-b_rrlsso-r32-global.py`

For example:

```bash
mim train mmdet \
  integrations/openmmlab/configs/mmdet/mask-rcnn_vllama-b_rrlsso-r32.py \
  --work-dir runs/coco/mask-rcnn_vllama-b_rrlsso-r32
```

The default dense setting uses 1024-pixel LSJ crops, 16x16 windows, global
blocks 2/5/8/11, BF16, AdamW, and a 100-epoch-equivalent iteration budget.
The `-global` config is an ablation with every mixer block global.

## ADE20K semantic segmentation

Place ADE20K under `data/ade/ADEChallengeData2016/`. The UPerNet variants are
`upernet_vllama-b_{mha,lsso-r32,rrlsso-r32}.py`. For example:

```bash
mim train mmseg \
  integrations/openmmlab/configs/mmseg/upernet_vllama-b_rrlsso-r32.py \
  --work-dir runs/ade20k/upernet_vllama-b_rrlsso-r32
```

This route uses 512x512 crops, a four-level SimpleFPN, the standard 160k
schedule, BF16, and AdamW.

## Initializing from a classification checkpoint

Set the backbone checkpoint without editing the config:

```bash
--cfg-options model.backbone.checkpoint=/absolute/path/to/best.pt
```

The loader accepts the ImageNet trainer's `model` or `state_dict` containers,
removes common `module.`/`backbone.` prefixes, ignores the classification
head, and interpolates the learned absolute position embedding when the dense
input grid differs from pretraining.

## Smoke tests

```bash
python -m pytest -q tests/openmmlab_integration_test.py
```

These tests cover both registries, dense feature shapes, SimpleFPN outputs,
checkpoint conversion, config parsing, and a CUDA BF16 forward when CUDA is
available. A real COCO/ADE20K one-iteration smoke remains necessary after
installing compiled MMCV, because it exercises framework CUDA operators that
are intentionally outside this repository.
