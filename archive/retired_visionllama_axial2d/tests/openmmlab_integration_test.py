from __future__ import annotations

from pathlib import Path

import pytest
import torch

mmengine = pytest.importorskip("mmengine")
pytest.importorskip("mmdet")
pytest.importorskip("mmseg")

import integrations.openmmlab  # noqa: E402,F401
from integrations.openmmlab import (  # noqa: E402
    LSSODenseVisionLLaMA,
    LSSOSimpleFPN,
)
from mmdet.registry import MODELS as MMDET_MODELS  # noqa: E402
from mmengine.config import Config  # noqa: E402
from mmseg.registry import MODELS as MMSEG_MODELS  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def compact_backbone(**kwargs) -> LSSODenseVisionLLaMA:
    return LSSODenseVisionLLaMA(
        scale="small",
        mixer="rrlsso",
        rank=16,
        image_size=32,
        patch_size=8,
        window_size=2,
        global_block_indices=(1,),
        model_kwargs=dict(
            dim=64,
            depth=2,
            num_heads=4,
            drop_path_rate=0.0,
        ),
        **kwargs,
    )


def test_models_are_registered_in_both_frameworks() -> None:
    assert MMDET_MODELS.module_dict["LSSODenseVisionLLaMA"] is LSSODenseVisionLLaMA
    assert MMSEG_MODELS.module_dict["LSSODenseVisionLLaMA"] is LSSODenseVisionLLaMA
    assert MMDET_MODELS.module_dict["LSSOSimpleFPN"] is LSSOSimpleFPN
    assert MMSEG_MODELS.module_dict["LSSOSimpleFPN"] is LSSOSimpleFPN


def test_backbone_and_five_level_detection_neck() -> None:
    backbone = compact_backbone()
    neck = LSSOSimpleFPN(in_channels=64, out_channels=32, num_outs=5)
    backbone_features = backbone(torch.randn(1, 3, 32, 48))
    outputs = neck(backbone_features)
    assert [tuple(output.shape) for output in outputs] == [
        (1, 32, 16, 24),
        (1, 32, 8, 12),
        (1, 32, 4, 6),
        (1, 32, 2, 3),
        (1, 32, 1, 2),
    ]


def test_backbone_loads_classification_checkpoint(tmp_path) -> None:
    source = compact_backbone()
    checkpoint = tmp_path / "classifier.pt"
    torch.save({"model_state": source.backbone.state_dict()}, checkpoint)
    target = compact_backbone(checkpoint=str(checkpoint))
    target.init_weights()
    torch.testing.assert_close(
        target.backbone.patch_embed.weight, source.backbone.patch_embed.weight
    )


@pytest.mark.parametrize(
    "relative",
    [
        "integrations/openmmlab/configs/mmdet/mask-rcnn_vllama-b_rrlsso-r32.py",
        "integrations/openmmlab/configs/mmseg/upernet_vllama-b_rrlsso-r32.py",
    ],
)
def test_openmmlab_configs_parse(relative: str) -> None:
    config = Config.fromfile(ROOT / relative)
    assert config.model.backbone.type == "LSSODenseVisionLLaMA"
    assert config.model.neck.type == "LSSOSimpleFPN"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_openmmlab_wrappers_cuda_bfloat16() -> None:
    backbone = compact_backbone().cuda().bfloat16()
    neck = LSSOSimpleFPN(in_channels=64, out_channels=32, num_outs=4).cuda().bfloat16()
    image = torch.randn(
        1, 3, 32, 48, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    outputs = neck(backbone(image))
    sum(output.float().mean() for output in outputs).backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
