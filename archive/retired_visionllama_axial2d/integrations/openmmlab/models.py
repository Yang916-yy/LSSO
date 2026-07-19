from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from examples.models import (
    SimpleFeaturePyramid,
    create_dense_vision_llama,
    load_dense_vision_llama_checkpoint,
)


class LSSODenseVisionLLaMA(BaseModule):
    """MMEngine wrapper for the shared plain dense VisionLLaMA backbone."""

    def __init__(
        self,
        *,
        scale: str = "base",
        mixer: str = "rrlsso",
        rank: int = 32,
        checkpoint: str | None = None,
        image_size: int = 224,
        patch_size: int = 16,
        window_size: int | None = 16,
        global_block_indices: tuple[int, ...] | None = None,
        frozen_stages: int = -1,
        model_kwargs: dict[str, Any] | None = None,
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.checkpoint = checkpoint
        self.frozen_stages = int(frozen_stages)
        kwargs = dict(model_kwargs or {})
        self.backbone = create_dense_vision_llama(
            scale,
            mixer=mixer,
            rank=rank,
            image_size=image_size,
            patch_size=patch_size,
            window_size=window_size,
            global_block_indices=global_block_indices,
            **kwargs,
        )
        self.out_channels = self.backbone.dim
        self.output_stride = patch_size
        self._freeze_stages()

    def init_weights(self) -> None:
        if self.checkpoint:
            missing, unexpected = load_dense_vision_llama_checkpoint(
                self.backbone, Path(self.checkpoint)
            )
            if missing or unexpected:
                raise RuntimeError(
                    "dense checkpoint did not load cleanly: "
                    f"missing={missing}, unexpected={unexpected}"
                )

    def _freeze_stages(self) -> None:
        if self.frozen_stages < 0:
            return
        for parameter in self.backbone.patch_embed.parameters():
            parameter.requires_grad = False
        self.backbone.cls_token.requires_grad = False
        if self.backbone.pos_embed is not None:
            self.backbone.pos_embed.requires_grad = False
        for block in self.backbone.blocks[:self.frozen_stages]:
            block.eval()
            for parameter in block.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_stages()
        return self

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor]:
        return (self.backbone(images),)


class LSSOSimpleFPN(BaseModule):
    """OpenMMLab neck producing P2--P5 and optional pooled P6."""

    def __init__(
        self,
        *,
        in_channels: int = 768,
        out_channels: int = 256,
        num_outs: int = 4,
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        if num_outs not in (4, 5):
            raise ValueError("num_outs must be 4 or 5")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.pyramid = SimpleFeaturePyramid(in_channels, out_channels)

    def forward(
        self, inputs: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]
    ) -> tuple[torch.Tensor, ...]:
        feature = inputs[-1] if isinstance(inputs, (tuple, list)) else inputs
        outputs = self.pyramid(feature)
        if self.num_outs == 5:
            outputs = outputs + (F.max_pool2d(outputs[-1], kernel_size=1, stride=2),)
        return outputs


def _register_openmmlab_models() -> None:
    registries = []
    try:
        from mmdet.registry import MODELS as mmdet_models

        registries.append(mmdet_models)
    except ImportError:
        pass
    try:
        from mmseg.registry import MODELS as mmseg_models

        registries.append(mmseg_models)
    except ImportError:
        pass
    for registry in registries:
        registry.register_module(module=LSSODenseVisionLLaMA, force=True)
        registry.register_module(module=LSSOSimpleFPN, force=True)


_register_openmmlab_models()
