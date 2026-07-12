from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn.functional as F

from .vision_llama import (
    VISION_LLAMA_CONFIGS,
    VisionLLaMA,
    convert_official_vision_llama_state_dict,
)
from .windowed_dense_mixer import run_windowed_block


def default_global_block_indices(depth: int) -> tuple[int, ...]:
    """Four approximately even cross-window propagation blocks."""
    if depth < 4:
        return (depth - 1,)
    return tuple(min(depth - 1, round((index + 1) * depth / 4) - 1) for index in range(4))


class DenseVisionLLaMA(VisionLLaMA):
    """Plain VisionLLaMA adapted to high-resolution dense prediction.

    Parameters retain the classification model's key layout. The CLS token and
    classification head remain loadable but are not used by dense forwarding.
    """

    def __init__(
        self,
        *,
        window_size: int | None = 16,
        global_block_indices: Iterable[int] | None = None,
        out_indices: Iterable[int] | None = None,
        **kwargs,
    ) -> None:
        kwargs.setdefault("num_classes", 0)
        super().__init__(**kwargs)
        self.window_size = window_size
        depth = len(self.blocks)
        if global_block_indices is None:
            global_block_indices = default_global_block_indices(depth)
        self.global_block_indices = tuple(sorted(set(global_block_indices)))
        invalid = [index for index in self.global_block_indices if not 0 <= index < depth]
        if invalid:
            raise ValueError(f"global block indices outside [0, {depth}): {invalid}")
        if out_indices is None:
            out_indices = (depth - 1,)
        self.out_indices = tuple(sorted(set(out_indices)))
        invalid = [index for index in self.out_indices if not 0 <= index < depth]
        if invalid:
            raise ValueError(f"output block indices outside [0, {depth}): {invalid}")

    def _dense_position_embedding(
        self, spatial_shape: tuple[int, int]
    ) -> torch.Tensor | None:
        if self.pos_embed is None:
            return None
        source_h = self.image_size[0] // self.patch_size
        source_w = self.image_size[1] // self.patch_size
        patch = self.pos_embed[:, 1:]
        if spatial_shape == (source_h, source_w):
            return patch
        patch = patch.reshape(1, source_h, source_w, self.dim).permute(0, 3, 1, 2)
        patch = F.interpolate(
            patch.float(), size=spatial_shape, mode="bicubic", align_corners=False
        )
        return patch.to(self.pos_embed.dtype).flatten(2).transpose(1, 2)

    def forward_intermediates(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if image.shape[-2] % self.patch_size or image.shape[-1] % self.patch_size:
            raise ValueError(
                f"input spatial shape {tuple(image.shape[-2:])} must be divisible "
                f"by patch size {self.patch_size}"
            )
        x = self.patch_embed(image)
        spatial_shape = (x.shape[-2], x.shape[-1])
        x = x.flatten(2).transpose(1, 2)
        position = self._dense_position_embedding(spatial_shape)
        if position is not None:
            x = x + position.to(device=x.device, dtype=x.dtype)
        outputs: list[torch.Tensor] = []
        for index, block in enumerate(self.blocks):
            if self.window_size is None or index in self.global_block_indices:
                x = block(x, spatial_shape=spatial_shape, num_prefix_tokens=0)
            else:
                x = run_windowed_block(
                    block,
                    x,
                    spatial_shape=spatial_shape,
                    window_size=self.window_size,
                )
            if index in self.out_indices:
                feature = self.norm(x) if index == len(self.blocks) - 1 else x
                outputs.append(
                    feature.reshape(image.shape[0], *spatial_shape, self.dim)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
        return tuple(outputs)

    def forward_features(self, image: torch.Tensor) -> torch.Tensor:
        return self.forward_intermediates(image)[-1]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.forward_features(image)


def create_dense_vision_llama(
    scale: str = "base", *, mixer: str = "rrlsso", rank: int = 32, **kwargs
) -> DenseVisionLLaMA:
    try:
        config = VISION_LLAMA_CONFIGS[scale]
    except KeyError as exc:
        raise ValueError(f"unknown scale {scale!r}; choose {tuple(VISION_LLAMA_CONFIGS)}") from exc
    model_config = dict(
        dim=config.dim,
        depth=config.depth,
        num_heads=config.num_heads,
        drop_path_rate=config.drop_path_rate,
    )
    model_config.update(kwargs)
    return DenseVisionLLaMA(mixer=mixer, rank=rank, **model_config)


def load_dense_vision_llama_checkpoint(
    model: DenseVisionLLaMA,
    checkpoint: str | Path | dict,
) -> tuple[list[str], list[str]]:
    """Load a project or upstream classification checkpoint into dense mode."""
    if not isinstance(checkpoint, dict):
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state", checkpoint.get("model", checkpoint))
    converted = convert_official_vision_llama_state_dict(model, state)
    converted = {
        key: value for key, value in converted.items()
        if not key.startswith("head.")
    }
    incompatible = model.load_state_dict(converted, strict=False)
    return incompatible.missing_keys, incompatible.unexpected_keys
