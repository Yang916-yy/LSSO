from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_llama import RMSNorm, VisionLLaMABlock


def _partition_windows(x: torch.Tensor, height: int, width: int, size: int):
    B, _N, D = x.shape
    grid = x.view(B, height, width, D)
    padded_h = (height + size - 1) // size * size
    padded_w = (width + size - 1) // size * size
    grid = F.pad(grid, (0, 0, 0, padded_w - width, 0, padded_h - height))
    valid = torch.zeros(B, padded_h, padded_w, device=x.device, dtype=torch.bool)
    valid[:, :height, :width] = True
    windows = (
        grid.view(B, padded_h // size, size, padded_w // size, size, D)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(-1, size * size, D)
    )
    masks = (
        valid.view(B, padded_h // size, size, padded_w // size, size)
        .permute(0, 1, 3, 2, 4)
        .reshape(-1, size * size)
    )
    return windows, masks, (padded_h, padded_w)


def _unpartition_windows(
    windows: torch.Tensor, batch: int, height: int, width: int,
    size: int, padded_shape: tuple[int, int],
) -> torch.Tensor:
    padded_h, padded_w = padded_shape
    D = windows.shape[-1]
    grid = (
        windows.view(batch, padded_h // size, padded_w // size, size, size, D)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch, padded_h, padded_w, D)
    )
    return grid[:, :height, :width].reshape(batch, height * width, D)


class PyramidBlock(VisionLLaMABlock):
    def __init__(self, *args, window_size: int | None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.window_size = window_size

    def forward(self, x: torch.Tensor, *, spatial_shape: tuple[int, int], **_):
        if self.window_size is None:
            return super().forward(x, spatial_shape=spatial_shape, num_prefix_tokens=0)
        B = x.shape[0]
        height, width = spatial_shape
        windows, masks, padded = _partition_windows(
            x, height, width, self.window_size
        )
        windows = super().forward(
            windows,
            spatial_shape=(self.window_size, self.window_size),
            valid_mask=masks,
            num_prefix_tokens=0,
        )
        return _unpartition_windows(
            windows, B, height, width, self.window_size, padded
        )


class PyramidPatchEmbed(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, patch_size: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        shape = (x.shape[-2], x.shape[-1])
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x, shape


@dataclass(frozen=True)
class PyramidConfig:
    dims: tuple[int, int, int, int]
    depths: tuple[int, int, int, int]
    heads: tuple[int, int, int, int]
    drop_path_rate: float


PYRAMID_CONFIGS = {
    "small": PyramidConfig((64, 128, 256, 512), (2, 2, 10, 4), (2, 4, 8, 16), 0.2),
    "base": PyramidConfig((96, 192, 384, 768), (2, 2, 18, 2), (3, 6, 12, 24), 0.3),
}


class PyramidVisionLLaMA(nn.Module):
    """Four-stage VisionLLaMA backbone for classification and dense tasks.

    ``windowed`` applies identical local windows to every mixer. With
    ``alternating-global``, even blocks are windowed and odd blocks are global.
    """

    def __init__(
        self,
        *,
        image_size: int = 224,
        in_channels: int = 3,
        num_classes: int = 1000,
        dims=(64, 128, 256, 512),
        depths=(2, 2, 10, 4),
        heads=(2, 4, 8, 16),
        mixer: str = "rrlsso",
        rank: int = 32,
        window_size: int = 7,
        attention_policy: str = "windowed",
        drop_path_rate: float = 0.2,
        learned_position: bool = True,
        out_indices=(0, 1, 2, 3),
        **mixer_kwargs,
    ) -> None:
        super().__init__()
        if attention_policy not in {"windowed", "alternating-global"}:
            raise ValueError("attention_policy must be windowed or alternating-global")
        self.dims, self.depths = tuple(dims), tuple(depths)
        self.out_indices = tuple(out_indices)
        self.patch_embeds = nn.ModuleList()
        input_dim = in_channels
        for index, dim in enumerate(dims):
            self.patch_embeds.append(PyramidPatchEmbed(input_dim, dim, 4 if index == 0 else 2))
            input_dim = dim
        rates = iter(torch.linspace(0, drop_path_rate, sum(depths)).tolist())
        self.stages = nn.ModuleList()
        for dim, depth, num_heads in zip(dims, depths, heads):
            stage = nn.ModuleList()
            for block_index in range(depth):
                local = attention_policy == "windowed" or block_index % 2 == 0
                stage.append(PyramidBlock(
                    dim, num_heads, mixer, rank, next(rates), 1e-4,
                    window_size=window_size if local else None,
                    **mixer_kwargs,
                ))
            self.stages.append(stage)
        self.stage_norms = nn.ModuleList(RMSNorm(dim) for dim in dims)
        self.pos_embed = nn.ParameterList()
        for index, dim in enumerate(dims):
            side = image_size // (4 * 2**index)
            self.pos_embed.append(
                nn.Parameter(torch.zeros(1, side * side, dim), requires_grad=learned_position)
            )
        self.head = nn.Linear(dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.learned_position = learned_position
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _add_position(self, x: torch.Tensor, stage: int, shape: tuple[int, int]):
        if not self.learned_position:
            return x
        pos = self.pos_embed[stage]
        source = int(pos.shape[1] ** 0.5)
        if source * source != pos.shape[1]:
            raise RuntimeError("pyramid position table must have a square source grid")
        if shape != (source, source):
            pos = pos.reshape(1, source, source, -1).permute(0, 3, 1, 2)
            pos = F.interpolate(pos.float(), size=shape, mode="bicubic", align_corners=False)
            pos = pos.to(x.dtype).flatten(2).transpose(1, 2)
        return x + pos.to(device=x.device, dtype=x.dtype)

    def forward_features(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = []
        x = image
        for index, (embed, blocks, norm) in enumerate(
            zip(self.patch_embeds, self.stages, self.stage_norms)
        ):
            x, shape = embed(x)
            x = self._add_position(x, index, shape)
            for block in blocks:
                x = block(x, spatial_shape=shape)
            x = norm(x)
            if index in self.out_indices:
                outputs.append(
                    x.view(image.shape[0], *shape, self.dims[index])
                    .permute(0, 3, 1, 2).contiguous()
                )
            if index + 1 < len(self.stages):
                x = x.view(image.shape[0], *shape, self.dims[index]).permute(0, 3, 1, 2).contiguous()
        return tuple(outputs)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(image)
        return self.head(features[-1].mean(dim=(-2, -1)))


def create_pyramid_vision_llama(
    scale: str = "small", *, mixer: str = "rrlsso", rank: int = 32, **kwargs
) -> PyramidVisionLLaMA:
    try:
        config = PYRAMID_CONFIGS[scale]
    except KeyError as exc:
        raise ValueError(f"unknown pyramid scale {scale!r}") from exc
    return PyramidVisionLLaMA(
        dims=config.dims, depths=config.depths, heads=config.heads,
        drop_path_rate=config.drop_path_rate, mixer=mixer, rank=rank, **kwargs,
    )
