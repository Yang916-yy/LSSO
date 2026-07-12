from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        x = ((x - mean) * torch.rsqrt(variance + self.eps)).to(dtype)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


def _output_projection(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        LayerNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        LayerNorm2d(out_channels),
    )


class SimpleFeaturePyramid(nn.Module):
    """ViTDet-style P2--P5 pyramid from one stride-16 feature map."""

    output_strides = (4, 8, 16, 32)

    def __init__(self, in_channels: int, out_channels: int = 256) -> None:
        super().__init__()
        if in_channels < 4:
            raise ValueError("in_channels must be at least four")
        half = max(1, in_channels // 2)
        quarter = max(1, in_channels // 4)
        self.scale_4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, half, kernel_size=2, stride=2),
            LayerNorm2d(half),
            nn.GELU(),
            nn.ConvTranspose2d(half, quarter, kernel_size=2, stride=2),
            _output_projection(quarter, out_channels),
        )
        self.scale_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, half, kernel_size=2, stride=2),
            _output_projection(half, out_channels),
        )
        self.scale_1 = _output_projection(in_channels, out_channels)
        self.scale_half = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            _output_projection(in_channels, out_channels),
        )

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if feature.ndim != 4:
            raise ValueError(f"expected BCHW input, got shape {tuple(feature.shape)}")
        return (
            self.scale_4(feature),
            self.scale_2(feature),
            self.scale_1(feature),
            self.scale_half(feature),
        )
