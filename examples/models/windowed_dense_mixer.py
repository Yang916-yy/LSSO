from __future__ import annotations

import torch
import torch.nn.functional as F

from .vision_llama import VisionLLaMABlock


def partition_windows(
    x: torch.Tensor,
    spatial_shape: tuple[int, int],
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    """Partition channel-last image tokens and return a padding-valid mask."""
    if window_size < 1:
        raise ValueError("window_size must be positive")
    batch, tokens, dim = x.shape
    height, width = spatial_shape
    if tokens != height * width:
        raise ValueError(f"token count {tokens} does not match grid {spatial_shape}")
    padded_h = (height + window_size - 1) // window_size * window_size
    padded_w = (width + window_size - 1) // window_size * window_size
    grid = x.reshape(batch, height, width, dim)
    grid = F.pad(grid, (0, 0, 0, padded_w - width, 0, padded_h - height))
    valid = torch.zeros(
        batch, padded_h, padded_w, dtype=torch.bool, device=x.device
    )
    valid[:, :height, :width] = True
    windows = (
        grid.reshape(
            batch,
            padded_h // window_size,
            window_size,
            padded_w // window_size,
            window_size,
            dim,
        )
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(-1, window_size * window_size, dim)
    )
    masks = (
        valid.reshape(
            batch,
            padded_h // window_size,
            window_size,
            padded_w // window_size,
            window_size,
        )
        .permute(0, 1, 3, 2, 4)
        .reshape(-1, window_size * window_size)
    )
    return windows, masks, (padded_h, padded_w)


def unpartition_windows(
    windows: torch.Tensor,
    *,
    batch: int,
    spatial_shape: tuple[int, int],
    window_size: int,
    padded_shape: tuple[int, int],
) -> torch.Tensor:
    height, width = spatial_shape
    padded_h, padded_w = padded_shape
    dim = windows.shape[-1]
    expected = batch * (padded_h // window_size) * (padded_w // window_size)
    if windows.shape[0] != expected:
        raise ValueError(f"expected {expected} windows, got {windows.shape[0]}")
    grid = (
        windows.reshape(
            batch,
            padded_h // window_size,
            padded_w // window_size,
            window_size,
            window_size,
            dim,
        )
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch, padded_h, padded_w, dim)
    )
    return grid[:, :height, :width].reshape(batch, height * width, dim)


def run_windowed_block(
    block: VisionLLaMABlock,
    x: torch.Tensor,
    *,
    spatial_shape: tuple[int, int],
    window_size: int,
) -> torch.Tensor:
    windows, valid_mask, padded_shape = partition_windows(
        x, spatial_shape, window_size
    )
    windows = block(
        windows,
        spatial_shape=(window_size, window_size),
        valid_mask=valid_mask,
        num_prefix_tokens=0,
    )
    return unpartition_windows(
        windows,
        batch=x.shape[0],
        spatial_shape=spatial_shape,
        window_size=window_size,
        padded_shape=padded_shape,
    )
