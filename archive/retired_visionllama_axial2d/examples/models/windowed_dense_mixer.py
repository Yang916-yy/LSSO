from __future__ import annotations

import torch
import torch.nn.functional as F

from .vision_llama import VisionLLaMABlock


def partition_windows(
    x: torch.Tensor,
    spatial_shape: tuple[int, int],
    window_size: int,
    *,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, tuple[int, int]]:
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
    needs_padding = padded_h != height or padded_w != width
    if needs_padding:
        grid = F.pad(grid, (0, 0, 0, padded_w - width, 0, padded_h - height))
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
    masks = valid_mask
    if needs_padding and masks is None:
        masks = build_window_valid_mask(
            batch=batch,
            spatial_shape=spatial_shape,
            window_size=window_size,
            device=x.device,
        )
    if masks is not None and masks.shape != windows.shape[:2]:
        raise ValueError(
            f"valid mask shape {tuple(masks.shape)} does not match windows "
            f"{tuple(windows.shape[:2])}"
        )
    return windows, masks, (padded_h, padded_w)


def build_window_valid_mask(
    *,
    batch: int,
    spatial_shape: tuple[int, int],
    window_size: int,
    device: torch.device,
) -> torch.Tensor:
    height, width = spatial_shape
    padded_h = (height + window_size - 1) // window_size * window_size
    padded_w = (width + window_size - 1) // window_size * window_size
    valid = torch.zeros(batch, padded_h, padded_w, dtype=torch.bool, device=device)
    valid[:, :height, :width] = True
    return (
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
    height, width = spatial_shape
    needs_padding = height % window_size != 0 or width % window_size != 0
    valid_mask = None
    if needs_padding:
        cache = getattr(block, "_dense_window_mask_cache", None)
        if cache is None:
            cache = {}
            block._dense_window_mask_cache = cache
        key = (x.shape[0], spatial_shape, window_size, x.device)
        valid_mask = cache.get(key)
        if valid_mask is None:
            if len(cache) >= 4:
                cache.pop(next(iter(cache)))
            valid_mask = build_window_valid_mask(
                batch=x.shape[0],
                spatial_shape=spatial_shape,
                window_size=window_size,
                device=x.device,
            )
            cache[key] = valid_mask
    windows, valid_mask, padded_shape = partition_windows(
        x, spatial_shape, window_size, valid_mask=valid_mask
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
