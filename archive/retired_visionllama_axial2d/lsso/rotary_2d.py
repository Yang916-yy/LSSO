from __future__ import annotations

import torch

from .mathdx_backend import try_rank_rotary


def make_2d_position_coords(
    spatial_shape: tuple[int, int],
    *,
    num_prefix_tokens: int = 0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return row-major ``[N, 2]`` coordinates in ``(x, y)`` order."""
    height, width = spatial_shape
    if height <= 0 or width <= 0:
        raise ValueError(f"spatial_shape must be positive, got {spatial_shape}")
    if num_prefix_tokens < 0:
        raise ValueError("num_prefix_tokens must be non-negative")
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    coords = torch.stack((x, y), dim=-1).reshape(height * width, 2)
    if num_prefix_tokens:
        prefix = torch.zeros(num_prefix_tokens, 2, device=device, dtype=dtype)
        coords = torch.cat((prefix, coords), dim=0)
    return coords


def _rotate_axis(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    even, odd = x[..., 0::2], x[..., 1::2]
    cos = angles.cos().to(x.dtype)
    sin = angles.sin().to(x.dtype)
    out = torch.empty_like(x)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return out


def build_2d_rotary_factors(
    dim: int,
    *,
    spatial_shape: tuple[int, int] | None = None,
    position_coords: torch.Tensor | None = None,
    num_prefix_tokens: int = 0,
    base: float = 10000.0,
    scale: float = 1.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build reusable 2-D cosine/sine factors shaped ``[...,N,D/2]``."""
    if dim % 4:
        raise ValueError(f"2-D rotary requires D divisible by 4, got D={dim}")
    calc_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    if position_coords is None:
        if spatial_shape is None:
            raise ValueError("spatial_shape or position_coords is required")
        coords = make_2d_position_coords(
            spatial_shape, num_prefix_tokens=num_prefix_tokens,
            device=device, dtype=calc_dtype,
        )
    else:
        coords = position_coords.to(device=device, dtype=calc_dtype)
        if coords.ndim not in (2, 3) or coords.shape[-1] != 2:
            raise ValueError("position_coords must have shape [N,2] or [B,N,2]")
    pair_count = dim // 4
    inv_freq = base ** (
        -torch.arange(pair_count, device=coords.device, dtype=calc_dtype) / pair_count
    )
    angles_x = scale * coords[..., 0, None] * inv_freq
    angles_y = scale * coords[..., 1, None] * inv_freq
    factors = torch.cat((angles_x, angles_y), dim=-1)
    leading = (1, 1) if coords.ndim == 2 else (coords.shape[0], 1)
    factors = factors.view(*leading, coords.shape[-2], dim // 2)
    return factors.cos().to(dtype).contiguous(), factors.sin().to(dtype).contiguous()


def apply_rotary_factors(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply precomputed interleaved-pair factors, using CUDA when available."""
    native_input = x.contiguous() if x.is_cuda and not x.is_contiguous() else x
    fused = try_rank_rotary(native_input, cos, sin)
    if fused is not None:
        return fused
    even, odd = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return out


def apply_2d_rotary(
    x: torch.Tensor,
    *,
    spatial_shape: tuple[int, int] | None = None,
    position_coords: torch.Tensor | None = None,
    num_prefix_tokens: int = 0,
    base: float = 10000.0,
    scale: float = 1.0,
) -> torch.Tensor:
    """Apply fixed separable 2-D rotation to ``[B,H,N,D]`` features.

    Half of the channel pairs encode x and half encode y. Explicit coordinates
    support shifted windows, batched layouts, and non-square feature maps.
    Prefix tokens use zero coordinates when coordinates are generated here.
    """
    if x.ndim != 4:
        raise ValueError(f"x must have shape [B,H,N,D], got {tuple(x.shape)}")
    B, _H, N, dim = x.shape
    if dim % 4:
        raise ValueError(f"2-D rotary requires D divisible by 4, got D={dim}")
    if base <= 0:
        raise ValueError(f"base must be positive, got {base}")
    calc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    if position_coords is None:
        if spatial_shape is None:
            raise ValueError("spatial_shape or position_coords is required")
        expected = num_prefix_tokens + spatial_shape[0] * spatial_shape[1]
        if expected != N:
            raise ValueError(f"prefix + H*W must equal N={N}, got {expected}")
        coords = make_2d_position_coords(
            spatial_shape,
            num_prefix_tokens=num_prefix_tokens,
            device=x.device,
            dtype=calc_dtype,
        )
    else:
        coords = position_coords.to(device=x.device, dtype=calc_dtype)
        if coords.shape not in ((N, 2), (B, N, 2)):
            raise ValueError(
                f"position_coords must have shape {(N, 2)} or {(B, N, 2)}, "
                f"got {tuple(coords.shape)}"
            )
    cos, sin = build_2d_rotary_factors(
        dim, position_coords=coords, base=base, scale=scale,
        device=x.device, dtype=x.dtype,
    )
    return apply_rotary_factors(x, cos, sin)


def apply_2d_rank_rotary(U: torch.Tensor, **kwargs) -> torch.Tensor:
    """Semantic alias for applying 2-D rotary to an RRLSSO relation basis."""
    return apply_2d_rotary(U, **kwargs)
