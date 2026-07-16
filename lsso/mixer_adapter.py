from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import LSSO
from .modules_v2 import RRLSSO
from .rotary_2d import (
    apply_2d_rotary,
    apply_rotary_factors,
    build_2d_rotary_factors,
)


class RotaryMHA(nn.Module):
    """Batch-first MHA with optional separable 2-D RoPE and SDPA backend."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim, self.num_heads = dim, num_heads
        self.head_dim = dim // num_heads
        self.dropout = float(dropout)
        self.qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.register_buffer("_rotary_cos", torch.empty(0), persistent=False)
        self.register_buffer("_rotary_sin", torch.empty(0), persistent=False)
        self._rotary_cache_key: tuple | None = None

    def _grid_factors(self, x: torch.Tensor, spatial_shape, num_prefix_tokens):
        key = (spatial_shape, num_prefix_tokens, self.head_dim, x.device, x.dtype)
        if self._rotary_cache_key != key or self._rotary_cos.is_inference() != x.is_inference():
            self._rotary_cos, self._rotary_sin = build_2d_rotary_factors(
                self.head_dim, spatial_shape=spatial_shape,
                num_prefix_tokens=num_prefix_tokens, device=x.device, dtype=x.dtype,
            )
            self._rotary_cache_key = key
        return self._rotary_cos, self._rotary_sin

    def forward(
        self,
        x: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
        padding_ratio_hint: float | None = None,
        spatial_shape: tuple[int, int] | None = None,
        position_coords: torch.Tensor | None = None,
        num_prefix_tokens: int = 0,
    ) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        if spatial_shape is not None and position_coords is None:
            cos, sin = self._grid_factors(x, spatial_shape, num_prefix_tokens)
            q = apply_rotary_factors(q, cos, sin)
            k = apply_rotary_factors(k, cos, sin)
        elif position_coords is not None:
            q = apply_2d_rotary(q, position_coords=position_coords)
            k = apply_2d_rotary(k, position_coords=position_coords)
        attn_mask = None
        if valid_mask is not None:
            if valid_mask.shape != (B, N):
                raise ValueError(f"valid_mask must have shape {(B, N)}")
            attn_mask = valid_mask[:, None, None, :].to(device=x.device)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, N, D)
        y = self.proj(y)
        if valid_mask is not None:
            y = y * valid_mask[:, :, None].to(device=y.device, dtype=y.dtype)
        return y


class MixerAdapter(nn.Module):
    """Unified MHA/LSSO/RRLSSO interface for vision and sequence backbones."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mixer: str = "mha",
        *,
        rank: int = 32,
        dropout: float = 0.0,
        bias: bool = True,
        rotary_2d: bool = True,
        **lsso_kwargs,
    ) -> None:
        super().__init__()
        name = mixer.lower().replace("_", "-")
        self.mixer_name = name
        self.rotary_2d = bool(rotary_2d)
        if name == "mha":
            self.mixer = RotaryMHA(dim, num_heads, dropout=dropout, bias=bias)
        elif name == "lsso":
            self.mixer = LSSO(dim, num_heads, rank=rank, dropout=dropout, bias=bias, **lsso_kwargs)
        elif name in {"rrlsso", "rope-lsso"}:
            self.mixer = RRLSSO(dim, num_heads, rank=rank, dropout=dropout, bias=bias, **lsso_kwargs)
        else:
            raise ValueError(f"unknown mixer {mixer!r}; expected mha, lsso, or rrlsso")

    def forward(
        self,
        x: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
        padding_ratio_hint: float | None = None,
        spatial_shape: tuple[int, int] | None = None,
        position_coords: torch.Tensor | None = None,
        num_prefix_tokens: int = 0,
    ) -> torch.Tensor:
        positional = spatial_shape is not None or position_coords is not None
        if self.mixer_name == "mha":
            return self.mixer(
                x, valid_mask=valid_mask,
                spatial_shape=spatial_shape if self.rotary_2d else None,
                position_coords=position_coords if self.rotary_2d else None,
                num_prefix_tokens=num_prefix_tokens,
            )
        if self.mixer_name == "lsso":
            return self.mixer(
                x,
                valid_mask=valid_mask,
                padding_ratio_hint=padding_ratio_hint,
            )
        return self.mixer(
            x,
            valid_mask=valid_mask,
            padding_ratio_hint=padding_ratio_hint,
            spatial_shape=spatial_shape if self.rotary_2d and positional else None,
            position_coords=position_coords if self.rotary_2d and positional else None,
            num_prefix_tokens=num_prefix_tokens,
        )
