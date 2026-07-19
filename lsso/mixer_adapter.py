from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import LSSO
from .modules_v2 import RRLSSO


class RotaryMHA(nn.Module):
    """Batch-first MHA with optional 1-D/2-D RoPE and SDPA backend."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        rotary_1d: bool = False,
    ):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim, self.num_heads = dim, num_heads
        self.head_dim = dim // num_heads
        self.dropout = float(dropout)
        self.rotary_1d = bool(rotary_1d)
        if self.rotary_1d and self.head_dim % 2:
            raise ValueError("1-D RoPE requires an even head dimension")
        self.qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.register_buffer("_rotary_1d_cos", torch.empty(0), persistent=False)
        self.register_buffer("_rotary_1d_sin", torch.empty(0), persistent=False)
        self._rotary_1d_cache_key: tuple | None = None

    def _sequence_factors(self, x: torch.Tensor, length: int):
        key = (length, self.head_dim, x.device, x.dtype)
        if (
            self._rotary_1d_cache_key != key
            or self._rotary_1d_cos.is_inference() != x.is_inference()
        ):
            calc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
            half = self.head_dim // 2
            inv_freq = 10000.0 ** (
                -torch.arange(half, device=x.device, dtype=calc_dtype) / half
            )
            positions = torch.arange(length, device=x.device, dtype=calc_dtype)
            angles = positions[:, None] * inv_freq[None, :]
            self._rotary_1d_cos = angles.cos().to(x.dtype).view(1, 1, length, half)
            self._rotary_1d_sin = angles.sin().to(x.dtype).view(1, 1, length, half)
            self._rotary_1d_cache_key = key
        return self._rotary_1d_cos, self._rotary_1d_sin

    @staticmethod
    def _apply_1d_rotary(
        value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        even, odd = value[..., 0::2], value[..., 1::2]
        output = torch.empty_like(value)
        output[..., 0::2] = even * cos - odd * sin
        output[..., 1::2] = even * sin + odd * cos
        return output

    def forward(
        self,
        x: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
        padding_ratio_hint: float | None = None,
    ) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        if self.rotary_1d:
            cos, sin = self._sequence_factors(x, N)
            q = self._apply_1d_rotary(q, cos, sin)
            k = self._apply_1d_rotary(k, cos, sin)
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
        rotary_1d: bool = False,
        **lsso_kwargs,
    ) -> None:
        super().__init__()
        name = mixer.lower().replace("_", "-")
        self.mixer_name = name
        if name == "mha":
            self.mixer = RotaryMHA(
                dim,
                num_heads,
                dropout=dropout,
                bias=bias,
                rotary_1d=rotary_1d,
            )
        elif name == "lsso":
            self.mixer = LSSO(dim, num_heads, rank=rank, dropout=dropout, bias=bias, **lsso_kwargs)
        elif name == "rrlsso":
            self.mixer = RRLSSO(dim, num_heads, rank=rank, dropout=dropout, bias=bias, **lsso_kwargs)
        else:
            raise ValueError(f"unknown mixer {mixer!r}; expected mha, lsso, or rrlsso")

    def forward(
        self,
        x: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
        padding_ratio_hint: float | None = None,
    ) -> torch.Tensor:
        if self.mixer_name == "mha":
            return self.mixer(
                x, valid_mask=valid_mask,
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
        )
