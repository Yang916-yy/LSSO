from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .flash_bidir_lsso import flash_bidir_lsso
except ImportError:  # Allow running this file directly from the prototype directory.
    from flash_bidir_lsso import flash_bidir_lsso
from lsso.modules import length_normalize_basis
from lsso.modules_v2 import apply_rank_rotary


class BidirLSSOFlashPrototype(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        rank: int = 32,
        dropout: float = 0.0,
        eps: float = 1e-5,
        gamma_max: float = 1.2,
        theta_gamma_init: float = 0.5,
        no_global: bool = False,
        normalize_u: bool = True,
        length_normalize: bool = True,
        length_reference: float = 1.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        if rank not in (16, 32):
            raise ValueError("flash bidir prototype currently supports rank 16 or 32.")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rank = rank
        self.eps = eps
        self.gamma_max = gamma_max
        self.no_global = no_global
        self.normalize_u = normalize_u
        self.length_normalize = length_normalize
        if length_reference <= 0:
            raise ValueError(f"length_reference must be positive, got {length_reference}")
        self.length_reference = float(length_reference)
        self.dropout_p = dropout
        self.uc_dim = num_heads * rank + dim
        self.w_uc = nn.Linear(dim, self.uc_dim, bias=bias)
        self.w_o = nn.Linear(dim, dim, bias=bias)
        self.theta_mu = nn.Parameter(torch.zeros(num_heads))
        self.theta_gamma = nn.Parameter(torch.full((num_heads,), float(theta_gamma_init), dtype=torch.float32))

    def _project_uc(self, x: torch.Tensor, position_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, D = x.shape
        H = self.num_heads
        r = self.rank
        dh = self.head_dim
        UC = self.w_uc(x)
        U, C = UC.split((H * r, D), dim=-1)
        U = U.view(B, N, H, r).transpose(1, 2).contiguous()
        C = C.view(B, N, H, dh).transpose(1, 2).contiguous()
        if self.normalize_u:
            U = U * torch.rsqrt(torch.mean(U * U, dim=-1, keepdim=True) + self.eps)
        return U, C

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, D = x.shape
        U, C = self._project_uc(x)
        if valid_mask is not None:
            mask = valid_mask[:, None, :, None].to(dtype=x.dtype)
            U = U * mask
            C = C * mask
        if self.length_normalize:
            U = length_normalize_basis(
                U,
                valid_mask,
                reference_length=self.length_reference,
            )

        mu = F.softplus(self.theta_mu) + self.eps
        gamma = self.gamma_max * torch.sigmoid(self.theta_gamma)
        if self.no_global:
            gamma = torch.zeros_like(gamma)
        Y = flash_bidir_lsso(U, C, mu, gamma)
        Y = Y.transpose(1, 2).contiguous().view(B, N, D)
        Y = self.w_o(Y)
        if valid_mask is not None:
            Y = Y * valid_mask[:, :, None].to(dtype=Y.dtype)
        if self.dropout_p:
            Y = F.dropout(Y, p=self.dropout_p, training=self.training)
        return Y


class BidirRRLSSOFlashPrototype(BidirLSSOFlashPrototype):
    def __init__(
        self,
        *args,
        rope_base: float = 10000.0,
        rope_scale: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.rope_base = rope_base
        self.rope_scale = rope_scale

    def _project_uc(self, x: torch.Tensor, position_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        U, C = super()._project_uc(x, position_ids=position_ids)
        U = apply_rank_rotary(U, position_ids, base=self.rope_base, scale=self.rope_scale)
        return U, C

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, D = x.shape
        U, C = self._project_uc(x, position_ids=position_ids)
        if valid_mask is not None:
            mask = valid_mask[:, None, :, None].to(dtype=x.dtype)
            U = U * mask
            C = C * mask
        if self.length_normalize:
            U = length_normalize_basis(
                U,
                valid_mask,
                reference_length=self.length_reference,
            )

        mu = F.softplus(self.theta_mu) + self.eps
        gamma = self.gamma_max * torch.sigmoid(self.theta_gamma)
        if self.no_global:
            gamma = torch.zeros_like(gamma)
        Y = flash_bidir_lsso(U, C, mu, gamma)
        Y = Y.transpose(1, 2).contiguous().view(B, N, D)
        Y = self.w_o(Y)
        if valid_mask is not None:
            Y = Y * valid_mask[:, :, None].to(dtype=Y.dtype)
        if self.dropout_p:
            Y = F.dropout(Y, p=self.dropout_p, training=self.training)
        return Y
