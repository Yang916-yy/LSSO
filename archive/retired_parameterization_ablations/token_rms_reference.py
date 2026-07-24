"""Historical per-token RMS ablation; not part of the supported package."""

import torch


def token_rms_basis(U: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return U * torch.rsqrt(torch.mean(U * U, dim=-1, keepdim=True) + eps)
