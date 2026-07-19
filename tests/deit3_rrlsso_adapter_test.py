from __future__ import annotations

import pytest
import torch

from experiments.cv_vit_rrlsso_cifar100 import TimmRRLSSOAttention


def test_timm_rrlsso_attention_uses_ordinary_rank_rotary_and_backpropagates() -> None:
    module = TimmRRLSSOAttention(
        embed_dim=64,
        num_heads=4,
        rank=8,
        dropout=0.0,
        bias=True,
        gain_init=1.4426742274994273,
        alpha_init=1.0776072417497349,
        length_normalize=True,
        length_reference=1.0,
    )
    x = torch.randn(2, 17, 64, requires_grad=True)
    output = module(x, attn_mask=None, is_causal=False)
    assert output.shape == x.shape
    assert module.rrlsso._rotary_cos_cache.shape == (1, 1, 17, 4)
    output.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_timm_rrlsso_attention_rejects_causal_mode() -> None:
    module = TimmRRLSSOAttention(
        embed_dim=32,
        num_heads=4,
        rank=8,
        dropout=0.0,
        bias=True,
        gain_init=1.4426742274994273,
        alpha_init=1.0776072417497349,
        length_normalize=True,
        length_reference=1.0,
    )
    with pytest.raises(ValueError, match="not causal"):
        module(torch.randn(1, 9, 32), is_causal=True)
