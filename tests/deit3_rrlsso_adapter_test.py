from __future__ import annotations

import pytest
import torch
import timm

from examples.models.deit3_rrlsso import (
    DEIT3_RRLSSO_MODELS,
    TimmRRLSSOAttention,
)


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
    with pytest.raises(ValueError, match="bidirectional"):
        module(torch.randn(1, 9, 32), is_causal=True)


def test_all_deit3_rrlsso_models_are_registered() -> None:
    assert set(DEIT3_RRLSSO_MODELS) == {
        "deit3_small_patch16_rrlsso",
        "deit3_base_patch16_rrlsso",
        "deit3_large_patch16_rrlsso",
    }
    assert all(timm.is_model(name) for name in DEIT3_RRLSSO_MODELS)
    assert not any(
        timm.is_model(f"deit3_{size}_patch16_192_rrlsso")
        for size in ("small", "base", "large")
    )


def test_registered_small_model_replaces_every_attention_block() -> None:
    model = timm.create_model(
        "deit3_small_patch16_rrlsso",
        img_size=32,
        num_classes=7,
        rank=8,
        drop_path_rate=0.05,
    )
    assert len(model.blocks) == 12
    assert model.rrlsso_config["replaced_layers"] == 12
    assert model.rrlsso_config["rank_rotary"] == "ordinary-1d"
    assert model.rrlsso_config["layerscale_init"] == 1e-4
    assert model.rrlsso_config["constant_drop_path_rate"] == 0.05
    assert model.blocks[0].ls1.gamma[0].item() == pytest.approx(1e-4)
    assert model.blocks[0].drop_path1.drop_prob == pytest.approx(0.05)
    assert model.blocks[-1].drop_path1.drop_prob == pytest.approx(0.05)
    assert all(isinstance(block.attn, TimmRRLSSOAttention) for block in model.blocks)
    assert model(torch.randn(1, 3, 32, 32)).shape == (1, 7)
