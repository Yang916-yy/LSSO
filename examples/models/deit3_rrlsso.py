"""timm-registered DeiT III backbones with RRLSSO token mixing."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import timm
from timm.layers import DropPath, trunc_normal_
from timm.models import register_model

from lsso import RRLSSO


DEIT3_RRLSSO_MODELS = {
    "deit3_small_patch16_rrlsso": "deit3_small_patch16_224",
    "deit3_base_patch16_rrlsso": "deit3_base_patch16_224",
    "deit3_large_patch16_rrlsso": "deit3_large_patch16_224",
}


class TimmRRLSSOAttention(nn.Module):
    """Drop-in bidirectional replacement for ``timm`` ViT attention."""

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        rank: int = 32,
        dropout: float = 0.0,
        bias: bool = True,
        gain_init: float = 1.0,
        alpha_init: float = 1.2,
        solve_parameterization: str = "gain_alpha",
        alpha_max: float = 3.0,
        basis_normalization: str = "trace",
        length_normalize: bool = True,
        length_reference: float = 1.0,
    ) -> None:
        super().__init__()
        self.rrlsso = RRLSSO(
            dim=embed_dim,
            num_heads=num_heads,
            rank=rank,
            dropout=dropout,
            bias=bias,
            gain_init=gain_init,
            alpha_init=alpha_init,
            solve_parameterization=solve_parameterization,
            alpha_max=alpha_max,
            basis_normalization=basis_normalization,
            length_normalize=length_normalize,
            length_reference=length_reference,
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if attn_mask is not None:
            raise ValueError("DeiT III image classification does not use an attention mask")
        if is_causal:
            raise ValueError("RRLSSO is a bidirectional token mixer")
        return self.rrlsso(x)


def replace_timm_attention_with_rrlsso(
    model: nn.Module,
    *,
    rank: int = 32,
    gain_init: float = 1.0,
    alpha_init: float = 1.2,
    solve_parameterization: str = "gain_alpha",
    alpha_max: float = 3.0,
    basis_normalization: str = "trace",
    length_normalize: bool = True,
    length_reference: float = 1.0,
) -> int:
    """Replace every DeiT/ViT attention block while retaining the backbone."""

    if not hasattr(model, "blocks"):
        raise TypeError(f"expected a timm VisionTransformer, got {type(model).__name__}")
    replaced = 0
    for block in model.blocks:
        old = block.attn
        if not hasattr(old, "qkv") or not hasattr(old, "num_heads"):
            raise TypeError(f"unsupported timm attention module {type(old).__name__}")
        block.attn = TimmRRLSSOAttention(
            embed_dim=int(old.qkv.in_features),
            num_heads=int(old.num_heads),
            rank=rank,
            dropout=float(old.attn_drop.p),
            bias=old.qkv.bias is not None,
            gain_init=gain_init,
            alpha_init=alpha_init,
            solve_parameterization=solve_parameterization,
            alpha_max=alpha_max,
            basis_normalization=basis_normalization,
            length_normalize=length_normalize,
            length_reference=length_reference,
        )
        replaced += 1
    return replaced


def _create_deit3_rrlsso(
    base_model: str,
    *,
    pretrained: bool,
    **kwargs: Any,
) -> nn.Module:
    if pretrained:
        raise ValueError(
            "pretrained MHA weights cannot be loaded into an RRLSSO mixer; "
            "train this registered model from scratch"
        )
    rank = int(kwargs.pop("rank", 32))
    mixer_kwargs = {
        "gain_init": float(kwargs.pop("gain_init", 1.0)),
        "alpha_init": float(kwargs.pop("alpha_init", 1.2)),
        "solve_parameterization": kwargs.pop("solve_parameterization", "gain_alpha"),
        "alpha_max": float(kwargs.pop("alpha_max", 3.0)),
        "basis_normalization": kwargs.pop("basis_normalization", "trace"),
        "length_normalize": bool(kwargs.pop("length_normalize", True)),
        "length_reference": float(kwargs.pop("length_reference", 1.0)),
    }
    # Meta's released DeiT-III implementation uses LayerScale=1e-4 and a
    # constant stochastic-depth rate in every block. timm's current DeiT-III
    # defaults use 1e-6 and a depth-wise schedule, so make the official
    # training architecture explicit here.
    kwargs.setdefault("img_size", 224)
    kwargs.setdefault("init_values", 1e-4)
    drop_path_rate = float(kwargs.pop("drop_path_rate", 0.0))
    model = timm.create_model(base_model, pretrained=False, **kwargs)
    # Meta's implementation initializes the class token with the same
    # truncated-normal std=0.02 used for learned positional embeddings. Current
    # timm initializes it near zero (std=1e-6).
    trunc_normal_(model.cls_token, std=0.02)
    for block in model.blocks:
        drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        block.drop_path1 = drop_path
        block.drop_path2 = drop_path
    replaced = replace_timm_attention_with_rrlsso(
        model,
        rank=rank,
        **mixer_kwargs,
    )
    model.rrlsso_config = {
        "rank": rank,
        "replaced_layers": replaced,
        "rank_rotary": "ordinary-1d",
        "layerscale_init": 1e-4,
        "cls_token_init_std": 0.02,
        "constant_drop_path_rate": drop_path_rate,
        **mixer_kwargs,
    }
    return model


@register_model
def deit3_small_patch16_rrlsso(
    pretrained: bool = False, **kwargs: Any
) -> nn.Module:
    return _create_deit3_rrlsso(
        "deit3_small_patch16_224", pretrained=pretrained, **kwargs
    )


@register_model
def deit3_base_patch16_rrlsso(
    pretrained: bool = False, **kwargs: Any
) -> nn.Module:
    return _create_deit3_rrlsso(
        "deit3_base_patch16_224", pretrained=pretrained, **kwargs
    )


@register_model
def deit3_large_patch16_rrlsso(
    pretrained: bool = False, **kwargs: Any
) -> nn.Module:
    return _create_deit3_rrlsso(
        "deit3_large_patch16_224", pretrained=pretrained, **kwargs
    )


__all__ = [
    "DEIT3_RRLSSO_MODELS",
    "TimmRRLSSOAttention",
    "deit3_base_patch16_rrlsso",
    "deit3_large_patch16_rrlsso",
    "deit3_small_patch16_rrlsso",
    "replace_timm_attention_with_rrlsso",
]
