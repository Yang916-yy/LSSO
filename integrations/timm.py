"""Thin timm VisionTransformer adapter for LSSO."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, Sequence, cast

import torch
import torch.nn as nn

from lsso import CoreMode, LSSO, LSSOConfig


_Implementation = Literal["reference", "cuda"]


@dataclass(frozen=True)
class DeiT3Spec:
    """The published DeiT III geometry and regularization for one scale."""

    embed_dim: int
    depth: int
    num_heads: int
    drop_path_rate: float


_DEIT3_SPECS: dict[str, DeiT3Spec] = {
    "small": DeiT3Spec(384, 12, 6, 0.05),
    "base": DeiT3Spec(768, 12, 12, 0.20),
    "large": DeiT3Spec(1024, 24, 16, 0.45),
}
_DEIT3_DEFAULT_RANKS: dict[str, int] = {
    "small": 32,
    "base": 48,
    "large": 64,
}


@dataclass(frozen=True)
class VisionTokenLayout:
    """Token validity and Rank-Rotary coordinates for one vision forward."""

    valid_mask: torch.Tensor | None
    position_ids: torch.Tensor | None


def deit3_spec(variant: str) -> DeiT3Spec:
    """Return the official DeiT III Small, Base, or Large geometry."""

    try:
        return _DEIT3_SPECS[variant]
    except KeyError as error:
        choices = ", ".join(_DEIT3_SPECS)
        raise ValueError(
            f"variant must be one of {{{choices}}}, got {variant!r}"
        ) from error


def deit3_default_rank(variant: str) -> int:
    """Return the agreed CUDA-fast LSSO rank for a DeiT III scale."""

    deit3_spec(variant)
    return _DEIT3_DEFAULT_RANKS[variant]


def _validate_implementation(implementation: str) -> _Implementation:
    if implementation not in ("reference", "cuda"):
        raise ValueError(
            "implementation must be 'reference' or 'cuda', "
            f"got {implementation!r}"
        )
    return cast(_Implementation, implementation)


def _require_timm_attention_mask_api(vision_transformer: type[nn.Module]) -> None:
    """Reject timm releases predating the token-layout forwarding contract."""

    required = ("forward", "forward_intermediates")
    missing = [
        name
        for name in required
        if "attn_mask" not in inspect.signature(
            getattr(vision_transformer, name)
        ).parameters
    ]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            "LSSO vision adapters require timm>=1.0.16 with attn_mask support "
            f"in VisionTransformer.{joined}"
        )


class _TimmLSSOMixer(nn.Module):
    """Adapt LSSO to timm's attention call signature without owning math."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        rank: int,
        core_mode: CoreMode,
        rank_rotary: bool,
        implementation: _Implementation,
        qkv_bias: bool,
        qk_norm: bool,
        scale_norm: bool,
        proj_bias: bool,
        attn_drop: float,
        proj_drop: float,
        norm_layer: Any,
        depth: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        del norm_layer, depth
        self.implementation = _validate_implementation(implementation)
        if qk_norm or scale_norm:
            raise ValueError("LSSO does not implement qk_norm or attention scale_norm")
        if attn_drop != 0.0 or proj_drop != 0.0:
            raise ValueError("mixer dropout belongs outside the LSSO operator")
        if qkv_bias != proj_bias:
            raise ValueError("LSSO requires matching input and output bias settings")

        self.mixer = LSSO(
            LSSOConfig(
                dim=dim,
                num_heads=num_heads,
                rank=rank,
                core_mode=core_mode,
                rank_rotary=rank_rotary,
                bias=qkv_bias,
            )
        )
        if device is not None or dtype is not None:
            self.mixer.to(device=device, dtype=dtype)

    @staticmethod
    def _vision_position_ids(length: int, device: torch.device) -> torch.Tensor:
        """Return zero-phase CLS and centered one-dimensional patch coordinates."""

        if length <= 0:
            return torch.empty(0, device=device, dtype=torch.float32)
        patch_positions = torch.arange(
            length - 1,
            device=device,
            dtype=torch.float32,
        )
        patch_positions -= 0.5 * (length - 2)
        return torch.cat(
            (torch.zeros(1, device=device, dtype=torch.float32), patch_positions)
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | VisionTokenLayout | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if is_causal:
            raise ValueError(
                "the vision adapter accepts only unmasked bidirectional input"
            )
        if attn_mask is None:
            layout = None
        elif isinstance(attn_mask, VisionTokenLayout):
            layout = attn_mask
        else:
            raise ValueError(
                "the vision adapter accepts only its token validity layout, "
                "not a generic attention mask"
            )
        valid_mask = None if layout is None else layout.valid_mask
        if not self.mixer.config.rank_rotary:
            return self.mixer(
                x,
                valid_mask=valid_mask,
                implementation=self.implementation,
            )
        positions = (
            self._vision_position_ids(x.shape[1], x.device)
            if layout is None or layout.position_ids is None
            else layout.position_ids
        )
        return self.mixer(
            x,
            valid_mask=valid_mask,
            position_ids=positions,
            implementation=self.implementation,
        )


def create_lsso_vit(
    *,
    image_size: int,
    patch_size: int,
    num_classes: int,
    embed_dim: int,
    depth: int,
    num_heads: int,
    rank: int,
    mlp_ratio: float,
    core_mode: CoreMode | str,
    rank_rotary: bool,
    bias: bool,
    implementation: _Implementation = "reference",
    drop_path_rate: float = 0.0,
) -> nn.Module:
    """Build timm's ViT with LSSO as the sole token mixer."""

    mode = CoreMode(core_mode)
    resolved_implementation = _validate_implementation(implementation)
    if not isinstance(depth, int) or isinstance(depth, bool) or depth <= 0:
        raise ValueError("depth must be a positive integer")

    from timm.models.vision_transformer import VisionTransformer

    _require_timm_attention_mask_api(VisionTransformer)

    class ConfiguredMixer(_TimmLSSOMixer):
        def __init__(
            self,
            dim: int,
            num_heads: int,
            **kwargs: Any,
        ) -> None:
            super().__init__(
                dim,
                num_heads,
                rank=rank,
                core_mode=mode,
                rank_rotary=rank_rotary,
                implementation=resolved_implementation,
                **kwargs,
            )

    return VisionTransformer(
        img_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        global_pool="token",
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=bias,
        proj_bias=bias,
        drop_path_rate=drop_path_rate,
        attn_layer=ConfiguredMixer,
    )


def _create_deit3_encoder(
    *,
    image_size: int,
    patch_size: int,
    num_classes: int,
    embed_dim: int,
    depth: int,
    num_heads: int,
    rank: int,
    mlp_ratio: float,
    core_mode: CoreMode | str,
    rank_rotary: bool,
    bias: bool,
    implementation: _Implementation,
    drop_path_rate: float,
    layer_scale_init_value: float,
    norm_eps: float,
    dynamic_img_size: bool,
    dynamic_img_pad: bool,
) -> nn.Module:
    """Build the one DeiT III-compatible LSSO encoder.

    The engineering form uses timm blocks, but it preserves DeiT III's
    no-CLS-position layout, LayerScale initialization, and constant stochastic
    depth. The latter differs from timm's default linearly increasing schedule.
    """

    if not isinstance(depth, int) or isinstance(depth, bool) or depth <= 0:
        raise ValueError("depth must be a positive integer")
    if not 0.0 <= drop_path_rate < 1.0:
        raise ValueError("drop_path_rate must be in [0, 1)")
    if layer_scale_init_value <= 0.0:
        raise ValueError("layer_scale_init_value must be positive")
    if norm_eps != 1e-6:
        raise ValueError("DeiT III requires norm_eps = 1e-6")

    from timm.layers import trunc_normal_
    from timm.models.vision_transformer import Block, VisionTransformer

    _require_timm_attention_mask_api(VisionTransformer)

    mode = CoreMode(core_mode)
    resolved_implementation = _validate_implementation(implementation)

    class ConfiguredMixer(_TimmLSSOMixer):
        def __init__(
            self,
            dim: int,
            num_heads: int,
            **kwargs: Any,
        ) -> None:
            super().__init__(
                dim,
                num_heads,
                rank=rank,
                core_mode=mode,
                rank_rotary=rank_rotary,
                implementation=resolved_implementation,
                **kwargs,
            )

    class ConstantDropPathBlock(Block):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["drop_path"] = drop_path_rate
            super().__init__(*args, **kwargs)

    encoder = VisionTransformer(
        img_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        global_pool="token",
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=bias,
        proj_bias=bias,
        no_embed_class=True,
        dynamic_img_size=dynamic_img_size,
        dynamic_img_pad=dynamic_img_pad,
        init_values=layer_scale_init_value,
        drop_path_rate=drop_path_rate,
        norm_layer=partial(nn.LayerNorm, eps=norm_eps),
        block_fn=ConstantDropPathBlock,
        attn_layer=ConfiguredMixer,
    )
    if encoder.cls_token is None:
        raise RuntimeError("DeiT III requires a class token")
    with torch.no_grad():
        trunc_normal_(encoder.cls_token, std=0.02)
    return encoder


class LSSODeiT3(nn.Module):
    """A DeiT III plain ViT with LSSO as its only token mixer."""

    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int = 16,
        num_classes: int = 1000,
        embed_dim: int,
        depth: int,
        num_heads: int,
        rank: int,
        mlp_ratio: float = 4.0,
        core_mode: CoreMode | str = CoreMode.DYNAMIC,
        rank_rotary: bool = True,
        bias: bool = True,
        implementation: _Implementation = "reference",
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-4,
        norm_eps: float = 1e-6,
        no_embed_class: bool = True,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
    ) -> None:
        super().__init__()
        if not no_embed_class:
            raise ValueError("DeiT III requires no_embed_class=True")
        self.encoder = _create_deit3_encoder(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            rank=rank,
            mlp_ratio=mlp_ratio,
            core_mode=core_mode,
            rank_rotary=rank_rotary,
            bias=bias,
            implementation=implementation,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            norm_eps=norm_eps,
            dynamic_img_size=dynamic_img_size,
            dynamic_img_pad=dynamic_img_pad,
        )

    @property
    def blocks(self) -> nn.Module:
        return self.encoder.blocks  # type: ignore[no-any-return]

    @property
    def patch_embed(self) -> nn.Module:
        return self.encoder.patch_embed  # type: ignore[no-any-return]

    def forward(
        self,
        x: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        layout = (
            None
            if valid_mask is None and position_ids is None
            else VisionTokenLayout(valid_mask, position_ids)
        )
        return self.encoder(x, attn_mask=layout)

    def forward_intermediates(
        self,
        x: torch.Tensor,
        *,
        indices: int | Sequence[int] | None = None,
        valid_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        norm: bool = False,
    ) -> list[torch.Tensor]:
        layout = (
            None
            if valid_mask is None and position_ids is None
            else VisionTokenLayout(valid_mask, position_ids)
        )
        result = self.encoder.forward_intermediates(
            x,
            indices=indices,
            intermediates_only=True,
            norm=norm,
            output_fmt="NCHW",
            attn_mask=layout,
        )
        if not isinstance(result, list):
            raise RuntimeError("timm did not return intermediate feature maps")
        return result


def create_lsso_deit3(
    *,
    image_size: int,
    num_classes: int,
    embed_dim: int,
    depth: int,
    num_heads: int,
    rank: int,
    mlp_ratio: float,
    core_mode: CoreMode | str,
    rank_rotary: bool,
    bias: bool,
    implementation: _Implementation = "reference",
    drop_path_rate: float = 0.0,
    layer_scale_init_value: float = 1e-4,
    norm_eps: float = 1e-6,
    no_embed_class: bool = True,
    patch_size: int = 16,
    dynamic_img_size: bool = False,
    dynamic_img_pad: bool = False,
) -> LSSODeiT3:
    """Build a DeiT III LSSO classifier or feature encoder."""

    return LSSODeiT3(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        rank=rank,
        mlp_ratio=mlp_ratio,
        core_mode=core_mode,
        rank_rotary=rank_rotary,
        bias=bias,
        implementation=implementation,
        drop_path_rate=drop_path_rate,
        layer_scale_init_value=layer_scale_init_value,
        norm_eps=norm_eps,
        no_embed_class=no_embed_class,
        dynamic_img_size=dynamic_img_size,
        dynamic_img_pad=dynamic_img_pad,
    )


def create_lsso_deit3_variant(
    variant: str,
    *,
    image_size: int,
    num_classes: int = 1000,
    rank: int | None = None,
    core_mode: CoreMode | str = CoreMode.DYNAMIC,
    rank_rotary: bool = True,
    bias: bool = True,
    implementation: _Implementation = "reference",
    dynamic_img_size: bool = False,
    dynamic_img_pad: bool = False,
) -> LSSODeiT3:
    """Build the agreed Small, Base, or Large LSSO DeiT III scale."""

    spec = deit3_spec(variant)
    return create_lsso_deit3(
        image_size=image_size,
        num_classes=num_classes,
        embed_dim=spec.embed_dim,
        depth=spec.depth,
        num_heads=spec.num_heads,
        rank=deit3_default_rank(variant) if rank is None else rank,
        mlp_ratio=4.0,
        core_mode=core_mode,
        rank_rotary=rank_rotary,
        bias=bias,
        implementation=implementation,
        drop_path_rate=spec.drop_path_rate,
        dynamic_img_size=dynamic_img_size,
        dynamic_img_pad=dynamic_img_pad,
    )


__all__ = [
    "DeiT3Spec",
    "LSSODeiT3",
    "VisionTokenLayout",
    "create_lsso_deit3",
    "create_lsso_deit3_variant",
    "create_lsso_vit",
    "deit3_default_rank",
    "deit3_spec",
]
