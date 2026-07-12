from __future__ import annotations

from typing import Any

import torch

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import build_model_with_cfg
from timm.models._registry import generate_default_cfgs, register_model

from .vision_llama import VisionLLaMA, convert_official_vision_llama_state_dict


def _cfg(**kwargs: Any) -> dict[str, Any]:
    return {
        "url": "",
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.875,
        "interpolation": "bicubic",
        "fixed_input_size": False,
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "patch_embed",
        "classifier": "head",
        "license": "apache-2.0",
        **kwargs,
    }


default_cfgs = generate_default_cfgs(
    {
        "vision_llama_small_mha.in1k": _cfg(),
        "vision_llama_small_lsso_r32.in1k": _cfg(),
        "vision_llama_small_rrlsso_r32.in1k": _cfg(),
        "vision_llama_base_mha.in1k": _cfg(),
        "vision_llama_base_lsso_r32.in1k": _cfg(),
        "vision_llama_base_rrlsso_r32.in1k": _cfg(),
        "vision_llama_large_mha.in1k": _cfg(),
        "vision_llama_large_lsso_r32.in1k": _cfg(),
        "vision_llama_large_rrlsso_r32.in1k": _cfg(),
    }
)


def _checkpoint_filter(
    state_dict: dict[str, torch.Tensor], model: VisionLLaMA
) -> dict[str, torch.Tensor]:
    state_dict = state_dict.get("model", state_dict)
    return convert_official_vision_llama_state_dict(model, state_dict)


def _create_vision_llama(
    variant: str,
    *,
    pretrained: bool,
    scale: str,
    mixer: str,
    rank: int = 32,
    **kwargs: Any,
) -> VisionLLaMA:
    if pretrained:
        raise RuntimeError(
            f"{variant} has no bundled pretrained weights; use checkpoint_path "
            "or load_official_vision_llama_checkpoint"
        )
    kwargs.pop("pretrained_cfg", None)
    kwargs.pop("pretrained_cfg_overlay", None)
    kwargs.pop("cache_dir", None)

    configs = {
        "small": dict(dim=384, depth=12, num_heads=6, drop_path_rate=0.05),
        "base": dict(dim=768, depth=12, num_heads=12, drop_path_rate=0.20),
        "large": dict(dim=1024, depth=24, num_heads=16, drop_path_rate=0.45),
    }
    model_kwargs = {**configs[scale], "mixer": mixer, "rank": rank, **kwargs}
    return build_model_with_cfg(
        VisionLLaMA,
        variant,
        pretrained,
        pretrained_filter_fn=_checkpoint_filter,
        **model_kwargs,
    )


@register_model
def vision_llama_small_mha(pretrained: bool = False, **kwargs: Any) -> VisionLLaMA:
    return _create_vision_llama(
        "vision_llama_small_mha", pretrained=pretrained,
        scale="small", mixer="mha", **kwargs,
    )


@register_model
def vision_llama_small_lsso_r32(pretrained: bool = False, **kwargs: Any) -> VisionLLaMA:
    return _create_vision_llama(
        "vision_llama_small_lsso_r32", pretrained=pretrained,
        scale="small", mixer="lsso", rank=kwargs.pop("rank", 32), **kwargs,
    )


@register_model
def vision_llama_small_rrlsso_r32(pretrained: bool = False, **kwargs: Any) -> VisionLLaMA:
    return _create_vision_llama(
        "vision_llama_small_rrlsso_r32", pretrained=pretrained,
        scale="small", mixer="rrlsso", rank=kwargs.pop("rank", 32), **kwargs,
    )


@register_model
def vision_llama_base_mha(pretrained: bool = False, **kwargs: Any) -> VisionLLaMA:
    return _create_vision_llama(
        "vision_llama_base_mha", pretrained=pretrained,
        scale="base", mixer="mha", **kwargs,
    )


@register_model
def vision_llama_base_lsso_r32(pretrained: bool = False, **kwargs: Any) -> VisionLLaMA:
    return _create_vision_llama(
        "vision_llama_base_lsso_r32", pretrained=pretrained,
        scale="base", mixer="lsso", rank=kwargs.pop("rank", 32), **kwargs,
    )


@register_model
def vision_llama_base_rrlsso_r32(pretrained: bool = False, **kwargs: Any) -> VisionLLaMA:
    return _create_vision_llama(
        "vision_llama_base_rrlsso_r32", pretrained=pretrained,
        scale="base", mixer="rrlsso", rank=kwargs.pop("rank", 32), **kwargs,
    )


@register_model
def vision_llama_large_mha(pretrained: bool = False, **kwargs: Any) -> VisionLLaMA:
    return _create_vision_llama(
        "vision_llama_large_mha", pretrained=pretrained,
        scale="large", mixer="mha", **kwargs,
    )


@register_model
def vision_llama_large_lsso_r32(pretrained: bool = False, **kwargs: Any) -> VisionLLaMA:
    return _create_vision_llama(
        "vision_llama_large_lsso_r32", pretrained=pretrained,
        scale="large", mixer="lsso", rank=kwargs.pop("rank", 32), **kwargs,
    )


@register_model
def vision_llama_large_rrlsso_r32(pretrained: bool = False, **kwargs: Any) -> VisionLLaMA:
    return _create_vision_llama(
        "vision_llama_large_rrlsso_r32", pretrained=pretrained,
        scale="large", mixer="rrlsso", rank=kwargs.pop("rank", 32), **kwargs,
    )
