from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from lsso import MixerAdapter


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, multiple_of: int = 256, bias: bool = False) -> None:
        super().__init__()
        hidden = int(2 * (4 * dim) / 3)
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden, bias=bias)
        self.w3 = nn.Linear(dim, hidden, bias=bias)
        self.w2 = nn.Linear(hidden, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, device=x.device, dtype=x.dtype).bernoulli_(keep)
        return x * mask / keep


class VisionLLaMABlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mixer: str,
        rank: int,
        drop_path: float,
        layer_scale_init: float,
        **mixer_kwargs,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.mixer = MixerAdapter(
            dim, num_heads, mixer=mixer, rank=rank, bias=False, **mixer_kwargs
        )
        self.mlp = SwiGLU(dim)
        self.drop_path = DropPath(drop_path)
        self.gamma_1 = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.gamma_2 = nn.Parameter(layer_scale_init * torch.ones(dim))

    def forward(
        self,
        x: torch.Tensor,
        *,
        spatial_shape: tuple[int, int],
        valid_mask: torch.Tensor | None = None,
        num_prefix_tokens: int = 1,
    ) -> torch.Tensor:
        x = x + self.drop_path(
            self.gamma_1 * self.mixer(
                self.norm1(x), valid_mask=valid_mask,
                spatial_shape=spatial_shape, num_prefix_tokens=num_prefix_tokens,
            )
        )
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        if valid_mask is not None:
            x = x * valid_mask[:, :, None].to(x.dtype)
        return x


@dataclass(frozen=True)
class VisionLLaMAConfig:
    dim: int
    depth: int
    num_heads: int
    drop_path_rate: float


VISION_LLAMA_CONFIGS = {
    "small": VisionLLaMAConfig(384, 12, 6, 0.05),
    "base": VisionLLaMAConfig(768, 12, 12, 0.20),
    "large": VisionLLaMAConfig(1024, 24, 16, 0.45),
}


class VisionLLaMA(nn.Module):
    """Optimized plain VisionLLaMA backbone with interchangeable token mixer."""

    def __init__(
        self,
        *,
        image_size: int | tuple[int, int] = 224,
        img_size: int | tuple[int, int] | None = None,
        patch_size: int = 16,
        in_channels: int = 3,
        in_chans: int | None = None,
        num_classes: int = 1000,
        dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mixer: str = "mha",
        rank: int = 32,
        drop_path_rate: float = 0.05,
        layer_scale_init: float = 1e-4,
        learned_position: bool = True,
        global_pool: str = "token",
        drop_rate: float = 0.0,
        **mixer_kwargs,
    ) -> None:
        super().__init__()
        if img_size is not None:
            image_size = img_size
        if in_chans is not None:
            in_channels = in_chans
        if global_pool not in ("", "token"):
            raise ValueError("VisionLLaMA supports token pooling only")
        if drop_rate:
            raise ValueError("non-zero drop_rate is not implemented for VisionLLaMA")
        image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.num_features = dim
        self.num_classes = num_classes
        self.mixer_name = mixer
        self.patch_embed = nn.Conv2d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size
        )
        grid = (image_size[0] // patch_size, image_size[1] // patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = (
            nn.Parameter(torch.zeros(1, 1 + grid[0] * grid[1], dim))
            if learned_position else None
        )
        rates = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            VisionLLaMABlock(
                dim, num_heads, mixer, rank, rates[i], layer_scale_init,
                **mixer_kwargs,
            )
            for i in range(depth)
        )
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        def init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(init)
        # Child solve mixers fold the historical initial gain into W_O when
        # g is fixed to one. ``self.apply(init)`` reinitializes those weights,
        # so repeat the fold once on the final initialized matrices.
        for module in self.modules():
            fold = getattr(module, "fold_fixed_gain_into_output", None)
            if fold is not None:
                fold(force=True)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

    def _position_embedding(self, spatial_shape: tuple[int, int]) -> torch.Tensor | None:
        if self.pos_embed is None:
            return None
        source_h = self.image_size[0] // self.patch_size
        source_w = self.image_size[1] // self.patch_size
        if spatial_shape == (source_h, source_w):
            return self.pos_embed
        prefix, patch = self.pos_embed[:, :1], self.pos_embed[:, 1:]
        patch = patch.reshape(1, source_h, source_w, self.dim).permute(0, 3, 1, 2)
        patch = F.interpolate(patch.float(), size=spatial_shape, mode="bicubic", align_corners=False)
        patch = patch.to(self.pos_embed.dtype).flatten(2).transpose(1, 2)
        return torch.cat((prefix, patch), dim=1)

    def forward_features(self, image: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(image)
        spatial_shape = (x.shape[-2], x.shape[-1])
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        position = self._position_embedding(spatial_shape)
        if position is not None:
            x = x + position.to(device=x.device, dtype=x.dtype)
        for block in self.blocks:
            x = block(x, spatial_shape=spatial_shape)
        return self.norm(x)[:, 0]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(image))

    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: str | None = None) -> None:
        if global_pool not in (None, "", "token"):
            raise ValueError("VisionLLaMA supports token pooling only")
        self.num_classes = int(num_classes)
        self.head = nn.Linear(self.dim, num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        names = {"cls_token"}
        if self.pos_embed is not None:
            names.add("pos_embed")
        return names


def create_vision_llama(
    scale: str = "small", *, mixer: str = "mha", rank: int = 32, **kwargs
) -> VisionLLaMA:
    try:
        config = VISION_LLAMA_CONFIGS[scale]
    except KeyError as exc:
        raise ValueError(f"unknown scale {scale!r}; choose {tuple(VISION_LLAMA_CONFIGS)}") from exc
    return VisionLLaMA(
        dim=config.dim, depth=config.depth, num_heads=config.num_heads,
        drop_path_rate=config.drop_path_rate, mixer=mixer, rank=rank, **kwargs,
    )


def load_official_vision_llama_checkpoint(
    model: VisionLLaMA,
    checkpoint: str | Path | dict,
) -> tuple[list[str], list[str]]:
    """Load shared official weights; MHA additionally imports QKV/projection."""
    if not isinstance(checkpoint, dict):
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint)
    converted = convert_official_vision_llama_state_dict(model, state)
    incompatible = model.load_state_dict(converted, strict=False)
    return incompatible.missing_keys, incompatible.unexpected_keys


def convert_official_vision_llama_state_dict(
    model: VisionLLaMA,
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert an upstream key layout without mutating or loading the model."""
    converted: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        key = key.removeprefix("module.")
        key = key.replace("patch_embed.proj.", "patch_embed.")
        key = key.replace(".mlp.c_fc1.", ".mlp.w1.")
        key = key.replace(".mlp.c_fc2.", ".mlp.w3.")
        key = key.replace(".mlp.c_proj.", ".mlp.w2.")
        if ".attn.qkv." in key:
            if model.mixer_name != "mha":
                continue
            key = key.replace(".attn.qkv.", ".mixer.mixer.qkv.")
        elif ".attn.proj." in key:
            if model.mixer_name != "mha":
                continue
            key = key.replace(".attn.proj.", ".mixer.mixer.proj.")
        converted[key] = value
    return converted
