from __future__ import annotations

import torch
import torch.nn as nn

from lsso import LSSO

from .common import EncoderBlock


class VisionEncoder(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 10,
        dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mixer: str = "mha",
        rank: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        gamma_max: float = 0.1,
        theta_gamma_init: float = -6.0,
        normalize_u: bool = True,
        use_custom_backward: bool = True,
        use_triton_backward: bool = False,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.patch_embed = nn.Conv2d(
            in_chans,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        num_patches = (image_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mixer=mixer,
                    rank=rank,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    gamma_max=gamma_max,
                    theta_gamma_init=theta_gamma_init,
                    normalize_u=normalize_u,
                    use_custom_backward=use_custom_backward,
                    use_triton_backward=use_triton_backward,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return self.head(x[:, 0])

    def lsso_layers(self) -> list[LSSO]:
        layers = []
        for block in self.blocks:
            mixer = getattr(block, "mixer", None)
            if isinstance(mixer, LSSO):
                layers.append(mixer)
        return layers
