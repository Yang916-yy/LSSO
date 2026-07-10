from __future__ import annotations

import torch
import torch.nn as nn

from lsso import LSSO

from .common import EncoderBlock


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        max_len: int = 128,
        dim: int = 128,
        depth: int = 3,
        num_heads: int = 4,
        mixer: str = "mha",
        rank: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        gamma_max: float = 1.2,
        theta_gamma_init: float = 0.5,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.token_embed = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, dim))
        self.drop = nn.Dropout(dropout)
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
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        key_padding_mask = input_ids.eq(self.pad_id)
        x = self.token_embed(input_ids)
        x = self.drop(x + self.pos_embed[:, : input_ids.shape[1]])
        x = x.masked_fill(key_padding_mask[:, :, None], 0.0)

        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)

        x = self.norm(x)
        return self.head(x[:, 0])

    def lsso_layers(self) -> list[LSSO]:
        layers = []
        for block in self.blocks:
            mixer = getattr(block, "mixer", None)
            if isinstance(mixer, LSSO):
                layers.append(mixer)
        return layers
