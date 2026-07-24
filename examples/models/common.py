from __future__ import annotations

import torch
import torch.nn as nn

from lsso import LSSO, RRLSSO
from .baselines import LinearAttention, OfficialNystromAttention, PerformerAttention


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mixer: str = "mha",
        rank: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        gain_init: float = 1.0,
        normalize_u: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        if mixer == "mha":
            self.mixer = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self._uses_mha = True
        elif mixer == "performer":
            self.mixer = PerformerAttention(
                dim=dim,
                num_heads=num_heads,
                nb_features=rank,
            )
            self._uses_mha = False
        elif mixer == "linear":
            self.mixer = LinearAttention(
                dim=dim,
                num_heads=num_heads,
            )
            self._uses_mha = False
        elif mixer == "nystrom":
            self.mixer = OfficialNystromAttention(
                dim=dim,
                num_heads=num_heads,
                num_landmarks=rank,
                dropout=dropout,
            )
            self._uses_mha = False
        elif mixer in {"lsso", "lsso-no-global", "rrlsso"}:
            mixer_cls = RRLSSO if mixer == "rrlsso" else LSSO
            self.mixer = mixer_cls(
                dim=dim,
                num_heads=num_heads,
                rank=rank,
                dropout=dropout,
                gain_init=gain_init,
                no_global=mixer == "lsso-no-global",
                normalize_u=normalize_u,
            )
            self._uses_mha = False
        else:
            raise ValueError(f"unknown mixer: {mixer}")

        self.mlp = MLP(dim=dim, mlp_ratio=mlp_ratio, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z = self.norm1(x)
        if self._uses_mha:
            mixed, _ = self.mixer(
                z,
                z,
                z,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
        else:
            valid_mask = None if key_padding_mask is None else ~key_padding_mask
            mixed = self.mixer(z, valid_mask=valid_mask)

        x = x + self.dropout(mixed)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask[:, :, None], 0.0)
        return x
