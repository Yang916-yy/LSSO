from __future__ import annotations

import torch
import torch.nn as nn

from .baselines import LinearAttention, OfficialNystromAttention, PerformerAttention
from lsso import LSSO, RoPELSSO

from .common import MLP


class BertStyleBlock(nn.Module):
    """Post-LN encoder block in the style of BERT."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mixer: str = "mha",
        rank: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        gamma_max: float = 0.3,
        theta_gamma_init: float = -4.0,
        normalize_u: bool = True,
    ) -> None:
        super().__init__()
        if mixer == "mha":
            self.mixer = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self._mixer_type = "mha"
        elif mixer == "performer":
            self.mixer = PerformerAttention(
                dim=dim,
                num_heads=num_heads,
                nb_features=rank,
            )
            self._mixer_type = "dense"
        elif mixer == "linear":
            self.mixer = LinearAttention(
                dim=dim,
                num_heads=num_heads,
            )
            self._mixer_type = "dense"
        elif mixer == "nystrom":
            self.mixer = OfficialNystromAttention(
                dim=dim,
                num_heads=num_heads,
                num_landmarks=rank,
                dropout=dropout,
            )
            self._mixer_type = "dense"
        elif mixer in {"lsso", "lsso-no-global", "rope-lsso"}:
            mixer_cls = RoPELSSO if mixer == "rope-lsso" else LSSO
            self.mixer = mixer_cls(
                dim=dim,
                num_heads=num_heads,
                rank=rank,
                dropout=dropout,
                gamma_max=gamma_max,
                theta_gamma_init=theta_gamma_init,
                no_global=mixer == "lsso-no-global",
                normalize_u=normalize_u,
            )
            self._mixer_type = "dense"
        else:
            raise ValueError(f"unknown mixer: {mixer}")

        self.attn_norm = nn.LayerNorm(dim)
        self.ffn = MLP(dim=dim, mlp_ratio=mlp_ratio, dropout=dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._mixer_type == "mha":
            mixed, _ = self.mixer(
                x,
                x,
                x,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
        else:
            valid_mask = None if key_padding_mask is None else ~key_padding_mask
            mixed = self.mixer(x, valid_mask=valid_mask)

        x = self.attn_norm(x + self.dropout(mixed))
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask[:, :, None], 0.0)

        x = self.ffn_norm(x + self.dropout(self.ffn(x)))
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask[:, :, None], 0.0)
        return x


class BertStyleEncoder(nn.Module):
    """
    Small BERT-style encoder classifier without HuggingFace dependencies.

    It keeps BERT's embedding LayerNorm, post-LN encoder blocks, CLS pooler,
    and tanh pooler head, while allowing the token mixer to be MHA or LSSO.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        max_len: int = 128,
        dim: int = 256,
        depth: int = 6,
        num_heads: int = 4,
        mixer: str = "mha",
        rank: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        gamma_max: float = 0.3,
        theta_gamma_init: float = -4.0,
        normalize_u: bool = True,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.token_embed = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.pos_embed = nn.Embedding(max_len, dim)
        self.type_embed = nn.Embedding(2, dim)
        self.embed_norm = nn.LayerNorm(dim)
        self.embed_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                BertStyleBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mixer=mixer,
                    rank=rank,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    gamma_max=gamma_max,
                    theta_gamma_init=theta_gamma_init,
                    normalize_u=normalize_u,
                )
                for _ in range(depth)
            ]
        )
        self.pooler = nn.Linear(dim, dim)
        self.pooler_act = nn.Tanh()
        self.classifier = nn.Linear(dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, N = input_ids.shape
        key_padding_mask = input_ids.eq(self.pad_id)
        pos_ids = torch.arange(N, device=input_ids.device).view(1, N).expand(B, N)
        type_ids = torch.zeros_like(input_ids)

        x = self.token_embed(input_ids) + self.pos_embed(pos_ids) + self.type_embed(type_ids)
        x = self.embed_drop(self.embed_norm(x))
        x = x.masked_fill(key_padding_mask[:, :, None], 0.0)

        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)

        return x

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(input_ids)
        pooled = self.pooler_act(self.pooler(x[:, 0]))
        return self.classifier(pooled)

    def lsso_layers(self) -> list[LSSO]:
        layers = []
        for block in self.blocks:
            mixer = getattr(block, "mixer", None)
            if isinstance(mixer, (LSSO, RoPELSSO)):
                layers.append(mixer)
        return layers
