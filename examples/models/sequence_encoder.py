from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lsso import MixerAdapter


class SequenceMixerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        mixer: str,
        rank: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = MixerAdapter(
            dim,
            num_heads,
            mixer,
            rank=rank,
            dropout=dropout,
            rotary_2d=False,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x), valid_mask=valid_mask)
        x = x + self.mlp(self.norm2(x))
        return x * valid_mask.unsqueeze(-1).to(x.dtype)


class SequenceMixerEncoder(nn.Module):
    """Small non-BERT bidirectional encoder shared by auxiliary tasks.

    Learned absolute embeddings supply task-level position capacity. RRLSSO
    independently applies its fixed 1-D rank rotary transform inside the solve.
    """

    def __init__(
        self,
        vocab_size: int,
        *,
        max_length: int,
        pad_token_id: int,
        dim: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        mixer: str = "rrlsso",
        rank: int = 32,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        projection_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.max_length = int(max_length)
        self.pad_token_id = int(pad_token_id)
        self.token_embedding = nn.Embedding(vocab_size, dim, padding_idx=pad_token_id)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_length, dim))
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            SequenceMixerBlock(
                dim,
                num_heads,
                mixer=mixer,
                rank=rank,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(dim)
        output_dim = projection_dim or dim
        self.projection = nn.Linear(dim, output_dim, bias=False)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, length]")
        if input_ids.shape[1] > self.max_length:
            raise ValueError(
                f"sequence length {input_ids.shape[1]} exceeds max_length={self.max_length}"
            )
        valid = input_ids.ne(self.pad_token_id) if attention_mask is None else attention_mask.bool()
        x = self.token_embedding(input_ids)
        x = self.embedding_dropout(x + self.position_embedding[:, : input_ids.shape[1]])
        x = x * valid.unsqueeze(-1).to(x.dtype)
        for block in self.blocks:
            x = block(x, valid)
        x = self.norm(x)
        weights = valid.unsqueeze(-1).to(x.dtype)
        pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.projection(pooled)

    def encode_normalized(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return F.normalize(self(input_ids, attention_mask), dim=-1)


class ProteinFitnessModel(nn.Module):
    def __init__(self, encoder: SequenceMixerEncoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.projection.out_features, 1)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.head(self.encoder(input_ids, attention_mask)).squeeze(-1)
