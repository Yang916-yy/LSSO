from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lsso import LSSO

from .common import MLP


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale[:, None]) + shift[:, None]


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0).reshape(2, 1, grid_size, grid_size)
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: torch.Tensor) -> torch.Tensor:
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return torch.cat([emb_h, emb_w], dim=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2)))
    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_embedding(t, self.frequency_embedding_size))


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_classes, hidden_size)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.embedding(labels)


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        mixer: str = "lsso",
        rank: int = 16,
        gamma_max: float = 1.2,
        theta_gamma_init: float = 0.5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if mixer == "mha":
            self.mixer = nn.MultiheadAttention(
                hidden_size,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.uses_mha = True
        elif mixer in {"lsso", "lsso-no-global"}:
            self.mixer = LSSO(
                dim=hidden_size,
                num_heads=num_heads,
                rank=rank,
                gamma_max=gamma_max,
                theta_gamma_init=theta_gamma_init,
                no_global=mixer == "lsso-no-global",
            )
            self.uses_mha = False
        else:
            raise ValueError(f"unknown mixer: {mixer}")
        self.mlp = MLP(hidden_size, mlp_ratio=mlp_ratio, dropout=dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        z = modulate(self.norm1(x), shift_msa, scale_msa)
        if self.uses_mha:
            mixed, _ = self.mixer(z, z, z, need_weights=False)
        else:
            mixed = self.mixer(z)
        x = x + gate_msa[:, None] * mixed
        x = x + gate_mlp[:, None] * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class LatentDiT(nn.Module):
    """
    DiT-style latent diffusion backbone with attention replaced by LSSO.

    The VAE remains outside this model. Inputs and outputs are latent tensors,
    typically [B, 4, H, W] for VAE latents.
    """

    def __init__(
        self,
        latent_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        rank: int = 16,
        mlp_ratio: float = 4.0,
        num_classes: int | None = None,
        learn_sigma: bool = False,
        mixer: str = "lsso",
        gamma_max: float = 1.2,
        theta_gamma_init: float = 0.5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if latent_size % patch_size != 0:
            raise ValueError("latent_size must be divisible by patch_size")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.latent_size = latent_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.rank = rank
        self.num_patches_side = latent_size // patch_size
        self.num_patches = self.num_patches_side * self.num_patches_side

        self.x_embedder = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        pos_embed = get_2d_sincos_pos_embed(hidden_size, self.num_patches_side)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0), persistent=False)

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size) if num_classes else None
        blocks = []
        for layer_idx in range(depth):
            block_mixer = mixer
            if mixer == "hybrid":
                block_mixer = "mha" if layer_idx % 2 == 0 else "lsso"
            elif mixer == "hybrid-lsso-first":
                block_mixer = "lsso" if layer_idx % 2 == 0 else "mha"
            elif mixer == "top2-mha":
                block_mixer = "mha" if layer_idx >= depth - 2 else "lsso"
            elif mixer == "bottom2-mha":
                block_mixer = "mha" if layer_idx < 2 else "lsso"
            blocks.append(
                DiTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    mixer=block_mixer,
                    rank=rank,
                    gamma_max=gamma_max,
                    theta_gamma_init=theta_gamma_init,
                    dropout=dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def init_linear(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(init_linear)
        nn.init.xavier_uniform_(self.x_embedder.weight.view(self.x_embedder.weight.shape[0], -1))
        nn.init.zeros_(self.x_embedder.bias)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        if self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.embedding.weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        p = self.patch_size
        c = self.out_channels
        h = w = self.num_patches_side
        x = x.reshape(B, h, w, p, p, c)
        x = torch.einsum("bhwpqc->bchpwq", x)
        return x.reshape(B, c, h * p, w * p)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.x_embedder(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed.to(dtype=x.dtype)

        c = self.t_embedder(timesteps)
        if self.y_embedder is not None:
            if labels is None:
                raise ValueError("labels are required when num_classes is set")
            c = c + self.y_embedder(labels)

        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        return self.unpatchify(x)

    def lsso_layers(self) -> list[LSSO]:
        layers = []
        for block in self.blocks:
            mixer = getattr(block, "mixer", None)
            if isinstance(mixer, LSSO):
                layers.append(mixer)
        return layers
