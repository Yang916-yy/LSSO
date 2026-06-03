from __future__ import annotations

import math

import torch
import torch.nn as nn


def _softmax_kernel(
    x: torch.Tensor,
    projection: torch.Tensor,
    *,
    is_query: bool,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Positive FAVOR+ random features for non-causal softmax attention."""
    data_normalizer = x.shape[-1] ** -0.25
    ratio = projection.shape[-2] ** -0.5
    x = data_normalizer * x
    proj = torch.einsum("bhnd,hmd->bhnm", x.float(), projection.float())
    diag = (x.float() * x.float()).sum(dim=-1, keepdim=True) * 0.5
    if is_query:
        proj = proj - diag - proj.amax(dim=-1, keepdim=True).detach()
    else:
        proj = proj - diag - proj.amax(dim=(-2, -1), keepdim=True).detach()
    return ratio * (torch.exp(proj.clamp(max=80.0)) + eps)


def _orthogonal_matrix(rows: int, cols: int, device: torch.device) -> torch.Tensor:
    blocks = []
    full_blocks = rows // cols
    for _ in range(full_blocks):
        q, _ = torch.linalg.qr(torch.randn(cols, cols, device=device), mode="reduced")
        blocks.append(q.t())
    remaining = rows - full_blocks * cols
    if remaining > 0:
        q, _ = torch.linalg.qr(torch.randn(cols, cols, device=device), mode="reduced")
        blocks.append(q.t()[:remaining])
    mat = torch.cat(blocks, dim=0)
    multiplier = torch.randn(rows, cols, device=device).norm(dim=1)
    return torch.diag(multiplier) @ mat


class PerformerAttention(nn.Module):
    """
    Non-causal Performer/FAVOR+ token mixer.

    This keeps the same QKV/out projection surface as MHA and uses fixed
    Gaussian orthogonal random features, matching the official FAVOR+ softmax
    kernel construction at the algorithm level.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        nb_features: int | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.nb_features = nb_features or int(self.head_dim * math.log(self.head_dim + 1))
        self.eps = eps
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.register_buffer(
            "projection_matrix",
            torch.empty(num_heads, self.nb_features, self.head_dim),
            persistent=False,
        )
        self.redraw_projection_matrix()

    @torch.no_grad()
    def redraw_projection_matrix(self) -> None:
        mats = [
            _orthogonal_matrix(self.nb_features, self.head_dim, self.projection_matrix.device)
            for _ in range(self.num_heads)
        ]
        self.projection_matrix.copy_(torch.stack(mats, dim=0))

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        if valid_mask is not None:
            mask = valid_mask[:, None, :, None].to(dtype=x.dtype)
            k = k * mask
            v = v * mask

        q_prime = _softmax_kernel(q, self.projection_matrix, is_query=True, eps=self.eps)
        k_prime = _softmax_kernel(k, self.projection_matrix, is_query=False, eps=self.eps)
        if valid_mask is not None:
            k_prime = k_prime * mask

        k_sum = k_prime.sum(dim=-2)
        denom = torch.einsum("bhnm,bhm->bhn", q_prime, k_sum).clamp_min(self.eps)
        kv = torch.einsum("bhnm,bhnd->bhmd", k_prime, v.float())
        y = torch.einsum("bhnm,bhmd,bhn->bhnd", q_prime, kv, denom.reciprocal())
        y = y.to(dtype=x.dtype).transpose(1, 2).contiguous().view(B, N, D)
        y = self.out_proj(y)
        if valid_mask is not None:
            y = y * valid_mask[:, :, None].to(dtype=y.dtype)
        return y


def _masked_segment_mean(
    x: torch.Tensor,
    valid_mask: torch.Tensor | None,
    num_landmarks: int,
) -> torch.Tensor:
    B, H, N, D = x.shape
    m = min(num_landmarks, N)
    seg_len = math.ceil(N / m)
    padded_len = seg_len * m
    pad = padded_len - N
    if pad:
        x = torch.nn.functional.pad(x, (0, 0, 0, pad))
    x = x.view(B, H, m, seg_len, D)

    if valid_mask is None:
        return x.mean(dim=-2)

    mask = valid_mask[:, None, :, None].to(dtype=x.dtype, device=x.device)
    if pad:
        mask = torch.nn.functional.pad(mask, (0, 0, 0, pad))
    mask = mask.view(B, 1, m, seg_len, 1)
    denom = mask.sum(dim=-2).clamp_min(1.0)
    return (x * mask).sum(dim=-2) / denom


def _iterative_pinv(x: torch.Tensor, num_iters: int = 6) -> torch.Tensor:
    I = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
    abs_x = x.abs()
    z = x.transpose(-1, -2) / (
        abs_x.sum(dim=-2, keepdim=True).amax(dim=-1, keepdim=True).clamp_min(1e-6)
        * abs_x.sum(dim=-1, keepdim=True).amax(dim=-2, keepdim=True).clamp_min(1e-6)
    )
    for _ in range(num_iters):
        xz = x @ z
        z = 0.25 * z @ (
            13 * I
            - xz
            @ (
                15 * I
                - xz
                @ (
                    7 * I
                    - xz
                )
            )
        )
    return z


class NystromAttention(nn.Module):
    """
    Nyströmformer token mixer.

    This follows the official Nyströmformer approximation pattern:
        softmax(Q K_landmark^T) @ pinv(softmax(Q_landmark K_landmark^T))
        @ softmax(Q_landmark K^T) @ V
    with segment-mean landmarks and a depthwise convolution residual.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_landmarks: int = 64,
        pinv_iters: int = 6,
        conv_kernel_size: int = 65,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_landmarks = num_landmarks
        self.pinv_iters = pinv_iters
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        padding = conv_kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=conv_kernel_size,
            padding=padding,
            groups=dim,
            bias=False,
        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2).contiguous() * (self.head_dim ** -0.5)
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        if valid_mask is not None:
            head_mask = valid_mask[:, None, :, None].to(dtype=x.dtype)
            q = q * head_mask
            k = k * head_mask
            v = v * head_mask

        q_landmarks = _masked_segment_mean(q, valid_mask, self.num_landmarks)
        k_landmarks = _masked_segment_mean(k, valid_mask, self.num_landmarks)

        sim1 = torch.matmul(q.float(), k_landmarks.float().transpose(-2, -1)).softmax(dim=-1)
        sim2 = torch.matmul(
            q_landmarks.float(),
            k_landmarks.float().transpose(-2, -1),
        ).softmax(dim=-1)
        sim3_scores = torch.matmul(q_landmarks.float(), k.float().transpose(-2, -1))
        if valid_mask is not None:
            sim3_scores = sim3_scores.masked_fill(~valid_mask[:, None, None, :], -1e4)
        sim3 = sim3_scores.softmax(dim=-1)

        z = torch.matmul(sim3, v.float())
        z = torch.matmul(_iterative_pinv(sim2, self.pinv_iters), z)
        y = torch.matmul(sim1, z).to(dtype=x.dtype)

        conv_res = self.conv(v.transpose(1, 2).contiguous().view(B, N, D).transpose(1, 2))
        conv_res = conv_res[..., :N].transpose(1, 2).contiguous().view(B, N, self.num_heads, self.head_dim)
        conv_res = conv_res.transpose(1, 2).contiguous()
        y = y + conv_res

        y = y.transpose(1, 2).contiguous().view(B, N, D)
        y = self.out_proj(y)
        if valid_mask is not None:
            y = y * valid_mask[:, :, None].to(dtype=y.dtype)
        return y


class OfficialNystromAttention(nn.Module):
    """Drop-in mixer using Transformers' official randomly initialized Nyströmformer attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_landmarks: int = 64,
        conv_kernel_size: int = 65,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        try:
            from transformers import NystromformerConfig
            from transformers.models.nystromformer.modeling_nystromformer import NystromformerSelfAttention
        except ImportError as exc:
            raise ImportError(
                "OfficialNystromAttention requires Hugging Face Transformers. "
                "Install it with: pip install transformers"
            ) from exc

        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.max_landmarks = num_landmarks
        self.config = NystromformerConfig(
            hidden_size=dim,
            num_attention_heads=num_heads,
            num_landmarks=num_landmarks,
            segment_means_seq_len=num_landmarks,
            conv_kernel_size=conv_kernel_size,
            attention_probs_dropout_prob=dropout,
            hidden_dropout_prob=dropout,
        )
        self.attn = NystromformerSelfAttention(self.config)
        self.out_proj = nn.Linear(dim, dim)

    @staticmethod
    def _pad_to_landmarks(
        x: torch.Tensor,
        valid_mask: torch.Tensor | None,
        num_landmarks: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        seq_len = x.shape[1]
        pad = (num_landmarks - seq_len % num_landmarks) % num_landmarks
        if pad == 0:
            return x, valid_mask, seq_len

        x = torch.nn.functional.pad(x, (0, 0, 0, pad))
        if valid_mask is None:
            valid_mask = torch.ones(seq_len, device=x.device, dtype=torch.bool).expand(x.shape[0], seq_len)
        valid_mask = torch.nn.functional.pad(valid_mask, (0, pad), value=False)
        return x, valid_mask, seq_len

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        num_landmarks = min(x.shape[1], self.max_landmarks)
        x, valid_mask, original_len = self._pad_to_landmarks(x, valid_mask, num_landmarks)
        seq_len = x.shape[1]
        self.attn.seq_len = seq_len
        self.attn.num_landmarks = num_landmarks

        attention_mask = None
        if valid_mask is not None:
            attention_mask = (~valid_mask)[:, None, None, :].to(dtype=x.dtype) * torch.finfo(x.dtype).min

        y = self.attn(x, attention_mask=attention_mask, output_attentions=False)[0]
        y = self.out_proj(y)
        y = y[:, :original_len]
        if valid_mask is not None:
            valid_mask = valid_mask[:, :original_len]
            y = y * valid_mask[:, :, None].to(dtype=y.dtype)
        return y
