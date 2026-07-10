"""Archived causal-LM benchmark; not runnable against the supported API."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.models.baselines import _orthogonal_matrix, _softmax_kernel
from lsso import (
    LSSO,
    RoPELSSO,
    SolveStateCache,
    apply_rank_rope,
    read_solve_state,
    update_solve_state,
)


class CausalMHA(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).contiguous().view(B, N, D))


class CausalLinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int, eps: float = 1e-6) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.eps = eps
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = F.elu(q.transpose(1, 2).contiguous()) + 1.0
        k = F.elu(k.transpose(1, 2).contiguous()) + 1.0
        v = v.transpose(1, 2).contiguous()
        k_cum = torch.cumsum(k.float(), dim=2)
        kv = torch.cumsum(k.float().unsqueeze(-1) * v.float().unsqueeze(-2), dim=2)
        denom = (q.float() * k_cum).sum(dim=-1, keepdim=True).clamp_min(self.eps)
        y = (q.float().unsqueeze(-2) @ kv).squeeze(-2) / denom
        return self.out(y.to(x.dtype).transpose(1, 2).contiguous().view(B, N, D))


class CausalPerformerAttention(nn.Module):
    def __init__(self, dim: int, heads: int, features: int = 32, eps: float = 1e-6) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.features = features
        self.eps = eps
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.register_buffer(
            "projection_matrix",
            torch.empty(heads, features, self.head_dim),
            persistent=False,
        )
        self.redraw_projection_matrix()

    @torch.no_grad()
    def redraw_projection_matrix(self) -> None:
        mats = [_orthogonal_matrix(self.features, self.head_dim, self.projection_matrix.device) for _ in range(self.heads)]
        self.projection_matrix.copy_(torch.stack(mats, dim=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        q_prime = _softmax_kernel(q, self.projection_matrix, is_query=True, eps=self.eps)
        k_prime = _softmax_kernel(k, self.projection_matrix, is_query=False, eps=self.eps)
        k_cum = torch.cumsum(k_prime.float(), dim=2)
        kv = torch.cumsum(k_prime.float().unsqueeze(-1) * v.float().unsqueeze(-2), dim=2)
        denom = (q_prime.float() * k_cum).sum(dim=-1, keepdim=True).clamp_min(self.eps)
        y = (q_prime.float().unsqueeze(-2) @ kv).squeeze(-2) / denom
        return self.out(y.to(x.dtype).transpose(1, 2).contiguous().view(B, N, D))


class CausalBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mixer: str, rank: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        if mixer == "mha":
            self.mixer = CausalMHA(dim, heads)
        elif mixer == "linear":
            self.mixer = CausalLinearAttention(dim, heads)
        elif mixer == "performer":
            self.mixer = CausalPerformerAttention(dim, heads, features=rank)
        elif mixer == "lsso":
            self.mixer = LSSO(dim, heads, rank=rank, causal=True, causal_chunk_size=256)
        elif mixer == "rope-lsso":
            self.mixer = RoPELSSO(dim, heads, rank=rank, causal=True, causal_chunk_size=256)
        elif mixer == "lsso-triton":
            self.mixer = LSSO(dim, heads, rank=rank, causal=True, causal_chunk_size=256, causal_backend="triton")
        else:
            raise ValueError(f"unknown mixer: {mixer}")
        hidden = int(dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        dim: int,
        depth: int,
        heads: int,
        mixer: str,
        rank: int,
        position_mode: str = "learned",
    ) -> None:
        super().__init__()
        if position_mode not in {"learned", "none"}:
            raise ValueError(f"unknown position_mode: {position_mode}")
        self.position_mode = position_mode
        self.token_embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, dim)) if position_mode == "learned" else None
        self.blocks = nn.ModuleList([CausalBlock(dim, heads, mixer, rank) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        if self.pos_embed is not None:
            nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embed(input_ids)
        if self.pos_embed is not None:
            x = x + self.pos_embed[:, : input_ids.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    @torch.no_grad()
    def forward_uc_cached(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not all(isinstance(block.mixer, (LSSO, RoPELSSO)) for block in self.blocks):
            raise ValueError("cached decoding is currently implemented for LSSO/RoPE-LSSO blocks")
        caches: list[dict[str, torch.Tensor] | None] = [None for _ in self.blocks]
        logits = []
        for pos in range(input_ids.shape[1]):
            x = self.token_embed(input_ids[:, pos : pos + 1])
            if self.pos_embed is not None:
                x = x + self.pos_embed[:, pos : pos + 1]
            for i, block in enumerate(self.blocks):
                z = block.norm1(x)
                mixed, caches[i] = lsso_step_with_uc_cache(block.mixer, z, caches[i], pos)
                x = x + mixed
                x = x + block.mlp(block.norm2(x))
            logits.append(self.head(self.norm(x)))
        return torch.cat(logits, dim=1)

    @torch.no_grad()
    def forward_sp_cached(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not all(isinstance(block.mixer, (LSSO, RoPELSSO)) for block in self.blocks):
            raise ValueError("cached decoding is currently implemented for LSSO/RoPE-LSSO blocks")
        caches: list[SolveStateCache | None] = [None for _ in self.blocks]
        logits = []
        for pos in range(input_ids.shape[1]):
            x = self.token_embed(input_ids[:, pos : pos + 1])
            if self.pos_embed is not None:
                x = x + self.pos_embed[:, pos : pos + 1]
            for i, block in enumerate(self.blocks):
                z = block.norm1(x)
                mixed, caches[i] = lsso_step_with_sp_cache(block.mixer, z, caches[i], pos)
                x = x + mixed
                x = x + block.mlp(block.norm2(x))
            logits.append(self.head(self.norm(x)))
        return torch.cat(logits, dim=1)


def _spd_solve(G: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    chol = torch.linalg.cholesky_ex(G, check_errors=False).L
    return torch.cholesky_solve(rhs, chol)


def lsso_step_with_uc_cache(
    layer: LSSO | RoPELSSO,
    x: torch.Tensor,
    cache: dict[str, torch.Tensor] | None,
    position: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    B, N, D = x.shape
    if N != 1:
        raise ValueError("cached LSSO step expects a single token")
    H = layer.num_heads
    r = layer.rank
    dh = layer.head_dim
    UC = layer.w_uc(x)
    U, C = UC.split((H * r, D), dim=-1)
    U = U.view(B, 1, H, r).transpose(1, 2).contiguous()
    C = C.view(B, 1, H, dh).transpose(1, 2).contiguous()
    if layer.normalize_u:
        U = U * torch.rsqrt(torch.mean(U * U, dim=-1, keepdim=True) + layer.eps)
    if isinstance(layer, RoPELSSO):
        pos = torch.full((B, 1), position, device=x.device, dtype=torch.long)
        U = apply_rank_rope(U, pos, base=layer.rope_base, scale=layer.rope_scale)

    calc_dtype = torch.float32 if x.dtype != torch.float64 else torch.float64
    if cache is None:
        U_cache = U
        C_cache = C
    else:
        U_cache = torch.cat((cache["U"], U), dim=2)
        C_cache = torch.cat((cache["C"], C), dim=2)

    U_cache_calc = U_cache.to(calc_dtype)
    C_cache_calc = C_cache.to(calc_dtype)
    U_calc = U.to(calc_dtype)
    C_calc = C.to(calc_dtype)
    Ut = U_cache_calc.transpose(-2, -1)
    S = torch.matmul(Ut, U_cache_calc)
    P = torch.matmul(Ut, C_cache_calc)

    mu = F.softplus(layer.theta_mu).to(calc_dtype) + layer.eps
    gamma = layer.gamma_max * torch.sigmoid(layer.theta_gamma).to(calc_dtype)
    if layer.no_global:
        gamma = torch.zeros_like(gamma)
    mu = mu.view(1, H, 1, 1)
    gamma = gamma.view(1, H, 1, 1)
    inv_mu = mu.reciprocal()
    alpha = gamma * inv_mu
    eye = torch.eye(S.shape[-1], device=x.device, dtype=calc_dtype).view(1, 1, S.shape[-1], S.shape[-1])
    G = eye + alpha * S
    K = _spd_solve(G.reshape(B * H, S.shape[-1], S.shape[-1]), P.reshape(B * H, S.shape[-1], dh))
    K = K.view(B, H, S.shape[-1], dh)
    correction = torch.matmul(U_calc.flatten(0, 1), K.flatten(0, 1)).view(B, H, 1, dh)
    Y = inv_mu * C_calc - (alpha * inv_mu) * correction
    Y = Y.to(x.dtype).transpose(1, 2).contiguous().view(B, 1, D)
    return layer.w_o(Y), {"U": U_cache.detach(), "C": C_cache.detach()}


def lsso_step_with_sp_cache(
    layer: LSSO | RoPELSSO,
    x: torch.Tensor,
    cache: SolveStateCache | None,
    position: int,
) -> tuple[torch.Tensor, SolveStateCache]:
    B, N, D = x.shape
    if N != 1:
        raise ValueError("cached LSSO step expects a single token")
    H = layer.num_heads
    r = layer.rank
    dh = layer.head_dim
    UC = layer.w_uc(x)
    U, C = UC.split((H * r, D), dim=-1)
    U = U.view(B, 1, H, r).transpose(1, 2).contiguous()
    C = C.view(B, 1, H, dh).transpose(1, 2).contiguous()
    if layer.normalize_u:
        U = U * torch.rsqrt(torch.mean(U * U, dim=-1, keepdim=True) + layer.eps)
    if isinstance(layer, RoPELSSO):
        pos = torch.full((B, 1), position, device=x.device, dtype=torch.long)
        U = apply_rank_rope(U, pos, base=layer.rope_base, scale=layer.rope_scale)

    cache = update_solve_state(cache, U, C)

    mu = F.softplus(layer.theta_mu) + layer.eps
    gamma = layer.gamma_max * torch.sigmoid(layer.theta_gamma)
    if layer.no_global:
        gamma = torch.zeros_like(gamma)
    Y = read_solve_state(U, C, cache, mu, gamma)
    Y = Y.to(x.dtype).transpose(1, 2).contiguous().view(B, 1, D)
    return layer.w_o(Y), SolveStateCache(S=cache.S.detach(), P=cache.P.detach(), length=cache.length)


@dataclass
class BatchSampler:
    tokens: torch.Tensor
    seq_len: int
    batch_size: int
    device: torch.device

    def sample(self) -> torch.Tensor:
        idx = torch.randint(0, self.tokens.shape[0], (self.batch_size,))
        x = self.tokens[idx, : self.seq_len].to(self.device, non_blocking=True)
        return x


def load_tokens(path: Path, seq_len: int, limit: int | None = None) -> tuple[torch.Tensor, int]:
    obj = torch.load(path, map_location="cpu")
    tokens = obj["input_ids"].long()
    if limit is not None:
        tokens = tokens[:limit]
    tokens = tokens[:, :seq_len].contiguous()
    vocab_size = int(tokens.max().item()) + 1
    return tokens, vocab_size


@torch.no_grad()
def evaluate(model: nn.Module, tokens: torch.Tensor, *, seq_len: int, batch_size: int, steps: int, device: torch.device) -> tuple[float, float]:
    model.eval()
    losses = []
    accs = []
    sampler = BatchSampler(tokens, seq_len, batch_size, device)
    for _ in range(steps):
        batch = sampler.sample()
        logits = model(batch[:, :-1])
        target = batch[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=0)
        valid = target.ne(0)
        pred = logits.argmax(dim=-1)
        acc = (pred.eq(target) & valid).sum().float() / valid.sum().clamp_min(1)
        losses.append(loss.item())
        accs.append(acc.item())
    return sum(losses) / len(losses), sum(accs) / len(accs)


@torch.no_grad()
def evaluate_cached(
    model: TinyGPT,
    tokens: torch.Tensor,
    *,
    seq_len: int,
    batch_size: int,
    steps: int,
    device: torch.device,
    cache_mode: str = "uc",
) -> tuple[float, float, float]:
    model.eval()
    if cache_mode not in {"uc", "sp"}:
        raise ValueError(f"unknown cache_mode: {cache_mode}")
    losses = []
    accs = []
    sampler = BatchSampler(tokens, seq_len, batch_size, device)
    token_count = 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        batch = sampler.sample()
        logits = cached_logits(model, batch[:, :-1], cache_mode)
        target = batch[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=0)
        valid = target.ne(0)
        pred = logits.argmax(dim=-1)
        acc = (pred.eq(target) & valid).sum().float() / valid.sum().clamp_min(1)
        losses.append(loss.item())
        accs.append(acc.item())
        token_count += int(batch[:, :-1].numel())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return sum(losses) / len(losses), sum(accs) / len(accs), token_count / max(elapsed, 1e-9)


@torch.no_grad()
def cached_logits(model: TinyGPT, input_ids: torch.Tensor, cache_mode: str) -> torch.Tensor:
    if cache_mode == "uc":
        return model.forward_uc_cached(input_ids)
    if cache_mode == "sp":
        return model.forward_sp_cached(input_ids)
    raise ValueError(f"unknown cache_mode: {cache_mode}")


@torch.no_grad()
def evaluate_cache_pair(
    model: TinyGPT,
    tokens: torch.Tensor,
    *,
    seq_len: int,
    batch_size: int,
    steps: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    sampler = BatchSampler(tokens, seq_len, batch_size, device)
    batches = [sampler.sample() for _ in range(steps)]

    def run_mode(cache_mode: str) -> tuple[float, float, float, list[torch.Tensor]]:
        losses = []
        accs = []
        logits_list = []
        token_count = 0
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for batch in batches:
            logits = cached_logits(model, batch[:, :-1], cache_mode)
            target = batch[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=0)
            valid = target.ne(0)
            pred = logits.argmax(dim=-1)
            acc = (pred.eq(target) & valid).sum().float() / valid.sum().clamp_min(1)
            losses.append(loss.item())
            accs.append(acc.item())
            logits_list.append(logits.detach())
            token_count += int(batch[:, :-1].numel())
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        return sum(losses) / len(losses), sum(accs) / len(accs), token_count / max(elapsed, 1e-9), logits_list

    uc_loss, uc_acc, uc_tps, uc_logits = run_mode("uc")
    sp_loss, sp_acc, sp_tps, sp_logits = run_mode("sp")
    uc_sp_max_diff = max((a - b).abs().max().item() for a, b in zip(uc_logits, sp_logits))
    return {
        "uc_cached_loss": uc_loss,
        "uc_cached_acc": uc_acc,
        "uc_cached_tokens_per_sec": uc_tps,
        "sp_cached_loss": sp_loss,
        "sp_cached_acc": sp_acc,
        "sp_cached_tokens_per_sec": sp_tps,
        "uc_sp_cache_max_diff": uc_sp_max_diff,
    }


@torch.no_grad()
def cached_forward_max_diff(
    model: TinyGPT,
    tokens: torch.Tensor,
    *,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    cache_mode: str = "uc",
) -> float:
    model.eval()
    batch = tokens[:batch_size, :seq_len].to(device, non_blocking=True)
    full = model(batch[:, :-1])
    cached = cached_logits(model, batch[:, :-1], cache_mode)
    return (full - cached).abs().max().item()


def run_one(args: argparse.Namespace, model_name: str, rank: int, train_tokens: torch.Tensor, val_tokens: torch.Tensor, vocab_size: int) -> dict[str, float | int | str]:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = TinyGPT(
        vocab_size=vocab_size,
        seq_len=args.seq_len - 1,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mixer=model_name,
        rank=rank,
        position_mode=args.position_mode,
    ).to(device)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler_enabled = args.amp and device.type == "cuda"
    train_sampler = BatchSampler(train_tokens, args.seq_len, args.batch_size, device)
    start = time.perf_counter()
    last_loss = float("nan")
    last_acc = float("nan")
    for step in range(1, args.steps + 1):
        model.train()
        batch = train_sampler.sample()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=scaler_enabled):
            logits = model(batch[:, :-1])
            target = batch[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), target.reshape(-1), ignore_index=0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            valid = target.ne(0)
            pred = logits.detach().argmax(dim=-1)
            acc = (pred.eq(target) & valid).sum().float() / valid.sum().clamp_min(1)
            last_loss = float(loss.item())
            last_acc = float(acc.item())
            print(
                {
                    "model": model_name,
                    "rank": rank,
                    "step": step,
                    "train_loss": last_loss,
                    "train_acc": last_acc,
                },
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    val_loss, val_acc = evaluate(
        model,
        val_tokens,
        seq_len=args.seq_len,
        batch_size=args.eval_batch_size,
        steps=args.eval_steps,
        device=device,
    )
    if model_name in {"lsso", "rope-lsso"}:
        cache_metrics = evaluate_cache_pair(
            model,
            val_tokens,
            seq_len=args.seq_len,
            batch_size=args.cache_eval_batch_size,
            steps=args.cache_eval_steps,
            device=device,
        )
        uc_cache_max_diff = cached_forward_max_diff(
            model,
            val_tokens,
            seq_len=args.seq_len,
            batch_size=min(args.cache_eval_batch_size, 4),
            device=device,
            cache_mode="uc",
        )
        sp_cache_max_diff = cached_forward_max_diff(
            model,
            val_tokens,
            seq_len=args.seq_len,
            batch_size=min(args.cache_eval_batch_size, 4),
            device=device,
            cache_mode="sp",
        )
    else:
        cache_metrics = {
            "uc_cached_loss": float("nan"),
            "uc_cached_acc": float("nan"),
            "uc_cached_tokens_per_sec": float("nan"),
            "sp_cached_loss": float("nan"),
            "sp_cached_acc": float("nan"),
            "sp_cached_tokens_per_sec": float("nan"),
            "uc_sp_cache_max_diff": float("nan"),
        }
        uc_cache_max_diff = float("nan")
        sp_cache_max_diff = float("nan")
    return {
        "model": model_name,
        "rank": rank,
        "steps": args.steps,
        "seq_len": args.seq_len,
        "dim": args.dim,
        "depth": args.depth,
        "heads": args.heads,
        "position_mode": args.position_mode,
        "params_m": params_m,
        "train_loss_last": last_loss,
        "train_acc_last": last_acc,
        "val_loss": val_loss,
        "val_ppl": math.exp(min(val_loss, 20.0)),
        "val_acc": val_acc,
        "uc_cached_loss": cache_metrics["uc_cached_loss"],
        "uc_cached_ppl": math.exp(min(cache_metrics["uc_cached_loss"], 20.0))
        if math.isfinite(cache_metrics["uc_cached_loss"])
        else float("nan"),
        "uc_cached_acc": cache_metrics["uc_cached_acc"],
        "uc_cached_tokens_per_sec": cache_metrics["uc_cached_tokens_per_sec"],
        "sp_cached_loss": cache_metrics["sp_cached_loss"],
        "sp_cached_ppl": math.exp(min(cache_metrics["sp_cached_loss"], 20.0))
        if math.isfinite(cache_metrics["sp_cached_loss"])
        else float("nan"),
        "sp_cached_acc": cache_metrics["sp_cached_acc"],
        "sp_cached_tokens_per_sec": cache_metrics["sp_cached_tokens_per_sec"],
        "uc_cache_max_diff": uc_cache_max_diff,
        "sp_cache_max_diff": sp_cache_max_diff,
        "uc_sp_cache_max_diff": cache_metrics["uc_sp_cache_max_diff"],
        "steps_per_sec": args.steps / elapsed,
    }


def parse_model_spec(text: str) -> tuple[str, int]:
    if ":" not in text:
        return text, 32
    name, rank = text.split(":", 1)
    return name, int(rank)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare causal mixers on local IMDB next-token modeling.")
    parser.add_argument("--train", default="data/imdb/encoded_max40000_min2_len512_train.pt")
    parser.add_argument("--val", default="data/imdb/encoded_max40000_min2_len512_test.pt")
    parser.add_argument("--models", default="mha:32,linear:32,performer:32,lsso:16,lsso:32")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--position-mode", choices=("learned", "none"), default="learned")
    parser.add_argument("--train-limit", type=int, default=12000)
    parser.add_argument("--val-limit", type=int, default=3000)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-steps", type=int, default=40)
    parser.add_argument("--cache-eval-batch-size", type=int, default=4)
    parser.add_argument("--cache-eval-steps", type=int, default=8)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", default="runs/imdb_causal_lm_compare.tsv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_tokens, train_vocab = load_tokens(Path(args.train), args.seq_len, args.train_limit)
    val_tokens, val_vocab = load_tokens(Path(args.val), args.seq_len, args.val_limit)
    vocab_size = max(train_vocab, val_vocab)
    rows = []
    for spec in args.models.split(","):
        model_name, rank = parse_model_spec(spec.strip())
        row = run_one(args, model_name, rank, train_tokens, val_tokens, vocab_size)
        print(row, flush=True)
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
