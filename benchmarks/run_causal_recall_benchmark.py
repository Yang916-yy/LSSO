from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.models.baselines import _orthogonal_matrix, _softmax_kernel
from lsso import LSSO, RoPELSSO


PAD_ID = 0
BOS_ID = 1
FILL_ID = 2
SET_ID = 3
QUERY_ID = 4
ANSWER_ID = 5
CAND_ID = 6
KEY_OFFSET = 16


@dataclass(frozen=True)
class ModelSpec:
    mixer: str
    rank: int

    @property
    def name(self) -> str:
        if self.mixer in {"mha", "linear"}:
            return self.mixer
        return f"{self.mixer}-r{self.rank}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_specs(text: str) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for raw in text.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if item in {"mha", "linear"}:
            specs.append(ModelSpec(item, 0))
            continue
        if ":" in item:
            name, value = item.split(":", 1)
            specs.append(ModelSpec(name, int(value)))
            continue
        raise ValueError(f"cannot parse model spec: {raw}")
    return specs


def parse_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def distribute_fillers(total: int, slots: int, gen: torch.Generator) -> list[int]:
    if total <= 0:
        return [0] * slots
    probs = torch.rand(slots, generator=gen).clamp_min(1e-6)
    probs = probs / probs.sum()
    counts = torch.multinomial(probs, total, replacement=True, generator=gen)
    out = [0] * slots
    for idx in counts.tolist():
        out[idx] += 1
    return out


def add_fillers(tokens: list[int], count: int, filler_tokens: torch.Tensor, gen: torch.Generator) -> None:
    if count <= 0:
        return
    picks = torch.randint(0, filler_tokens.numel(), (count,), generator=gen)
    tokens.extend(filler_tokens[picks].tolist())


def make_mqar_data(
    samples: int,
    seq_len: int,
    num_pairs: int,
    num_queries: int,
    num_keys: int,
    num_values: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    min_len = 1 + 2 * num_pairs + 2 * num_queries
    if min_len > seq_len:
        raise ValueError(f"seq_len={seq_len} too short for MQAR; need at least {min_len}")
    gen = torch.Generator().manual_seed(seed)
    key_ids = torch.arange(KEY_OFFSET, KEY_OFFSET + num_keys)
    value_offset = KEY_OFFSET + num_keys
    value_ids = torch.arange(value_offset, value_offset + num_values)
    filler_offset = value_offset + num_values
    filler_tokens = torch.arange(filler_offset, filler_offset + 64)
    vocab_size = filler_offset + 64

    x = torch.full((samples, seq_len), PAD_ID, dtype=torch.long)
    labels = torch.full((samples, seq_len), -100, dtype=torch.long)
    for i in range(samples):
        keys = key_ids[torch.randperm(num_keys, generator=gen)[:num_pairs]]
        values = value_ids[torch.randint(0, num_values, (num_pairs,), generator=gen)]
        query_indices = torch.randint(0, num_pairs, (num_queries,), generator=gen)

        pair_events = [[int(k), int(v)] for k, v in zip(keys.tolist(), values.tolist(), strict=True)]
        query_events = [[QUERY_ID, int(keys[idx]), ANSWER_ID] for idx in query_indices.tolist()]
        base_len = 1 + 2 * num_pairs + 3 * num_queries
        if base_len > seq_len:
            raise ValueError(f"seq_len={seq_len} too short for {num_queries} MQAR queries; need {base_len}")
        filler_counts = distribute_fillers(seq_len - base_len, len(pair_events) + len(query_events) + 1, gen)
        tokens = [BOS_ID]
        slot = 0
        for event in pair_events:
            add_fillers(tokens, filler_counts[slot], filler_tokens, gen)
            slot += 1
            tokens.extend(event)
        for event, q_idx in zip(query_events, query_indices.tolist(), strict=True):
            add_fillers(tokens, filler_counts[slot], filler_tokens, gen)
            slot += 1
            answer_pos = len(tokens) + 2
            tokens.extend(event)
            labels[i, answer_pos] = int(values[q_idx] - value_offset)
        add_fillers(tokens, filler_counts[slot], filler_tokens, gen)
        tokens = tokens[:seq_len]
        x[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
    return x, labels, vocab_size, num_values


def make_mqar_candidate_data(
    samples: int,
    seq_len: int,
    num_pairs: int,
    num_queries: int,
    num_keys: int,
    num_values: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    if num_values < 2:
        raise ValueError("mqar-candidate requires num_values >= 2")
    min_len = 1 + 2 * num_pairs + 5 * num_queries
    if min_len > seq_len:
        raise ValueError(f"seq_len={seq_len} too short for MQAR candidate; need at least {min_len}")
    gen = torch.Generator().manual_seed(seed)
    key_ids = torch.arange(KEY_OFFSET, KEY_OFFSET + num_keys)
    value_offset = KEY_OFFSET + num_keys
    value_ids = torch.arange(value_offset, value_offset + num_values)
    filler_offset = value_offset + num_values
    filler_tokens = torch.arange(filler_offset, filler_offset + 64)
    vocab_size = filler_offset + 64

    x = torch.full((samples, seq_len), PAD_ID, dtype=torch.long)
    labels = torch.full((samples, seq_len), -100, dtype=torch.long)
    for i in range(samples):
        keys = key_ids[torch.randperm(num_keys, generator=gen)[:num_pairs]]
        value_idx = torch.randint(0, num_values, (num_pairs,), generator=gen)
        values = value_ids[value_idx]
        query_indices = torch.randint(0, num_pairs, (num_queries,), generator=gen)

        pair_events = [[int(k), int(v)] for k, v in zip(keys.tolist(), values.tolist(), strict=True)]
        query_events: list[list[int]] = []
        query_labels: list[int] = []
        for q_idx in query_indices.tolist():
            positive = bool(torch.randint(0, 2, (1,), generator=gen))
            correct_idx = int(value_idx[q_idx])
            if positive:
                cand_idx = correct_idx
            else:
                cand_idx = int(torch.randint(0, num_values - 1, (1,), generator=gen))
                if cand_idx >= correct_idx:
                    cand_idx += 1
            query_events.append([QUERY_ID, int(keys[q_idx]), CAND_ID, int(value_ids[cand_idx]), ANSWER_ID])
            query_labels.append(int(positive))

        base_len = 1 + 2 * num_pairs + 5 * num_queries
        filler_counts = distribute_fillers(seq_len - base_len, len(pair_events) + len(query_events) + 1, gen)
        tokens = [BOS_ID]
        slot = 0
        for event in pair_events:
            add_fillers(tokens, filler_counts[slot], filler_tokens, gen)
            slot += 1
            tokens.extend(event)
        for event, label in zip(query_events, query_labels, strict=True):
            add_fillers(tokens, filler_counts[slot], filler_tokens, gen)
            slot += 1
            answer_pos = len(tokens) + 4
            tokens.extend(event)
            labels[i, answer_pos] = label
        add_fillers(tokens, filler_counts[slot], filler_tokens, gen)
        tokens = tokens[:seq_len]
        x[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
    return x, labels, vocab_size, 2


def make_ruler_vt_data(
    samples: int,
    seq_len: int,
    num_assignments: int,
    num_queries: int,
    num_vars: int,
    num_values: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    min_len = 1 + 3 * num_assignments + 3 * num_queries
    if min_len > seq_len:
        raise ValueError(f"seq_len={seq_len} too short for ruler-vt; need at least {min_len}")
    gen = torch.Generator().manual_seed(seed)
    var_ids = torch.arange(KEY_OFFSET, KEY_OFFSET + num_vars)
    value_offset = KEY_OFFSET + num_vars
    value_ids = torch.arange(value_offset, value_offset + num_values)
    filler_offset = value_offset + num_values
    filler_tokens = torch.arange(filler_offset, filler_offset + 64)
    vocab_size = filler_offset + 64

    x = torch.full((samples, seq_len), PAD_ID, dtype=torch.long)
    labels = torch.full((samples, seq_len), -100, dtype=torch.long)
    for i in range(samples):
        latest = torch.randint(0, num_values, (num_vars,), generator=gen)
        assignments = []
        for _ in range(num_assignments):
            var_idx = int(torch.randint(0, num_vars, (1,), generator=gen))
            value_idx = int(torch.randint(0, num_values, (1,), generator=gen))
            latest[var_idx] = value_idx
            assignments.append([SET_ID, int(var_ids[var_idx]), int(value_ids[value_idx])])
        query_vars = torch.randint(0, num_vars, (num_queries,), generator=gen)
        query_events = [[QUERY_ID, int(var_ids[v]), ANSWER_ID] for v in query_vars.tolist()]

        base_len = 1 + 3 * num_assignments + 3 * num_queries
        filler_counts = distribute_fillers(seq_len - base_len, len(assignments) + len(query_events) + 1, gen)
        tokens = [BOS_ID]
        slot = 0
        for event in assignments:
            add_fillers(tokens, filler_counts[slot], filler_tokens, gen)
            slot += 1
            tokens.extend(event)
        for event, var_idx in zip(query_events, query_vars.tolist(), strict=True):
            add_fillers(tokens, filler_counts[slot], filler_tokens, gen)
            slot += 1
            answer_pos = len(tokens) + 2
            tokens.extend(event)
            labels[i, answer_pos] = int(latest[var_idx])
        add_fillers(tokens, filler_counts[slot], filler_tokens, gen)
        tokens = tokens[:seq_len]
        x[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
    return x, labels, vocab_size, num_values


def make_ruler_vt_candidate_data(
    samples: int,
    seq_len: int,
    num_assignments: int,
    num_queries: int,
    num_vars: int,
    num_values: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    if num_values < 2:
        raise ValueError("ruler-vt-candidate requires num_values >= 2")
    min_len = 1 + 3 * num_assignments + 5 * num_queries
    if min_len > seq_len:
        raise ValueError(f"seq_len={seq_len} too short for ruler-vt candidate; need at least {min_len}")
    gen = torch.Generator().manual_seed(seed)
    var_ids = torch.arange(KEY_OFFSET, KEY_OFFSET + num_vars)
    value_offset = KEY_OFFSET + num_vars
    value_ids = torch.arange(value_offset, value_offset + num_values)
    filler_offset = value_offset + num_values
    filler_tokens = torch.arange(filler_offset, filler_offset + 64)
    vocab_size = filler_offset + 64

    x = torch.full((samples, seq_len), PAD_ID, dtype=torch.long)
    labels = torch.full((samples, seq_len), -100, dtype=torch.long)
    for i in range(samples):
        latest = torch.randint(0, num_values, (num_vars,), generator=gen)
        assignments = []
        for _ in range(num_assignments):
            var_idx = int(torch.randint(0, num_vars, (1,), generator=gen))
            value_idx = int(torch.randint(0, num_values, (1,), generator=gen))
            latest[var_idx] = value_idx
            assignments.append([SET_ID, int(var_ids[var_idx]), int(value_ids[value_idx])])

        query_events: list[list[int]] = []
        query_labels: list[int] = []
        query_vars = torch.randint(0, num_vars, (num_queries,), generator=gen)
        for var_idx in query_vars.tolist():
            positive = bool(torch.randint(0, 2, (1,), generator=gen))
            correct_idx = int(latest[var_idx])
            if positive:
                cand_idx = correct_idx
            else:
                cand_idx = int(torch.randint(0, num_values - 1, (1,), generator=gen))
                if cand_idx >= correct_idx:
                    cand_idx += 1
            query_events.append([QUERY_ID, int(var_ids[var_idx]), CAND_ID, int(value_ids[cand_idx]), ANSWER_ID])
            query_labels.append(int(positive))

        base_len = 1 + 3 * num_assignments + 5 * num_queries
        filler_counts = distribute_fillers(seq_len - base_len, len(assignments) + len(query_events) + 1, gen)
        tokens = [BOS_ID]
        slot = 0
        for event in assignments:
            add_fillers(tokens, filler_counts[slot], filler_tokens, gen)
            slot += 1
            tokens.extend(event)
        for event, label in zip(query_events, query_labels, strict=True):
            add_fillers(tokens, filler_counts[slot], filler_tokens, gen)
            slot += 1
            answer_pos = len(tokens) + 4
            tokens.extend(event)
            labels[i, answer_pos] = label
        add_fillers(tokens, filler_counts[slot], filler_tokens, gen)
        tokens = tokens[:seq_len]
        x[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
    return x, labels, vocab_size, 2


class CausalMHA(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
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
        self.heads = heads
        self.head_dim = dim // heads
        self.features = features
        self.eps = eps
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.register_buffer("projection_matrix", torch.empty(heads, features, self.head_dim), persistent=False)
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
    def __init__(self, dim: int, heads: int, spec: ModelSpec, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        if spec.mixer == "mha":
            self.mixer = CausalMHA(dim, heads)
        elif spec.mixer == "linear":
            self.mixer = CausalLinearAttention(dim, heads)
        elif spec.mixer == "performer":
            self.mixer = CausalPerformerAttention(dim, heads, features=spec.rank)
        elif spec.mixer == "rope-lsso":
            self.mixer = RoPELSSO(dim, heads, rank=spec.rank, causal=True, causal_chunk_size=256)
        elif spec.mixer == "lsso":
            self.mixer = LSSO(dim, heads, rank=spec.rank, causal=True, causal_chunk_size=256)
        else:
            raise ValueError(f"unknown mixer: {spec.mixer}")
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CausalRecallModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        seq_len: int,
        spec: ModelSpec,
        dim: int,
        depth: int,
        heads: int,
        mlp_ratio: float,
        position_mode: str,
    ) -> None:
        super().__init__()
        if position_mode not in {"learned", "none"}:
            raise ValueError(f"unknown position_mode={position_mode}")
        self.token_embed = nn.Embedding(vocab_size, dim, padding_idx=PAD_ID)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, dim)) if position_mode == "learned" else None
        self.blocks = nn.ModuleList([CausalBlock(dim, heads, spec, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_labels)
        if self.pos_embed is not None:
            nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embed(input_ids)
        if self.pos_embed is not None:
            x = x + self.pos_embed[:, : input_ids.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def mixer_macs_g(spec: ModelSpec, seq_len: int, dim: int, heads: int, depth: int) -> float:
    if spec.mixer == "mha":
        per_layer = 4 * seq_len * dim * dim + 2 * seq_len * seq_len * dim
    elif spec.mixer == "linear":
        per_layer = 4 * seq_len * dim * dim + 2 * seq_len * dim * dim
    elif spec.mixer == "performer":
        f = spec.rank
        per_layer = 4 * seq_len * dim * dim + 4 * seq_len * dim * f + seq_len * heads * f
    elif spec.mixer in {"lsso", "rope-lsso"}:
        r = spec.rank
        per_layer = seq_len * dim * (heads * r + dim) + seq_len * dim * dim
        per_layer += heads * seq_len * r * r + 2 * seq_len * r * dim + dim * r * r + heads * r * r * r
    else:
        raise ValueError(spec.mixer)
    return depth * per_layer / 1e9


def batch_metrics(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, float, int]:
    loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100)
    valid = labels.ne(-100)
    correct = (logits.argmax(dim=-1).eq(labels) & valid).sum().item()
    total = valid.sum().item()
    return loss, correct / max(total, 1), int(total)


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, labels: torch.Tensor, batch_size: int, device: torch.device) -> tuple[float, float]:
    model.eval()
    losses = []
    correct = 0
    total = 0
    for start in range(0, x.shape[0], batch_size):
        xb = x[start : start + batch_size].to(device, non_blocking=True)
        yb = labels[start : start + batch_size].to(device, non_blocking=True)
        logits = model(xb)
        loss, _acc, n = batch_metrics(logits, yb)
        losses.append(loss.item())
        valid = yb.ne(-100)
        correct += (logits.argmax(dim=-1).eq(yb) & valid).sum().item()
        total += n
    return sum(losses) / len(losses), correct / max(total, 1)


def train_one(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    batch_fn: Callable[[], tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> dict[str, float]:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    last_loss = float("nan")
    last_acc = float("nan")
    token_count = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        if batch_fn is None:
            idx = torch.randint(0, train_x.shape[0], (args.batch_size,))
            xb_cpu = train_x[idx]
            yb_cpu = train_y[idx]
        else:
            xb_cpu, yb_cpu = batch_fn()
        xb = xb_cpu.to(device, non_blocking=True)
        yb = yb_cpu.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
            logits = model(xb)
            loss, acc, n = batch_metrics(logits, yb)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        last_loss = loss.item()
        last_acc = acc
        token_count += xb.numel()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print({"step": step, "train_loss": last_loss, "train_acc": last_acc}, flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else float("nan")
    val_loss, val_acc = evaluate(model, val_x, val_y, args.eval_batch_size, device)
    return {
        "train_loss": last_loss,
        "train_acc": last_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "seconds": elapsed,
        "tokens_per_sec": token_count / max(elapsed, 1e-9),
        "peak_mem_mb": peak_mem_mb,
    }


def make_data(task: str, seq_len: int, args: argparse.Namespace, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
    if task == "mqar":
        difficulty = max(8, seq_len // args.mqar_pair_divisor)
        train_x, train_y, vocab_size, num_labels = make_mqar_data(
            args.train_samples,
            seq_len,
            difficulty,
            args.num_queries,
            args.num_keys,
            args.num_values,
            seed,
        )
        val_x, val_y, _vocab, _labels = make_mqar_data(
            args.val_samples,
            seq_len,
            difficulty,
            args.num_queries,
            args.num_keys,
            args.num_values,
            seed + 100000,
        )
    elif task == "mqar-candidate":
        difficulty = max(8, seq_len // args.mqar_pair_divisor)
        train_x, train_y, vocab_size, num_labels = make_mqar_candidate_data(
            args.train_samples,
            seq_len,
            difficulty,
            args.num_queries,
            args.num_keys,
            args.num_values,
            seed,
        )
        val_x, val_y, _vocab, _labels = make_mqar_candidate_data(
            args.val_samples,
            seq_len,
            difficulty,
            args.num_queries,
            args.num_keys,
            args.num_values,
            seed + 100000,
        )
    elif task == "ruler-vt":
        difficulty = max(8, seq_len // args.ruler_assignment_divisor)
        train_x, train_y, vocab_size, num_labels = make_ruler_vt_data(
            args.train_samples,
            seq_len,
            difficulty,
            args.num_queries,
            args.num_keys,
            args.num_values,
            seed,
        )
        val_x, val_y, _vocab, _labels = make_ruler_vt_data(
            args.val_samples,
            seq_len,
            difficulty,
            args.num_queries,
            args.num_keys,
            args.num_values,
            seed + 100000,
        )
    elif task == "ruler-vt-candidate":
        difficulty = max(8, seq_len // args.ruler_assignment_divisor)
        train_x, train_y, vocab_size, num_labels = make_ruler_vt_candidate_data(
            args.train_samples,
            seq_len,
            difficulty,
            args.num_queries,
            args.num_keys,
            args.num_values,
            seed,
        )
        val_x, val_y, _vocab, _labels = make_ruler_vt_candidate_data(
            args.val_samples,
            seq_len,
            difficulty,
            args.num_queries,
            args.num_keys,
            args.num_values,
            seed + 100000,
        )
    else:
        raise ValueError(task)
    return train_x, train_y, val_x, val_y, vocab_size, num_labels, difficulty


def make_online_batch_fn(
    task: str,
    seq_len: int,
    difficulty: int,
    args: argparse.Namespace,
    seed: int,
) -> Callable[[], tuple[torch.Tensor, torch.Tensor]]:
    counter = 0

    def next_batch() -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal counter
        batch_seed = seed + counter * 104729
        counter += 1
        if task == "mqar":
            x, y, _vocab_size, _num_labels = make_mqar_data(
                args.batch_size,
                seq_len,
                difficulty,
                args.num_queries,
                args.num_keys,
                args.num_values,
                batch_seed,
            )
        elif task == "mqar-candidate":
            x, y, _vocab_size, _num_labels = make_mqar_candidate_data(
                args.batch_size,
                seq_len,
                difficulty,
                args.num_queries,
                args.num_keys,
                args.num_values,
                batch_seed,
            )
        elif task == "ruler-vt":
            x, y, _vocab_size, _num_labels = make_ruler_vt_data(
                args.batch_size,
                seq_len,
                difficulty,
                args.num_queries,
                args.num_keys,
                args.num_values,
                batch_seed,
            )
        elif task == "ruler-vt-candidate":
            x, y, _vocab_size, _num_labels = make_ruler_vt_candidate_data(
                args.batch_size,
                seq_len,
                difficulty,
                args.num_queries,
                args.num_keys,
                args.num_values,
                batch_seed,
            )
        else:
            raise ValueError(task)
        return x, y

    return next_batch


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small/medium causal recall benchmark for LSSO.")
    parser.add_argument("--tasks", default="mqar,ruler-vt")
    parser.add_argument("--seq-lens", type=parse_ints, default=parse_ints("256,512,1024"))
    parser.add_argument("--models", default="mha,linear,performer:32,rope-lsso:16,rope-lsso:32")
    parser.add_argument("--out", default="paper_results/causal_recall/trial_summary.tsv")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--train-samples", type=int, default=4096)
    parser.add_argument("--val-samples", type=int, default=1024)
    parser.add_argument("--online-train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--position-mode", choices=("learned", "none"), default="learned")
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-keys", type=int, default=512)
    parser.add_argument("--num-values", type=int, default=128)
    parser.add_argument("--num-queries", type=int, default=4)
    parser.add_argument("--mqar-pair-divisor", type=int, default=8)
    parser.add_argument("--ruler-assignment-divisor", type=int, default=6)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = parse_specs(args.models)
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    device = torch.device(args.device)
    rows: list[dict] = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["models"] = [spec.name for spec in specs]
    (out.parent / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    for task in tasks:
        for seq_len in args.seq_lens:
            data_seed = args.seed + seq_len * 13 + sum(ord(c) for c in task)
            train_x, train_y, val_x, val_y, vocab_size, num_labels, difficulty = make_data(task, seq_len, args, data_seed)
            for spec in specs:
                run_seed = args.seed + seq_len * 101 + difficulty * 17 + spec.rank + sum(ord(c) for c in spec.name)
                set_seed(run_seed)
                model = CausalRecallModel(
                    vocab_size=vocab_size,
                    num_labels=num_labels,
                    seq_len=seq_len,
                    spec=spec,
                    dim=args.dim,
                    depth=args.depth,
                    heads=args.heads,
                    mlp_ratio=args.mlp_ratio,
                    position_mode=args.position_mode,
                )
                print({"task": task, "seq_len": seq_len, "difficulty": difficulty, "model": spec.name}, flush=True)
                batch_fn = (
                    make_online_batch_fn(task, seq_len, difficulty, args, run_seed + 99991)
                    if args.online_train
                    else None
                )
                metrics = train_one(model, train_x, train_y, val_x, val_y, args, device, batch_fn=batch_fn)
                row = {
                    "task": task,
                    "seq_len": seq_len,
                    "difficulty": difficulty,
                    "model": spec.name,
                    "mixer": spec.mixer,
                    "rank": spec.rank,
                    "seed": args.seed,
                    "steps": args.steps,
                    "dim": args.dim,
                    "depth": args.depth,
                    "heads": args.heads,
                    "position_mode": args.position_mode,
                    "online_train": args.online_train,
                    "params_m": count_params(model) / 1e6,
                    "mixer_macs_g": mixer_macs_g(spec, seq_len, args.dim, args.heads, args.depth),
                    **metrics,
                }
                print(json.dumps(row, sort_keys=True), flush=True)
                rows.append(row)
                write_rows(out, rows)
    write_rows(out, rows)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
