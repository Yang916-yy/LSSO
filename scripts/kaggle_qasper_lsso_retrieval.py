from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


def ensure_deps() -> None:
    try:
        import datasets  # noqa: F401
        import transformers  # noqa: F401
        import torch  # noqa: F401
    except Exception:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "datasets",
                "transformers",
                "accelerate",
                "tqdm",
            ]
        )


ensure_deps()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer


@dataclass
class RetrievalData:
    train_pairs: list[tuple[str, str]]
    eval_queries: list[str]
    eval_positives: list[set[int]]
    eval_corpus: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Kaggle QASPER LSSO/MHA retrieval experiment")
    parser.add_argument("--dataset", default="allenai/qasper")
    parser.add_argument("--dataset-config", default="qasper")
    parser.add_argument("--tokenizer-name", default="bert-base-uncased")
    parser.add_argument("--mixer", choices=["lsso", "mha", "both"], default="both")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gamma-max", type=float, default=0.3)
    parser.add_argument("--theta-gamma-init", type=float, default=-4.0)
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--max-query-len", type=int, default=96)
    parser.add_argument("--max-doc-len", type=int, default=1024)
    parser.add_argument("--chunk-words", type=int, default=420)
    parser.add_argument("--chunk-overlap", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--max-train-pairs", type=int, default=0)
    parser.add_argument("--max-eval-queries", type=int, default=0)
    parser.add_argument("--max-eval-corpus", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out-dir", default="/kaggle/working/lsso_qasper_runs")
    parser.add_argument("--cache-dir", default="/kaggle/working/lsso_qasper_cache")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def window_words(text: str, max_words: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    step = max(1, max_words - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + max_words])
        if chunk:
            chunks.append(chunk)
        if start + max_words >= len(words):
            break
    return chunks


def flatten_qasper_paper(row: dict, chunk_words: int, chunk_overlap: int):
    title = row.get("title") or ""
    full_text = row.get("full_text") or {}
    section_names = full_text.get("section_name") or []
    paragraphs_by_section = full_text.get("paragraphs") or []

    chunks: list[str] = []
    para_to_chunk: dict[str, set[int]] = defaultdict(set)

    for sec_name, paragraphs in zip(section_names, paragraphs_by_section):
        if not isinstance(paragraphs, list):
            continue
        for para in paragraphs:
            if not para or not isinstance(para, str):
                continue
            prefix = f"{title}. {sec_name}. " if sec_name else f"{title}. "
            pieces = window_words(prefix + para.strip(), chunk_words, chunk_overlap)
            for piece in pieces:
                idx = len(chunks)
                chunks.append(piece)
                para_to_chunk[normalize_text(para)].add(idx)
                para_to_chunk[normalize_text(piece)].add(idx)
    return chunks, para_to_chunk


def collect_evidence_indices(answer_group, para_to_chunk: dict[str, set[int]]) -> set[int]:
    positives: set[int] = set()
    if not isinstance(answer_group, dict):
        return positives

    answers = answer_group.get("answer") or []
    for answer in answers:
        if not isinstance(answer, dict) or answer.get("unanswerable"):
            continue
        evidence_items = []
        evidence_items.extend(answer.get("evidence") or [])
        evidence_items.extend(answer.get("highlighted_evidence") or [])
        for ev in evidence_items:
            if not ev or not isinstance(ev, str) or ev.startswith("FLOAT SELECTED"):
                continue
            ev_norm = normalize_text(ev)
            if ev_norm in para_to_chunk:
                positives.update(para_to_chunk[ev_norm])
                continue
            for key, idxs in para_to_chunk.items():
                if ev_norm and (ev_norm in key or key in ev_norm):
                    positives.update(idxs)
                    break
    return positives


def build_split_examples(rows, chunk_words: int, chunk_overlap: int):
    corpus: list[str] = []
    queries: list[str] = []
    positives: list[set[int]] = []

    for row in tqdm(rows, desc="build-qasper"):
        paper_chunks, para_to_local = flatten_qasper_paper(row, chunk_words, chunk_overlap)
        if not paper_chunks:
            continue

        base = len(corpus)
        corpus.extend(paper_chunks)
        para_to_global = {k: {base + i for i in idxs} for k, idxs in para_to_local.items()}

        qas = row.get("qas") or {}
        questions = qas.get("question") or []
        answers = qas.get("answers") or []
        for q, answer_group in zip(questions, answers):
            pos = collect_evidence_indices(answer_group, para_to_global)
            if q and pos:
                queries.append(str(q))
                positives.append(pos)
    return corpus, queries, positives


def load_qasper_retrieval(args: argparse.Namespace) -> RetrievalData:
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = (
        f"qasper_cw{args.chunk_words}_ov{args.chunk_overlap}_"
        f"train{args.max_train_pairs}_eq{args.max_eval_queries}_ec{args.max_eval_corpus}.pt"
    )
    cache_path = cache_dir / cache_name
    if cache_path.exists():
        print(f"loading processed cache: {cache_path}", flush=True)
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    raw = load_dataset(args.dataset, args.dataset_config)
    train_corpus, train_queries, train_pos = build_split_examples(
        raw["train"], args.chunk_words, args.chunk_overlap
    )
    eval_corpus, eval_queries, eval_pos = build_split_examples(
        raw["test"], args.chunk_words, args.chunk_overlap
    )

    train_pairs = []
    rng = random.Random(args.seed)
    for query, pos in zip(train_queries, train_pos):
        doc_idx = rng.choice(sorted(pos))
        if 0 <= doc_idx < len(train_corpus):
            train_pairs.append((query, train_corpus[doc_idx]))
    rng.shuffle(train_pairs)
    if args.max_train_pairs:
        train_pairs = train_pairs[: args.max_train_pairs]

    if args.max_eval_corpus:
        keep = set(range(min(args.max_eval_corpus, len(eval_corpus))))
        filtered_queries = []
        filtered_pos = []
        for query, pos in zip(eval_queries, eval_pos):
            pos = pos & keep
            if pos:
                filtered_queries.append(query)
                filtered_pos.append(pos)
        eval_queries = filtered_queries
        eval_pos = filtered_pos
        eval_corpus = eval_corpus[: args.max_eval_corpus]

    if args.max_eval_queries:
        eval_queries = eval_queries[: args.max_eval_queries]
        eval_pos = eval_pos[: args.max_eval_queries]

    data = RetrievalData(
        train_pairs=train_pairs,
        eval_queries=eval_queries,
        eval_positives=eval_pos,
        eval_corpus=eval_corpus,
    )
    torch.save(data, cache_path)
    print(f"saved processed cache: {cache_path}", flush=True)
    return data


class PairDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.pairs[idx]


def build_collate(tokenizer, max_query_len: int, max_doc_len: int):
    def collate(batch: list[tuple[str, str]]):
        q = [item[0] for item in batch]
        d = [item[1] for item in batch]
        q_tok = tokenizer(q, padding=True, truncation=True, max_length=max_query_len, return_tensors="pt")
        d_tok = tokenizer(d, padding=True, truncation=True, max_length=max_doc_len, return_tensors="pt")
        return q_tok, d_tok

    return collate


@dataclass
class LSSOAux:
    UtU: torch.Tensor | None
    local: torch.Tensor
    correction: torch.Tensor
    mu: torch.Tensor
    gamma: torch.Tensor


def lsso_core(
    U: torch.Tensor,
    C: torch.Tensor,
    mu: torch.Tensor,
    gamma: torch.Tensor,
    eye: torch.Tensor,
    no_global: bool = False,
    return_aux: bool = False,
):
    B, H, N, r = U.shape
    dh = C.shape[-1]
    if mu.dim() == 1:
        mu = mu.view(1, H, 1, 1)
    if gamma.dim() == 1:
        gamma = gamma.view(1, H, 1, 1)

    inv_mu = mu.reciprocal()
    local = inv_mu * C
    U_bh = U.flatten(0, 1)
    C_bh = C.flatten(0, 1)
    Ut = U_bh.transpose(1, 2)

    if no_global:
        correction = torch.zeros_like(local)
        UtU = torch.bmm(Ut, U_bh).view(B, H, r, r) if return_aux else None
        Y = local
    else:
        UtU = torch.bmm(Ut, U_bh).view(B, H, r, r)
        UtC = torch.bmm(Ut, C_bh).view(B, H, r, dh)
        G = eye.float() + (gamma * inv_mu).float() * UtU.float()
        K = torch.linalg.solve_ex(
            G.view(B * H, r, r),
            UtC.float().view(B * H, r, dh),
            check_errors=False,
        ).result.to(U.dtype)
        UK = torch.bmm(U_bh, K).view(B, H, N, dh)
        correction = gamma * inv_mu * inv_mu * UK
        Y = local - correction

    if return_aux:
        return Y, LSSOAux(UtU=UtU, local=local, correction=correction, mu=mu, gamma=gamma)
    return Y


class LSSO(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        rank: int,
        gamma_max: float,
        theta_gamma_init: float,
        eps: float = 1e-5,
        no_global: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rank = rank
        self.eps = eps
        self.gamma_max = gamma_max
        self.no_global = no_global
        self.w_uc = nn.Linear(dim, num_heads * rank + dim, bias=False)
        self.w_o = nn.Linear(dim, dim, bias=False)
        self.theta_mu = nn.Parameter(torch.zeros(num_heads))
        self.theta_gamma = nn.Parameter(torch.full((num_heads,), float(theta_gamma_init)))
        self.register_buffer("_eye", torch.eye(rank).view(1, 1, rank, rank), persistent=False)
        self.record_diagnostics = False
        self.last_diag: dict[str, float] | None = None

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, D = x.shape
        H = self.num_heads
        r = self.rank
        dh = self.head_dim
        UC = self.w_uc(x)
        U, C = UC.split((H * r, D), dim=-1)
        U = U.view(B, N, H, r).transpose(1, 2).contiguous()
        C = C.view(B, N, H, dh).transpose(1, 2).contiguous()
        U = U * torch.rsqrt(torch.mean(U * U, dim=-1, keepdim=True) + self.eps)
        if valid_mask is not None:
            mask = valid_mask[:, None, :, None].to(dtype=x.dtype)
            U = U * mask
            C = C * mask

        mu = F.softplus(self.theta_mu) + self.eps
        gamma = self.gamma_max * torch.sigmoid(self.theta_gamma)
        if self.no_global:
            gamma = torch.zeros_like(gamma)

        if self.record_diagnostics:
            Y, aux = lsso_core(
                U, C, mu, gamma, self._eye, no_global=self.no_global, return_aux=True
            )
            with torch.no_grad():
                eigvals = torch.linalg.eigvalsh(aux.UtU.float()).clamp_min(0.0)
                eig_sum = eigvals.sum(dim=-1)
                eff_rank = (eig_sum * eig_sum) / (eigvals.square().sum(dim=-1).clamp_min(self.eps))
                corr = aux.correction.float().norm(dim=(-2, -1))
                local = aux.local.float().norm(dim=(-2, -1)).clamp_min(self.eps)
                self.last_diag = {
                    "gamma_over_mu": (aux.gamma / aux.mu).mean().item(),
                    "effective_rank": eff_rank.mean().item(),
                    "correction_ratio": (corr / local).mean().item(),
                }
        else:
            Y = lsso_core(U, C, mu, gamma, self._eye, no_global=self.no_global)

        Y = Y.transpose(1, 2).contiguous().view(B, N, D)
        Y = self.w_o(Y)
        if valid_mask is not None:
            Y = Y * valid_mask[:, :, None].to(dtype=Y.dtype)
        return Y


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BertStyleBlock(nn.Module):
    def __init__(self, args: argparse.Namespace, mixer: str) -> None:
        super().__init__()
        self.uses_mha = mixer == "mha"
        if self.uses_mha:
            self.mixer = nn.MultiheadAttention(
                args.dim,
                args.num_heads,
                dropout=args.dropout,
                batch_first=True,
            )
        else:
            self.mixer = LSSO(
                args.dim,
                args.num_heads,
                args.rank,
                args.gamma_max,
                args.theta_gamma_init,
            )
        self.attn_norm = nn.LayerNorm(args.dim)
        self.ffn = MLP(args.dim, args.mlp_ratio, args.dropout)
        self.ffn_norm = nn.LayerNorm(args.dim)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        if self.uses_mha:
            mixed, _ = self.mixer(
                x,
                x,
                x,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
        else:
            mixed = self.mixer(x, valid_mask=~key_padding_mask)
        x = self.attn_norm(x + self.dropout(mixed))
        x = x.masked_fill(key_padding_mask[:, :, None], 0.0)
        x = self.ffn_norm(x + self.dropout(self.ffn(x)))
        x = x.masked_fill(key_padding_mask[:, :, None], 0.0)
        return x


class BertStyleEmbedder(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, args: argparse.Namespace, mixer: str) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.pooling = args.pooling
        max_len = max(args.max_query_len, args.max_doc_len)
        self.token_embed = nn.Embedding(vocab_size, args.dim, padding_idx=pad_id)
        self.pos_embed = nn.Embedding(max_len, args.dim)
        self.type_embed = nn.Embedding(2, args.dim)
        self.embed_norm = nn.LayerNorm(args.dim)
        self.embed_drop = nn.Dropout(args.dropout)
        self.blocks = nn.ModuleList([BertStyleBlock(args, mixer) for _ in range(args.depth)])
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B, N = input_ids.shape
        key_padding_mask = attention_mask.eq(0)
        pos = torch.arange(N, device=input_ids.device).view(1, N).expand(B, N)
        typ = torch.zeros_like(input_ids)
        x = self.token_embed(input_ids) + self.pos_embed(pos) + self.type_embed(typ)
        x = self.embed_drop(self.embed_norm(x))
        x = x.masked_fill(key_padding_mask[:, :, None], 0.0)
        for block in self.blocks:
            x = block(x, key_padding_mask)

        if self.pooling == "cls":
            emb = x[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).to(dtype=x.dtype)
            emb = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
        return F.normalize(emb, dim=-1)

    def lsso_layers(self) -> list[LSSO]:
        return [m for m in self.modules() if isinstance(m, LSSO)]


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def collect_lsso_diag(model: nn.Module) -> dict[str, float]:
    base = unwrap_model(model)
    layers = base.lsso_layers() if hasattr(base, "lsso_layers") else []
    diags = [layer.last_diag for layer in layers if layer.last_diag]
    if not diags:
        return {}
    return {
        "diag_gamma_over_mu": float(np.mean([d["gamma_over_mu"] for d in diags])),
        "diag_effective_rank": float(np.mean([d["effective_rank"] for d in diags])),
        "diag_correction_ratio": float(np.mean([d["correction_ratio"] for d in diags])),
    }


def set_lsso_diag(model: nn.Module, enabled: bool) -> None:
    base = unwrap_model(model)
    for layer in base.lsso_layers() if hasattr(base, "lsso_layers") else []:
        layer.record_diagnostics = enabled


def build_model(tokenizer, args: argparse.Namespace, mixer: str, device: torch.device) -> nn.Module:
    model = BertStyleEmbedder(
        vocab_size=len(tokenizer),
        pad_id=tokenizer.pad_token_id or 0,
        args=args,
        mixer=mixer,
    )
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"using DataParallel on {torch.cuda.device_count()} GPUs", flush=True)
        model = nn.DataParallel(model)
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    use_amp: bool,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_count = 0
    optimizer.zero_grad(set_to_none=True)
    for step, (q, d) in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        q_ids = q["input_ids"].to(device, non_blocking=True)
        q_mask = q["attention_mask"].to(device, non_blocking=True)
        d_ids = d["input_ids"].to(device, non_blocking=True)
        d_mask = d["attention_mask"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            q_emb = model(q_ids, q_mask)
            d_emb = model(d_ids, d_mask)
            logits = torch.matmul(q_emb, d_emb.transpose(0, 1)) / args.temperature
            targets = torch.arange(logits.shape[0], device=device)
            loss = F.cross_entropy(logits, targets) / args.grad_accum

        scaler.scale(loss).backward()
        if step % args.grad_accum == 0 or step == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch = q_ids.shape[0]
        total_loss += loss.item() * args.grad_accum * batch
        total_acc += (logits.argmax(dim=1) == targets).float().mean().item() * batch
        total_count += batch
    return {"loss": total_loss / max(1, total_count), "acc": total_acc / max(1, total_count)}


@torch.no_grad()
def encode_texts(
    model: nn.Module,
    tokenizer,
    texts: list[str],
    max_len: int,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> torch.Tensor:
    model.eval()
    outs = []
    for start in tqdm(range(0, len(texts), batch_size), desc="encode", leave=False):
        batch = texts[start : start + batch_size]
        tok = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        ids = tok["input_ids"].to(device, non_blocking=True)
        mask = tok["attention_mask"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outs.append(model(ids, mask).float().cpu())
    return torch.cat(outs, dim=0)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    tokenizer,
    data: RetrievalData,
    device: torch.device,
    args: argparse.Namespace,
    use_amp: bool,
) -> dict[str, float]:
    set_lsso_diag(model, True)
    corpus_emb = encode_texts(
        model, tokenizer, data.eval_corpus, args.max_doc_len, args.eval_batch_size, device, use_amp
    )
    query_emb = encode_texts(
        model, tokenizer, data.eval_queries, args.max_query_len, args.eval_batch_size, device, use_amp
    )
    set_lsso_diag(model, False)

    recall1 = recall10 = recall50 = mrr10 = 0.0
    corpus_t = corpus_emb.transpose(0, 1)
    k = min(50, corpus_emb.shape[0])
    for i in tqdm(range(query_emb.shape[0]), desc="retrieval", leave=False):
        top = torch.topk(torch.matmul(query_emb[i], corpus_t), k=k).indices.tolist()
        pos = data.eval_positives[i]
        recall1 += float(top[0] in pos)
        recall10 += float(any(idx in pos for idx in top[:10]))
        recall50 += float(any(idx in pos for idx in top[:50]))
        for rank, idx in enumerate(top[:10], start=1):
            if idx in pos:
                mrr10 += 1.0 / rank
                break
    n = max(1, len(data.eval_queries))
    metrics = {
        "recall@1": recall1 / n,
        "recall@10": recall10 / n,
        "recall@50": recall50 / n,
        "mrr@10": mrr10 / n,
    }
    metrics.update(collect_lsso_diag(model))
    return metrics


def run_one_mixer(mixer: str, tokenizer, data: RetrievalData, args: argparse.Namespace) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    model = build_model(tokenizer, args, mixer, device)
    params = sum(p.numel() for p in unwrap_model(model).parameters())
    print(f"mixer={mixer} params={params:,} device={device} amp={use_amp}", flush=True)

    loader = DataLoader(
        PairDataset(data.train_pairs),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        collate_fn=build_collate(tokenizer, args.max_query_len, args.max_doc_len),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = (
        f"{time.strftime('%Y%m%d-%H%M%S')}_qasper_{mixer}_r{args.rank}_"
        f"d{args.dim}_L{args.depth}_h{args.num_heads}_doc{args.max_doc_len}"
    )
    log_path = out_dir / f"{run_name}.jsonl"
    ckpt_path = out_dir / f"{run_name}.pt"
    best_r10 = 0.0

    with log_path.open("w", encoding="utf-8") as f:
        header = {
            "args": vars(args),
            "mixer": mixer,
            "params": params,
            "train_pairs": len(data.train_pairs),
            "eval_queries": len(data.eval_queries),
            "eval_corpus": len(data.eval_corpus),
            "num_gpus": torch.cuda.device_count(),
        }
        f.write(json.dumps(header, sort_keys=True) + "\n")
        for epoch in range(1, args.epochs + 1):
            print(f"[{mixer}] epoch {epoch}/{args.epochs} train", flush=True)
            train_metrics = train_one_epoch(model, loader, optimizer, scaler, device, args, use_amp)
            print(f"[{mixer}] epoch {epoch}/{args.epochs} eval", flush=True)
            eval_metrics = evaluate(model, tokenizer, data, device, args, use_amp)
            row = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **eval_metrics,
            }
            print(json.dumps(row, sort_keys=True), flush=True)
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
            if eval_metrics["recall@10"] > best_r10:
                best_r10 = eval_metrics["recall@10"]
                torch.save(
                    {
                        "model": unwrap_model(model).state_dict(),
                        "args": vars(args),
                        "mixer": mixer,
                        "epoch": epoch,
                        "metrics": eval_metrics,
                    },
                    ckpt_path,
                )
    print(f"[{mixer}] best_recall@10={best_r10:.4f} log={log_path} ckpt={ckpt_path}", flush=True)
    return log_path


def summarize_logs(paths: list[Path]) -> None:
    print("\nsummary", flush=True)
    for path in paths:
        rows = []
        header = None
        for line in path.read_text(encoding="utf-8").splitlines():
            obj = json.loads(line)
            if "args" in obj:
                header = obj
            elif "epoch" in obj:
                rows.append(obj)
        best = max(rows, key=lambda r: r["recall@10"])
        print(
            f"{header['mixer']:>4} params={header['params']:,} "
            f"best_epoch={best['epoch']} r@1={best['recall@1']:.4f} "
            f"r@10={best['recall@10']:.4f} r@50={best['recall@50']:.4f} "
            f"mrr@10={best['mrr@10']:.4f}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"gpu{i}: {torch.cuda.get_device_name(i)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    data = load_qasper_retrieval(args)
    print(
        f"data train_pairs={len(data.train_pairs)} eval_queries={len(data.eval_queries)} "
        f"eval_corpus={len(data.eval_corpus)}",
        flush=True,
    )
    if not data.train_pairs or not data.eval_queries or not data.eval_corpus:
        raise RuntimeError("QASPER processing produced empty train/eval data.")

    mixers = ["lsso", "mha"] if args.mixer == "both" else [args.mixer]
    logs = [run_one_mixer(mixer, tokenizer, data, args) for mixer in mixers]
    summarize_logs(logs)


if __name__ == "__main__":
    main()
