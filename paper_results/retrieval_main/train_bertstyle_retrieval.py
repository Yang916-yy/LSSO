from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from examples.models.bertstyle import BertStyleEncoder
from train_cifar import collect_lsso_diagnostics, set_lsso_diagnostics_enabled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hand-written BERT-style MHA/LSSO retrieval encoder.")
    parser.add_argument("--dataset", choices=["nfcorpus", "fiqa", "scifact"], default="nfcorpus")
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument("--tokenizer-name", default="bert-base-uncased")
    parser.add_argument(
        "--mixer",
        choices=["mha", "performer", "nystrom", "bimamba", "lsso", "lsso-no-global", "rope-lsso"],
        default="lsso",
    )
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gamma-max", type=float, default=0.3)
    parser.add_argument("--theta-gamma-init", type=float, default=-4.0)
    parser.add_argument("--fixed-mu-gamma", action="store_true")
    parser.add_argument("--no-u-rms-norm", action="store_true")
    parser.add_argument("--pooling", choices=["cls", "mean"], default="mean")
    parser.add_argument("--max-query-len", type=int, default=64)
    parser.add_argument("--max-doc-len", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-pairs", type=int, default=50000)
    parser.add_argument("--max-eval-queries", type=int, default=1000)
    parser.add_argument("--max-corpus-docs", type=int, default=0)
    parser.add_argument("--candidate-negatives", type=int, default=0)
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def freeze_lsso_scales(model: torch.nn.Module) -> None:
    if not hasattr(model, "lsso_layers"):
        return
    for layer in model.lsso_layers():
        layer.theta_mu.requires_grad_(False)
        layer.theta_gamma.requires_grad_(False)


def estimate_encoder_macs(args: argparse.Namespace, seq_len: int) -> dict[str, int]:
    """Analytic forward MAC estimate for one encoder pass at a fixed sequence length."""
    D = args.dim
    H = args.num_heads
    r = args.rank
    hidden = int(D * args.mlp_ratio)
    depth = args.depth

    ffn_macs = 2 * seq_len * D * hidden
    if args.mixer == "mha":
        mixer_macs = 4 * seq_len * D * D + 2 * seq_len * seq_len * D
    elif args.mixer == "performer":
        head_dim = D // H
        features = int(head_dim * np.log(head_dim + 1))
        mixer_macs = 4 * seq_len * D * D + 4 * seq_len * D * features + seq_len * H * features
    elif args.mixer == "nystrom":
        landmarks = min(64, seq_len)
        conv_kernel = 65
        mixer_macs = 4 * seq_len * D * D
        mixer_macs += 2 * seq_len * landmarks * D
        mixer_macs += H * landmarks * landmarks * (D // H)
        mixer_macs += H * landmarks * landmarks * landmarks
        mixer_macs += conv_kernel * seq_len * D
    elif args.mixer == "bimamba":
        # Official Mamba kernels mix projections, convolution, scan, and gating.
        # Keep this conservative placeholder out of paper tables until profiled.
        mixer_macs = 0
    elif args.mixer in {"lsso", "lsso-no-global", "rope-lsso"}:
        mixer_macs = seq_len * D * (H * r + D) + seq_len * D * D
        if args.mixer != "lsso-no-global":
            mixer_macs += H * seq_len * r * r
            mixer_macs += 2 * seq_len * r * D
            mixer_macs += D * r * r
            mixer_macs += H * r * r * r
    else:
        raise ValueError(f"unknown mixer: {args.mixer}")

    total = depth * (mixer_macs + ffn_macs)
    return {
        "mixer_macs": int(depth * mixer_macs),
        "ffn_macs": int(depth * ffn_macs),
        "total_macs": int(total),
        "total_flops": int(2 * total),
    }


def join_title_text(title: str, text: str) -> str:
    title = (title or "").strip()
    text = (text or "").strip()
    if title and text:
        return f"{title}. {text}"
    return title or text


def load_beir_dataset(dataset: str, offline: bool):
    if offline:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from datasets import load_dataset

    dataset_name = f"BeIR/{dataset}"
    qrels_name = f"BeIR/{dataset}-qrels"
    corpus = load_dataset(dataset_name, "corpus", split="corpus")
    queries = load_dataset(dataset_name, "queries", split="queries")
    qrels_train = load_dataset(qrels_name, split="train")
    qrels_test = load_dataset(qrels_name, split="test")
    return corpus, queries, qrels_train, qrels_test


def build_qrels_map(rows) -> dict[str, list[tuple[str, int]]]:
    grouped: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["query-id"])].append((str(row["corpus-id"]), int(row["score"])))
    for qid in grouped:
        grouped[qid].sort(key=lambda x: x[1], reverse=True)
    return grouped


def build_train_pairs(
    queries: dict[str, str],
    corpus: dict[str, str],
    qrels_train: dict[str, list[tuple[str, int]]],
    max_pairs: int,
    seed: int,
) -> list[tuple[str, str]]:
    pairs = []
    for qid, docs in qrels_train.items():
        qtext = queries.get(qid)
        if not qtext:
            continue
        for doc_id, _score in docs:
            dtext = corpus.get(doc_id)
            if dtext:
                pairs.append((qtext, dtext))
    rng = random.Random(seed)
    rng.shuffle(pairs)
    if max_pairs:
        pairs = pairs[:max_pairs]
    return pairs


def build_eval_sets(
    queries: dict[str, str],
    corpus: dict[str, str],
    qrels_test: dict[str, list[tuple[str, int]]],
    max_queries: int,
    max_docs: int,
):
    corpus_ids = list(corpus.keys())
    if max_docs:
        positive_ids = []
        seen = set()
        for docs in qrels_test.values():
            for doc_id, _score in docs:
                if doc_id in corpus and doc_id not in seen:
                    positive_ids.append(doc_id)
                    seen.add(doc_id)
        corpus_ids = positive_ids[:max_docs]
        if len(corpus_ids) < max_docs:
            for doc_id in corpus:
                if doc_id not in seen:
                    corpus_ids.append(doc_id)
                    if len(corpus_ids) >= max_docs:
                        break

    corpus_index = {doc_id: i for i, doc_id in enumerate(corpus_ids)}
    corpus_texts = [corpus[doc_id] for doc_id in corpus_ids]
    eval_queries = []
    positives = []
    for qid, docs in qrels_test.items():
        qtext = queries.get(qid)
        if not qtext:
            continue
        pos = {corpus_index[doc_id] for doc_id, _score in docs if doc_id in corpus_index}
        if not pos:
            continue
        eval_queries.append(qtext)
        positives.append(pos)
        if max_queries and len(eval_queries) >= max_queries:
            break
    return corpus_texts, eval_queries, positives, None


def build_candidate_eval_sets(
    queries: dict[str, str],
    corpus: dict[str, str],
    qrels_test: dict[str, list[tuple[str, int]]],
    max_queries: int,
    negatives: int,
    seed: int,
):
    rng = random.Random(seed)
    all_doc_ids = list(corpus.keys())
    eval_corpus = []
    doc_to_eval_idx: dict[str, int] = {}
    eval_queries = []
    positives = []
    candidate_indices = []

    def add_doc(doc_id: str) -> int:
        if doc_id not in doc_to_eval_idx:
            doc_to_eval_idx[doc_id] = len(eval_corpus)
            eval_corpus.append(corpus[doc_id])
        return doc_to_eval_idx[doc_id]

    for qid, docs in qrels_test.items():
        qtext = queries.get(qid)
        if not qtext:
            continue

        pos_ids = [doc_id for doc_id, _score in docs if doc_id in corpus]
        if not pos_ids:
            continue

        pos_id = pos_ids[0]
        banned = set(pos_ids)
        neg_pool = []
        while len(neg_pool) < negatives and len(neg_pool) + len(banned) < len(all_doc_ids):
            doc_id = rng.choice(all_doc_ids)
            if doc_id in banned:
                continue
            banned.add(doc_id)
            neg_pool.append(doc_id)

        doc_ids = [pos_id] + neg_pool
        rng.shuffle(doc_ids)
        indices = [add_doc(doc_id) for doc_id in doc_ids]
        pos_index = doc_to_eval_idx[pos_id]
        eval_queries.append(qtext)
        positives.append({pos_index})
        candidate_indices.append(indices)

        if max_queries and len(eval_queries) >= max_queries:
            break

    return eval_corpus, eval_queries, positives, candidate_indices


class PairDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[str, str]:
        return self.pairs[index]


def build_collate(tokenizer, max_query_len: int, max_doc_len: int):
    def collate(batch: list[tuple[str, str]]):
        q_text = [x[0] for x in batch]
        d_text = [x[1] for x in batch]
        q = tokenizer(q_text, padding=True, truncation=True, max_length=max_query_len, return_tensors="pt")
        d = tokenizer(d_text, padding=True, truncation=True, max_length=max_doc_len, return_tensors="pt")
        return q, d

    return collate


def encode_batch(model: BertStyleEncoder, input_ids: torch.Tensor, attention_mask: torch.Tensor, pooling: str):
    hidden = model.forward_features(input_ids)
    if pooling == "cls":
        emb = hidden[:, 0]
    else:
        mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
    return F.normalize(emb, dim=-1)


def train_one_epoch(
    model: BertStyleEncoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    use_amp: bool,
) -> dict[str, float]:
    model.train()
    set_lsso_diagnostics_enabled(model, False)
    total_loss = 0.0
    total_acc = 0.0
    total_count = 0

    for q, d in tqdm(loader, desc="train", leave=False):
        q_ids = q["input_ids"].to(device, non_blocking=True)
        q_mask = q["attention_mask"].to(device, non_blocking=True)
        d_ids = d["input_ids"].to(device, non_blocking=True)
        d_mask = d["attention_mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            q_emb = encode_batch(model, q_ids, q_mask, args.pooling)
            d_emb = encode_batch(model, d_ids, d_mask, args.pooling)
            logits = torch.matmul(q_emb, d_emb.transpose(0, 1)) / args.temperature
            targets = torch.arange(logits.shape[0], device=device)
            loss = F.cross_entropy(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss: {loss.item()}")

        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        batch = q_ids.shape[0]
        total_loss += loss.item() * batch
        total_acc += (logits.argmax(dim=1) == targets).float().mean().item() * batch
        total_count += batch

    return {
        "loss": total_loss / total_count,
        "acc": total_acc / total_count,
        "lr": optimizer.param_groups[0]["lr"],
    }


@torch.no_grad()
def encode_texts(
    model: BertStyleEncoder,
    tokenizer,
    texts: list[str],
    max_len: int,
    batch_size: int,
    device: torch.device,
    args: argparse.Namespace,
    use_amp: bool,
) -> torch.Tensor:
    model.eval()
    out = []
    for start in tqdm(range(0, len(texts), batch_size), desc="encode", leave=False):
        batch = texts[start : start + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        ids = enc["input_ids"].to(device, non_blocking=True)
        mask = enc["attention_mask"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out.append(encode_batch(model, ids, mask, args.pooling).float().cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def evaluate(
    model: BertStyleEncoder,
    tokenizer,
    corpus_texts: list[str],
    query_texts: list[str],
    positives: list[set[int]],
    candidate_indices: list[list[int]] | None,
    device: torch.device,
    args: argparse.Namespace,
    use_amp: bool,
) -> dict[str, float]:
    set_lsso_diagnostics_enabled(model, True)
    corpus_emb = encode_texts(
        model, tokenizer, corpus_texts, args.max_doc_len, args.eval_batch_size, device, args, use_amp
    )
    query_emb = encode_texts(
        model, tokenizer, query_texts, args.max_query_len, args.eval_batch_size, device, args, use_amp
    )
    set_lsso_diagnostics_enabled(model, False)

    recall1 = 0.0
    recall10 = 0.0
    mrr10 = 0.0
    corpus_t = corpus_emb.transpose(0, 1)
    for i in tqdm(range(query_emb.shape[0]), desc="retrieval", leave=False):
        if candidate_indices is None:
            scores = torch.matmul(query_emb[i], corpus_t)
            base = 0
            top = torch.topk(scores, k=min(10, scores.shape[0])).indices.tolist()
        else:
            idx = candidate_indices[i]
            scores = torch.matmul(query_emb[i], corpus_emb[idx].transpose(0, 1))
            top = [idx[j] for j in torch.topk(scores, k=min(10, scores.shape[0])).indices.tolist()]
        pos = positives[i]
        recall1 += float(top[0] in pos)
        recall10 += float(any(idx in pos for idx in top))
        for rank, idx in enumerate(top, start=1):
            if idx in pos:
                mrr10 += 1.0 / rank
                break

    n = max(1, len(query_texts))
    metrics = {"recall@1": recall1 / n, "recall@10": recall10 / n, "mrr@10": mrr10 / n}
    metrics.update(collect_lsso_diagnostics(model))
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.offline:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    corpus_ds, queries_ds, qrels_train_ds, qrels_test_ds = load_beir_dataset(args.dataset, args.offline)
    corpus = {str(row["_id"]): join_title_text(row.get("title", ""), row.get("text", "")) for row in corpus_ds}
    queries = {str(row["_id"]): join_title_text(row.get("title", ""), row.get("text", "")) for row in queries_ds}
    qrels_train = build_qrels_map(qrels_train_ds)
    qrels_test = build_qrels_map(qrels_test_ds)
    train_pairs = build_train_pairs(queries, corpus, qrels_train, args.max_train_pairs, args.seed)
    if args.candidate_negatives:
        corpus_texts, eval_queries, positives, candidate_indices = build_candidate_eval_sets(
            queries,
            corpus,
            qrels_test,
            args.max_eval_queries,
            args.candidate_negatives,
            args.seed,
        )
    else:
        corpus_texts, eval_queries, positives, candidate_indices = build_eval_sets(
            queries, corpus, qrels_test, args.max_eval_queries, args.max_corpus_docs
        )

    model = BertStyleEncoder(
        vocab_size=len(tokenizer),
        num_classes=2,
        max_len=max(args.max_query_len, args.max_doc_len),
        dim=args.dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mixer=args.mixer,
        rank=args.rank,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        gamma_max=args.gamma_max,
        theta_gamma_init=args.theta_gamma_init,
        normalize_u=not args.no_u_rms_norm,
        pad_id=tokenizer.pad_token_id or 0,
    )
    if args.fixed_mu_gamma:
        freeze_lsso_scales(model)

    train_loader = DataLoader(
        PairDataset(train_pairs),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=build_collate(tokenizer, args.max_query_len, args.max_doc_len),
    )
    device = torch.device(args.device)
    use_amp = args.amp and device.type == "cuda"
    model.to(device)

    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    query_cost = estimate_encoder_macs(args, args.max_query_len)
    doc_cost = estimate_encoder_macs(args, args.max_doc_len)
    print(
        f"dataset={args.dataset} train_pairs={len(train_pairs)} eval_queries={len(eval_queries)} "
        f"corpus_docs={len(corpus_texts)} vocab={len(tokenizer)} params={params:,} "
        f"trainable={trainable_params:,} "
        f"query_macs={query_cost['total_macs']:,} doc_macs={doc_cost['total_macs']:,} "
        f"device={device} amp={use_amp} fixed_mu_gamma={args.fixed_mu_gamma} "
        f"u_rms_norm={not args.no_u_rms_norm}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, max(0.0, progress))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    run_name = (
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_bertstyle_retr_"
        f"{args.dataset}_{args.mixer}_r{args.rank}_g{args.gamma_max}_tgi{args.theta_gamma_init}_"
        f"d{args.dim}_L{args.depth}_h{args.num_heads}_lend{args.max_doc_len}_s{args.seed}"
        f"{'_fixedscale' if args.fixed_mu_gamma else ''}"
        f"{'_nou' if args.no_u_rms_norm else ''}"
    )
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"{run_name}.jsonl"
    ckpt_path = run_dir / f"{run_name}.pt"

    best_recall10 = 0.0
    with log_path.open("w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "args": vars(args),
                    "params": params,
                    "query_cost": query_cost,
                    "doc_cost": doc_cost,
                },
                sort_keys=True,
            )
            + "\n"
        )
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            print(f"epoch {epoch}/{args.epochs} train", flush=True)
            train_start = time.perf_counter()
            train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, args, use_amp)
            train_seconds = time.perf_counter() - train_start
            print(f"epoch {epoch}/{args.epochs} eval", flush=True)
            eval_start = time.perf_counter()
            eval_metrics = evaluate(
                model,
                tokenizer,
                corpus_texts,
                eval_queries,
                positives,
                candidate_indices,
                device,
                args,
                use_amp,
            )
            eval_seconds = time.perf_counter() - eval_start
            epoch_seconds = time.perf_counter() - epoch_start
            row = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **eval_metrics,
                "train_seconds": train_seconds,
                "eval_seconds": eval_seconds,
                "epoch_seconds": epoch_seconds,
                "train_samples_per_sec": len(train_pairs) / max(1e-9, train_seconds),
            }
            print(json.dumps(row, sort_keys=True), flush=True)
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
            if eval_metrics["recall@10"] > best_recall10:
                best_recall10 = eval_metrics["recall@10"]
                torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, ckpt_path)

    print(f"best_recall@10={best_recall10:.4f}")
    print(f"log={log_path}")
    print(f"checkpoint={ckpt_path}")


if __name__ == "__main__":
    main()
