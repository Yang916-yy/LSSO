from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from examples.models.bert import iter_bert_lsso_layers, replace_bert_self_attention_with_lsso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BERT/LSSO encoder for retrieval on cached BeIR nfcorpus.")
    parser.add_argument("--dataset", choices=["nfcorpus"], default="nfcorpus")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--mixer", choices=["mha", "lsso", "lsso-no-global"], default="lsso")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--gamma-max", type=float, default=0.3)
    parser.add_argument("--theta-gamma-init", type=float, default=-4.0)
    parser.add_argument("--pooling", choices=["cls", "mean"], default="mean")
    parser.add_argument("--max-query-len", type=int, default=64)
    parser.add_argument("--max-doc-len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lsso-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-pairs", type=int, default=50000)
    parser.add_argument("--max-eval-queries", type=int, default=1000)
    parser.add_argument("--max-corpus-docs", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def join_title_text(title: str, text: str) -> str:
    title = (title or "").strip()
    text = (text or "").strip()
    if title and text:
        return f"{title}. {text}"
    return title or text


def load_nfcorpus_cached(offline: bool):
    if offline:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from datasets import load_dataset

    corpus_ds = load_dataset("BeIR/nfcorpus", "corpus", split="corpus")
    queries_ds = load_dataset("BeIR/nfcorpus", "queries", split="queries")
    qrels_train = load_dataset("BeIR/nfcorpus-qrels", split="train")
    qrels_test = load_dataset("BeIR/nfcorpus-qrels", split="test")
    return corpus_ds, queries_ds, qrels_train, qrels_test


def build_qrels_map(qrels_rows) -> dict[str, list[tuple[str, int]]]:
    grouped: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for row in qrels_rows:
        grouped[row["query-id"]].append((row["corpus-id"], int(row["score"])))
    for qid in grouped:
        grouped[qid].sort(key=lambda x: x[1], reverse=True)
    return grouped


def build_train_pairs(
    queries: dict[str, str],
    corpus: dict[str, str],
    qrels_train: dict[str, list[tuple[str, int]]],
    max_train_pairs: int,
    seed: int,
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for qid, docs in qrels_train.items():
        if qid not in queries:
            continue
        qtext = queries[qid]
        for doc_id, _score in docs:
            dtext = corpus.get(doc_id)
            if dtext:
                pairs.append((qtext, dtext))
                break
    rng = random.Random(seed)
    rng.shuffle(pairs)
    if max_train_pairs and max_train_pairs < len(pairs):
        pairs = pairs[:max_train_pairs]
    return pairs


def build_eval_sets(
    queries: dict[str, str],
    corpus: dict[str, str],
    qrels_test: dict[str, list[tuple[str, int]]],
    max_eval_queries: int,
    max_corpus_docs: int,
):
    positive_doc_ids = set()
    for docs in qrels_test.values():
        for doc_id, _score in docs:
            if doc_id in corpus:
                positive_doc_ids.add(doc_id)

    corpus_ids = list(positive_doc_ids)
    if max_corpus_docs and max_corpus_docs < len(corpus_ids):
        corpus_ids = corpus_ids[:max_corpus_docs]
    elif max_corpus_docs and max_corpus_docs > len(corpus_ids):
        for doc_id in corpus:
            if doc_id in positive_doc_ids:
                continue
            corpus_ids.append(doc_id)
            if len(corpus_ids) >= max_corpus_docs:
                break
    elif not max_corpus_docs:
        corpus_ids = list(corpus.keys())

    corpus_texts = [corpus[doc_id] for doc_id in corpus_ids]
    corpus_index = {doc_id: i for i, doc_id in enumerate(corpus_ids)}

    eval_queries = []
    eval_positive = []
    for qid, docs in qrels_test.items():
        qtext = queries.get(qid)
        if not qtext:
            continue
        pos = {corpus_index[doc_id] for doc_id, _score in docs if doc_id in corpus_index}
        if not pos:
            continue
        eval_queries.append(qtext)
        eval_positive.append(pos)
        if max_eval_queries and len(eval_queries) >= max_eval_queries:
            break
    return corpus_texts, eval_queries, eval_positive


class PairTextDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.pairs[idx]


def build_collate_fn(tokenizer, max_query_len: int, max_doc_len: int):
    def collate(batch: list[tuple[str, str]]):
        q_texts = [x[0] for x in batch]
        d_texts = [x[1] for x in batch]
        q = tokenizer(
            q_texts,
            truncation=True,
            padding=True,
            max_length=max_query_len,
            return_tensors="pt",
        )
        d = tokenizer(
            d_texts,
            truncation=True,
            padding=True,
            max_length=max_doc_len,
            return_tensors="pt",
        )
        return q, d

    return collate


def encode_embeddings(model, input_ids, attention_mask, pooling: str) -> torch.Tensor:
    out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    hidden = out.last_hidden_state
    if pooling == "cls":
        emb = hidden[:, 0]
    else:
        mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
    return F.normalize(emb, dim=-1)


def compute_lsso_strength(model) -> dict[str, float]:
    layers = iter_bert_lsso_layers(model)
    if not layers:
        return {}

    ratios = []
    for layer in layers:
        mu = F.softplus(layer.theta_mu.detach()) + layer.eps
        gamma = layer.gamma_max * torch.sigmoid(layer.theta_gamma.detach())
        if layer.no_global:
            gamma = torch.zeros_like(gamma)
        ratios.append((gamma / mu).float())
    v = torch.cat(ratios)
    return {"diag_gamma_over_mu": v.mean().item(), "diag_gamma_over_mu_max": v.max().item()}


def build_optimizer(model, args: argparse.Namespace):
    lsso_param_ids = set()
    for layer in iter_bert_lsso_layers(model):
        for p in layer.parameters():
            lsso_param_ids.add(id(p))

    lsso_params = []
    base_params = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in lsso_param_ids:
            lsso_params.append(p)
        else:
            base_params.append(p)

    groups = [{"params": base_params, "lr": args.lr, "weight_decay": args.weight_decay}]
    if lsso_params:
        groups.append({"params": lsso_params, "lr": args.lsso_lr, "weight_decay": args.weight_decay})
    return torch.optim.AdamW(groups)


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    temperature: float,
    pooling: str,
    use_amp: bool,
):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_count = 0

    for q_batch, d_batch in tqdm(loader, desc="train", leave=False):
        q_ids = q_batch["input_ids"].to(device, non_blocking=True)
        q_mask = q_batch["attention_mask"].to(device, non_blocking=True)
        d_ids = d_batch["input_ids"].to(device, non_blocking=True)
        d_mask = d_batch["attention_mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            q_emb = encode_embeddings(model, q_ids, q_mask, pooling)
            d_emb = encode_embeddings(model, d_ids, d_mask, pooling)
            logits = torch.matmul(q_emb, d_emb.transpose(0, 1)) / temperature
            targets = torch.arange(logits.shape[0], device=device)
            loss = F.cross_entropy(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch = logits.shape[0]
        total_loss += loss.item() * batch
        total_acc += (logits.argmax(dim=-1) == targets).float().mean().item() * batch
        total_count += batch

    return {"loss": total_loss / total_count, "acc": total_acc / total_count}


@torch.no_grad()
def encode_text_list(
    model,
    tokenizer,
    texts: list[str],
    max_len: int,
    batch_size: int,
    device: torch.device,
    pooling: str,
    use_amp: bool,
) -> torch.Tensor:
    all_emb = []
    model.eval()
    for i in tqdm(range(0, len(texts), batch_size), desc="encode", leave=False):
        batch_text = texts[i : i + batch_size]
        enc = tokenizer(
            batch_text,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )
        ids = enc["input_ids"].to(device, non_blocking=True)
        mask = enc["attention_mask"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            emb = encode_embeddings(model, ids, mask, pooling)
        all_emb.append(emb.float().cpu())
    return torch.cat(all_emb, dim=0)


@torch.no_grad()
def evaluate_retrieval(
    model,
    tokenizer,
    corpus_texts: list[str],
    query_texts: list[str],
    positives: list[set[int]],
    args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    corpus_emb = encode_text_list(
        model,
        tokenizer,
        corpus_texts,
        args.max_doc_len,
        args.eval_batch_size,
        device,
        args.pooling,
        use_amp,
    )
    query_emb = encode_text_list(
        model,
        tokenizer,
        query_texts,
        args.max_query_len,
        args.eval_batch_size,
        device,
        args.pooling,
        use_amp,
    )

    recall1 = 0.0
    recall10 = 0.0
    mrr10 = 0.0

    k = min(10, corpus_emb.shape[0])
    for i in tqdm(range(query_emb.shape[0]), desc="retrieval", leave=False):
        scores = torch.matmul(query_emb[i], corpus_emb.transpose(0, 1))
        top_idx = torch.topk(scores, k=k, largest=True).indices.tolist()
        pos = positives[i]
        if top_idx[0] in pos:
            recall1 += 1.0
        if any(idx in pos for idx in top_idx):
            recall10 += 1.0
        rank = 0
        for j, idx in enumerate(top_idx, start=1):
            if idx in pos:
                rank = j
                break
        if rank > 0:
            mrr10 += 1.0 / rank

    n = max(1, query_emb.shape[0])
    return {"recall@1": recall1 / n, "recall@10": recall10 / n, "mrr@10": mrr10 / n}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    from transformers import AutoModel, AutoTokenizer

    tokenizer_name = args.tokenizer_name or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    model = AutoModel.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
    )
    if args.mixer in {"lsso", "lsso-no-global"}:
        replace_bert_self_attention_with_lsso(
            model,
            rank=args.rank,
            gamma_max=args.gamma_max,
            theta_gamma_init=args.theta_gamma_init,
            no_global=args.mixer == "lsso-no-global",
        )

    corpus_ds, queries_ds, qrels_train_ds, qrels_test_ds = load_nfcorpus_cached(args.offline)
    corpus = {row["_id"]: join_title_text(row.get("title", ""), row.get("text", "")) for row in corpus_ds}
    queries = {row["_id"]: join_title_text(row.get("title", ""), row.get("text", "")) for row in queries_ds}
    qrels_train = build_qrels_map(qrels_train_ds)
    qrels_test = build_qrels_map(qrels_test_ds)

    train_pairs = build_train_pairs(queries, corpus, qrels_train, args.max_train_pairs, args.seed)
    corpus_texts, eval_queries, eval_positive = build_eval_sets(
        queries, corpus, qrels_test, args.max_eval_queries, args.max_corpus_docs
    )
    print(
        f"train_pairs={len(train_pairs)} eval_queries={len(eval_queries)} corpus_docs={len(corpus_texts)}",
        flush=True,
    )

    train_set = PairTextDataset(train_pairs)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=build_collate_fn(tokenizer, args.max_query_len, args.max_doc_len),
    )

    device = torch.device(args.device)
    use_amp = args.amp and device.type == "cuda"
    model.to(device)
    optimizer = build_optimizer(model, args)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run_name = (
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_"
        f"hfretr_{args.dataset}_{args.mixer}_r{args.rank}_g{args.gamma_max}_"
        f"lenq{args.max_query_len}_lend{args.max_doc_len}_s{args.seed}"
    )
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"{run_name}.jsonl"
    ckpt_path = run_dir / f"{run_name}.pt"

    best_r10 = 0.0
    with log_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"args": vars(args)}, sort_keys=True) + "\n")
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
                args.temperature,
                args.pooling,
                use_amp,
            )
            eval_metrics = evaluate_retrieval(
                model,
                tokenizer,
                corpus_texts,
                eval_queries,
                eval_positive,
                args,
                device,
                use_amp,
            )
            eval_metrics.update(compute_lsso_strength(model))

            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **eval_metrics}
            print(json.dumps(row, sort_keys=True))
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()

            if eval_metrics["recall@10"] > best_r10:
                best_r10 = eval_metrics["recall@10"]
                torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, ckpt_path)

    print(f"best_recall@10={best_r10:.4f}")
    print(f"log={log_path}")
    print(f"checkpoint={ckpt_path}")


if __name__ == "__main__":
    main()
