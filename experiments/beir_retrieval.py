"""Train the non-BERT LSSO sequence encoder on BEIR retrieval tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.models import SequenceMixerEncoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("nfcorpus", "fiqa", "scifact"), default="scifact")
    p.add_argument("--mixer", choices=("mha", "lsso", "rrlsso"), default="rrlsso")
    p.add_argument("--tokenizer", default="google-t5/t5-base")
    p.add_argument("--output", default="runs/auxiliary/beir-scifact-rrlsso-r32")
    p.add_argument("--cache-dir", default="data/auxiliary_cache")
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--projection-dim", type=int, default=256)
    p.add_argument("--max-query-length", type=int, default=64)
    p.add_argument("--max-doc-length", type=int, default=384)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-train-pairs", type=int, default=0)
    p.add_argument("--max-eval-queries", type=int, default=0)
    p.add_argument("--max-corpus-docs", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init-checkpoint", default=None)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_beir(name: str):
    from datasets import load_dataset

    data = f"BeIR/{name}"
    qrels = f"BeIR/{name}-qrels"
    corpus_rows = load_dataset(data, "corpus", split="corpus")
    query_rows = load_dataset(data, "queries", split="queries")
    train_rows = load_dataset(qrels, split="train")
    test_rows = load_dataset(qrels, split="test")
    corpus = {
        str(row["_id"]): " ".join(x for x in (row.get("title", ""), row["text"]) if x)
        for row in corpus_rows
    }
    queries = {str(row["_id"]): row["text"] for row in query_rows}
    return corpus, queries, train_rows, test_rows


def qrels_map(rows) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if int(row["score"]) > 0:
            result[str(row["query-id"])].add(str(row["corpus-id"]))
    return result


def make_pairs(corpus, queries, qrels, limit: int, seed: int):
    pairs = [
        (queries[qid], corpus[docid])
        for qid, docs in qrels.items()
        if qid in queries
        for docid in docs
        if docid in corpus
    ]
    random.Random(seed).shuffle(pairs)
    return pairs[:limit] if limit else pairs


def split_train_validation(qrels, seed: int, validation_fraction: float = 0.1):
    query_ids = sorted(qrels)
    random.Random(seed).shuffle(query_ids)
    validation_count = max(1, int(len(query_ids) * validation_fraction))
    validation_ids = set(query_ids[:validation_count])
    train = {qid: docs for qid, docs in qrels.items() if qid not in validation_ids}
    validation = {qid: docs for qid, docs in qrels.items() if qid in validation_ids}
    return train, validation


def token_cache_key(name: str, tokenizer, max_length: int, texts) -> str:
    identity = f"{name}|{tokenizer.name_or_path}|{max_length}|{len(texts)}"
    return hashlib.sha1(identity.encode()).hexdigest()[:16]


def tokenize_cached(name: str, tokenizer, texts, max_length: int, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{token_cache_key(name, tokenizer, max_length, texts)}.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    encoded = tokenizer(
        list(texts), padding=False, truncation=True, max_length=max_length,
        add_special_tokens=True,
    )["input_ids"]
    tokens = [torch.tensor(row, dtype=torch.int32) for row in encoded]
    temporary = path.with_suffix(".pt.tmp")
    torch.save(tokens, temporary)
    os.replace(temporary, path)
    return tokens


def pad_token_sequences(rows, pad_token_id: int):
    ids = pad_sequence(
        [row.long() for row in rows], batch_first=True, padding_value=pad_token_id
    )
    return ids, ids.ne(pad_token_id)


class TokenizedPairDataset(Dataset):
    def __init__(self, queries, documents) -> None:
        if len(queries) != len(documents):
            raise ValueError("query/document token counts must match")
        self.queries, self.documents = queries, documents

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        return self.queries[index], self.documents[index]


def collate_token_pairs(rows, pad_token_id: int):
    queries, documents = zip(*rows)
    return (
        *pad_token_sequences(queries, pad_token_id),
        *pad_token_sequences(documents, pad_token_id),
    )


@torch.inference_mode()
def encode_tokens(model, tokens, pad_token_id, batch_size, device):
    model.eval()
    outputs = []
    for start in range(0, len(tokens), batch_size):
        ids, mask = pad_token_sequences(tokens[start:start + batch_size], pad_token_id)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            vectors = model.encode_normalized(
                ids.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            )
        outputs.append(vectors.float().cpu())
    return torch.cat(outputs) if outputs else torch.empty(0, model.projection.out_features)


@torch.inference_mode()
def chunked_topk(query_vectors, document_vectors, *, k: int, device: torch.device,
                 query_chunk: int = 128, document_chunk: int = 8192):
    if not len(document_vectors):
        raise ValueError("cannot retrieve from an empty corpus")
    k = min(k, len(document_vectors))
    all_indices = []
    if device.type == "cuda":
        document_vectors = document_vectors.pin_memory()
    for query_start in range(0, len(query_vectors), query_chunk):
        query = query_vectors[query_start:query_start + query_chunk].to(
            device, non_blocking=True
        )
        best_scores = torch.full((len(query), k), -torch.inf, device=device)
        best_indices = torch.full((len(query), k), -1, dtype=torch.long, device=device)
        for document_start in range(0, len(document_vectors), document_chunk):
            document = document_vectors[
                document_start:document_start + document_chunk
            ].to(device, non_blocking=True)
            local_scores, local_indices = (query @ document.T).topk(
                min(k, len(document)), dim=1
            )
            local_indices += document_start
            scores = torch.cat((best_scores, local_scores), dim=1)
            indices = torch.cat((best_indices, local_indices), dim=1)
            best_scores, selected = scores.topk(k, dim=1)
            best_indices = indices.gather(1, selected)
        all_indices.append(best_indices.cpu())
    return torch.cat(all_indices)


@torch.inference_mode()
def evaluate(model, corpus, queries, corpus_tokens, query_tokens, qrels, args, device):
    query_ids = [qid for qid in qrels if qid in queries and qrels[qid]]
    if args.max_eval_queries:
        query_ids = query_ids[: args.max_eval_queries]
    positive_ids = {docid for qid in query_ids for docid in qrels[qid] if docid in corpus}
    doc_ids = list(corpus)
    if args.max_corpus_docs:
        chosen = sorted(positive_ids)
        chosen.extend(docid for docid in doc_ids if docid not in positive_ids)
        doc_ids = chosen[: max(args.max_corpus_docs, len(positive_ids))]
    doc_index = {docid: index for index, docid in enumerate(doc_ids)}
    query_ids = [qid for qid in query_ids if any(d in doc_index for d in qrels[qid])]
    q = encode_tokens(
        model, [query_tokens[x] for x in query_ids], args.pad_token_id,
        args.eval_batch_size, device,
    )
    d = encode_tokens(
        model, [corpus_tokens[x] for x in doc_ids], args.pad_token_id,
        args.eval_batch_size, device,
    )
    recalls = {1: 0.0, 5: 0.0, 10: 0.0, 100: 0.0}
    mrr10 = ndcg10 = 0.0
    top = chunked_topk(q, d, k=100, device=device)
    for start in range(0, len(query_ids), 128):
        for row, qid in enumerate(query_ids[start:start + 128]):
            positives = {doc_index[x] for x in qrels[qid] if x in doc_index}
            ranked = top[start + row].tolist()
            for k in recalls:
                recalls[k] += float(any(index in positives for index in ranked[:k]))
            hits = [rank + 1 for rank, index in enumerate(ranked[:10]) if index in positives]
            if hits:
                mrr10 += 1.0 / hits[0]
                dcg = sum(1.0 / math.log2(rank + 1) for rank in hits)
                ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(10, len(positives))))
                ndcg10 += dcg / ideal
    count = max(1, len(query_ids))
    result = {f"recall@{k}": value / count for k, value in recalls.items()}
    result.update({"mrr@10": mrr10 / count, "ndcg@10": ndcg10 / count})
    return result


def atomic_save(state: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    args.pad_token_id = tokenizer.pad_token_id
    corpus, queries, train_rows, test_rows = load_beir(args.dataset)
    all_train_qrels, test_qrels = qrels_map(train_rows), qrels_map(test_rows)
    train_qrels, validation_qrels = split_train_validation(all_train_qrels, args.seed)
    pairs = make_pairs(corpus, queries, train_qrels, args.max_train_pairs, args.seed)
    cache_dir = Path(args.cache_dir)
    corpus_ids = list(corpus)
    query_ids = list(queries)
    corpus_token_rows = tokenize_cached(
        f"beir-{args.dataset}-corpus", tokenizer, [corpus[x] for x in corpus_ids],
        args.max_doc_length, cache_dir,
    )
    query_token_rows = tokenize_cached(
        f"beir-{args.dataset}-queries", tokenizer, [queries[x] for x in query_ids],
        args.max_query_length, cache_dir,
    )
    corpus_tokens = dict(zip(corpus_ids, corpus_token_rows))
    query_tokens = dict(zip(query_ids, query_token_rows))
    pair_query_tokens = tokenize_cached(
        f"beir-{args.dataset}-train-q-s{args.seed}-n{len(pairs)}", tokenizer,
        [query for query, _ in pairs], args.max_query_length, cache_dir,
    )
    pair_document_tokens = tokenize_cached(
        f"beir-{args.dataset}-train-d-s{args.seed}-n{len(pairs)}", tokenizer,
        [document for _, document in pairs], args.max_doc_length, cache_dir,
    )
    model = SequenceMixerEncoder(
        len(tokenizer), max_length=max(args.max_query_length, args.max_doc_length),
        pad_token_id=tokenizer.pad_token_id, dim=args.dim, depth=args.depth,
        num_heads=args.heads, mixer=args.mixer, rank=args.rank,
        projection_dim=args.projection_dim,
    ).to(device)
    if args.init_checkpoint:
        state = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("model", state))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))
    if args.eval_only:
        test = evaluate(
            model, corpus, queries, corpus_tokens, query_tokens, test_qrels, args, device
        )
        (output / "test_metrics.json").write_text(
            json.dumps(test, indent=2, sort_keys=True)
        )
        print(json.dumps({"test": test}, sort_keys=True))
        return
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    loader = DataLoader(
        TokenizedPairDataset(pair_query_tokens, pair_document_tokens),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
        collate_fn=partial(collate_token_pairs, pad_token_id=tokenizer.pad_token_id),
    )
    total_steps = max(1, len(loader) * args.epochs)
    warmup = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / max(1, warmup),
                                    max(0.0, (total_steps - step) / max(1, total_steps - warmup)))
    )
    start_epoch, best = 0, -1.0
    last = output / "last.pt"
    if args.resume and last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best = state["epoch"] + 1, state["best"]
    for epoch in range(start_epoch, args.epochs):
        model.train()
        loss_sum = 0.0
        for query_ids_batch, query_mask, doc_ids_batch, doc_mask in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                q = model.encode_normalized(
                    query_ids_batch.to(device, non_blocking=True),
                    query_mask.to(device, non_blocking=True),
                )
                d = model.encode_normalized(
                    doc_ids_batch.to(device, non_blocking=True),
                    doc_mask.to(device, non_blocking=True),
                )
                logits = q @ d.T / args.temperature
                labels = torch.arange(logits.shape[0], device=device)
                loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item()
        metrics = evaluate(
            model, corpus, queries, corpus_tokens, query_tokens,
            validation_qrels, args, device,
        )
        metrics.update(epoch=epoch, train_loss=loss_sum / max(1, len(loader)))
        print(json.dumps(metrics, sort_keys=True), flush=True)
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics, sort_keys=True) + "\n")
        score = metrics["ndcg@10"]
        improved = score > best
        best = max(best, score)
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(), epoch=epoch, best=best, metrics=metrics)
        atomic_save(state, last)
        if improved:
            atomic_save(state, output / "best.pt")
    best_state = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best_state["model"])
    test = evaluate(
        model, corpus, queries, corpus_tokens, query_tokens, test_qrels, args, device
    )
    (output / "test_metrics.json").write_text(json.dumps(test, indent=2, sort_keys=True))
    print(json.dumps({"test": test}, sort_keys=True))


if __name__ == "__main__":
    main()
