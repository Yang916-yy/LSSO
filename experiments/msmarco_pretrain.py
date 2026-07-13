"""Stream MS MARCO hard triplets to pretrain the auxiliary dual encoder."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.models import SequenceMixerEncoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        default="sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1",
    )
    p.add_argument("--subset", default="triplet-hard")
    p.add_argument("--tokenizer", default="google-t5/t5-base")
    p.add_argument("--cache-dir", default="data/auxiliary_cache/huggingface")
    p.add_argument("--output", default="runs/auxiliary/msmarco-rrlsso-r32")
    p.add_argument("--mixer", choices=("mha", "lsso", "rrlsso"), default="rrlsso")
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--projection-dim", type=int, default=256)
    p.add_argument("--max-query-length", type=int, default=32)
    p.add_argument("--max-doc-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=10_000)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--shuffle-buffer", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_triplets(rows, tokenizer, query_length: int, doc_length: int):
    query = tokenizer(
        [row["query"] for row in rows], padding=True, truncation=True,
        max_length=query_length, return_tensors="pt",
    )
    positive = tokenizer(
        [row["positive"] for row in rows], padding=True, truncation=True,
        max_length=doc_length, return_tensors="pt",
    )
    negative = tokenizer(
        [row["negative"] for row in rows], padding=True, truncation=True,
        max_length=doc_length, return_tensors="pt",
    )
    return query, positive, negative


def cosine_lr(step: int, total: int, warmup: int, peak: float, floor: float) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))


def atomic_save(state: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("MS MARCO scaling pretraining requires CUDA")
    seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, cache_dir=args.cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    stream = load_dataset(
        args.dataset, args.subset, split="train", streaming=True,
        cache_dir=args.cache_dir,
    ).shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    loader = DataLoader(
        stream, batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
        collate_fn=partial(
            collate_triplets, tokenizer=tokenizer,
            query_length=args.max_query_length, doc_length=args.max_doc_length,
        ),
    )
    model = SequenceMixerEncoder(
        len(tokenizer), max_length=max(args.max_query_length, args.max_doc_length),
        pad_token_id=tokenizer.pad_token_id, dim=args.dim, depth=args.depth,
        num_heads=args.heads, mixer=args.mixer, rank=args.rank,
        projection_dim=args.projection_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    config = {**vars(args), "parameters": parameters,
              "effective_batch": args.batch_size * args.grad_accum}
    (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    start_step = 0
    last = output / "last.pt"
    if args.resume and last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        print(f"resumed from update {start_step}", flush=True)
    iterator = iter(loader)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    samples_since_log = 0
    for update in range(start_step, args.max_steps):
        lr = cosine_lr(update, args.max_steps, args.warmup_steps, args.lr, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(args.grad_accum):
            (query, positive, negative), iterator = next_batch(iterator, loader)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                q = model.encode_normalized(
                    query["input_ids"].to(device, non_blocking=True),
                    query["attention_mask"].to(device, non_blocking=True),
                )
                p = model.encode_normalized(
                    positive["input_ids"].to(device, non_blocking=True),
                    positive["attention_mask"].to(device, non_blocking=True),
                )
                n = model.encode_normalized(
                    negative["input_ids"].to(device, non_blocking=True),
                    negative["attention_mask"].to(device, non_blocking=True),
                )
                labels = torch.arange(len(q), device=device)
                candidates = torch.cat((p, n), dim=0)
                query_loss = F.cross_entropy(q @ candidates.T / args.temperature, labels)
                positive_loss = F.cross_entropy(p @ q.T / args.temperature, labels)
                loss = (query_loss + positive_loss) / (2 * args.grad_accum)
            loss.backward()
            loss_sum += loss.item()
            samples_since_log += len(q)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step = update + 1
        if step % 10 == 0 or step == args.max_steps:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            metrics = {
                "step": step, "loss": loss_sum,
                "lr": lr, "samples_per_second": samples_since_log / elapsed,
                "peak_gb": torch.cuda.max_memory_allocated() / 1024**3,
            }
            print(json.dumps(metrics, sort_keys=True), flush=True)
            with (output / "metrics.jsonl").open("a") as stream_out:
                stream_out.write(json.dumps(metrics, sort_keys=True) + "\n")
            started, samples_since_log = time.perf_counter(), 0
        if step % args.save_steps == 0 or step == args.max_steps:
            atomic_save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "step": step, "config": config},
                last,
            )
    del iterator, loader, stream
    gc.collect()
    # fsspec/datasets may retain an async streaming callback on Python 3.12;
    # all state is durable here, so bypass interpreter-finalizer races.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
