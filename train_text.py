from __future__ import annotations

import argparse
import csv
import json
import random
import re
import tarfile
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from examples.models.text import TextEncoder
from train_cifar import collect_lsso_diagnostics, set_lsso_diagnostics_enabled


PAD = "<pad>"
UNK = "<unk>"
CLS = "<cls>"
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
AG_NEWS_URLS = {
    "train": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
    "test": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",
}
IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
YAHOO_DATASET = "community-datasets/yahoo_answers_topics"
YAHOO_CONFIG = "yahoo_answers_topics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MHA/LSSO text classifier.")
    parser.add_argument("--dataset", choices=["ag_news", "imdb", "yahoo_answers"], default="ag_news")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument("--mixer", choices=["mha", "lsso", "lsso-no-global"], default="lsso")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--max-vocab", type=int, default=30000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gamma-max", type=float, default=0.3)
    parser.add_argument("--theta-gamma-init", type=float, default=-4.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-eval-examples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def download_ag_news(data_dir: Path) -> tuple[Path, Path]:
    data_dir = data_dir / "ag_news"
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, url in AG_NEWS_URLS.items():
        path = data_dir / f"{split}.csv"
        if not path.exists():
            urllib.request.urlretrieve(url, path)
        paths[split] = path
    return paths["train"], paths["test"]


def download_imdb(data_dir: Path) -> Path:
    data_dir = data_dir / "imdb"
    data_dir.mkdir(parents=True, exist_ok=True)
    root = data_dir / "aclImdb"
    if root.exists():
        return root

    archive = data_dir / "aclImdb_v1.tar.gz"
    if not archive.exists():
        urllib.request.urlretrieve(IMDB_URL, archive)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(data_dir)
    return root


def read_ag_news(path: Path) -> list[tuple[str, int]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for label, title, desc in csv.reader(f):
            rows.append((f"{title} {desc}", int(label) - 1))
    return rows


def read_imdb(root: Path, split: str) -> list[tuple[str, int]]:
    cache_path = root / f"{split}.jsonl"
    if cache_path.exists():
        rows = []
        with cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                rows.append((row["text"], row["label"]))
        return rows

    rows = []
    for label_name, label in [("neg", 0), ("pos", 1)]:
        folder = root / split / label_name
        for path in sorted(folder.glob("*.txt")):
            rows.append((path.read_text(encoding="utf-8", errors="replace"), label))

    with cache_path.open("w", encoding="utf-8") as f:
        for text, label in rows:
            f.write(json.dumps({"text": text, "label": label}, ensure_ascii=False) + "\n")
    return rows


def read_yahoo_answers(data_dir: Path, split: str) -> list[tuple[str, int]]:
    from datasets import load_dataset

    cache_dir = data_dir / "yahoo_answers"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{split}.jsonl"
    if cache_path.exists():
        rows = []
        with cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                rows.append((row["text"], row["label"]))
        return rows

    dataset = load_dataset(
        YAHOO_DATASET,
        YAHOO_CONFIG,
        split=split,
        cache_dir=str(cache_dir / "hf_cache"),
    )
    rows = []
    with cache_path.open("w", encoding="utf-8") as f:
        for row in dataset:
            text = " ".join(
                part
                for part in [
                    row.get("question_title", ""),
                    row.get("question_content", ""),
                    row.get("best_answer", ""),
                ]
                if part
            )
            label = int(row["topic"])
            rows.append((text, label))
            f.write(json.dumps({"text": text, "label": label}, ensure_ascii=False) + "\n")
    return rows


def build_vocab(rows: list[tuple[str, int]], max_vocab: int, min_freq: int) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for text, _ in rows:
        counter.update(tokenize(text))

    vocab = {PAD: 0, UNK: 1, CLS: 2}
    for token, count in counter.most_common(max_vocab - len(vocab)):
        if count < min_freq:
            break
        vocab[token] = len(vocab)
    return vocab


def build_or_load_vocab(
    rows: list[tuple[str, int]],
    cache_dir: Path,
    max_vocab: int,
    min_freq: int,
) -> dict[str, int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"vocab_max{max_vocab}_min{min_freq}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    vocab = build_vocab(rows, max_vocab, min_freq)
    cache_path.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    return vocab


def balanced_limit_rows(
    rows: list[tuple[str, int]],
    max_examples: int,
    num_classes: int,
) -> list[tuple[str, int]]:
    if not max_examples or max_examples >= len(rows):
        return rows

    buckets: list[list[tuple[str, int]]] = [[] for _ in range(num_classes)]
    for text, label in rows:
        if 0 <= label < num_classes:
            buckets[label].append((text, label))

    per_class = max_examples // num_classes
    remainder = max_examples % num_classes
    selected: list[tuple[str, int]] = []
    for label, bucket in enumerate(buckets):
        random.shuffle(bucket)
        take = per_class + (1 if label < remainder else 0)
        selected.extend(bucket[:take])
    random.shuffle(selected)
    return selected


class TextDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        rows: list[tuple[str, int]],
        vocab: dict[str, int],
        max_len: int,
        cache_path: Path | None = None,
    ) -> None:
        if cache_path is not None and cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu")
            self.input_ids = cached["input_ids"]
            self.labels = cached["labels"]
            return

        input_ids = []
        labels = []
        for text, label in rows:
            input_ids.append(encode_text(text, vocab, max_len))
            labels.append(label)
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"input_ids": self.input_ids, "labels": self.labels}, cache_path)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[index], self.labels[index]


def encode_text(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    ids = [vocab[CLS]]
    ids.extend(vocab.get(token, vocab[UNK]) for token in tokenize(text)[: max_len - 1])
    if len(ids) < max_len:
        ids.extend([vocab[PAD]] * (max_len - len(ids)))
    return ids


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, int, int]:
    data_root = Path(args.data_dir)
    if args.dataset == "ag_news":
        train_path, test_path = download_ag_news(data_root)
        train_rows = read_ag_news(train_path)
        test_rows = read_ag_news(test_path)
        num_classes = 4
        cache_dir = data_root / "ag_news"
    elif args.dataset == "imdb":
        root = download_imdb(data_root)
        train_rows = read_imdb(root, "train")
        test_rows = read_imdb(root, "test")
        num_classes = 2
        cache_dir = data_root / "imdb"
    elif args.dataset == "yahoo_answers":
        train_rows = read_yahoo_answers(data_root, "train")
        test_rows = read_yahoo_answers(data_root, "test")
        num_classes = 10
        cache_dir = data_root / "yahoo_answers"
    else:
        raise ValueError(f"unknown dataset: {args.dataset}")

    max_train_examples = getattr(args, "max_train_examples", 0)
    max_eval_examples = getattr(args, "max_eval_examples", 0)
    train_rows = balanced_limit_rows(train_rows, max_train_examples, num_classes)
    test_rows = balanced_limit_rows(test_rows, max_eval_examples, num_classes)

    vocab = build_or_load_vocab(train_rows, cache_dir, args.max_vocab, args.min_freq)
    encoded_tag = f"encoded_max{args.max_vocab}_min{args.min_freq}_len{args.max_len}"

    train_loader = DataLoader(
        TextDataset(train_rows, vocab, args.max_len, cache_dir / f"{encoded_tag}_train.pt"),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        TextDataset(test_rows, vocab, args.max_len, cache_dir / f"{encoded_tag}_test.pt"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader, len(vocab), num_classes


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == target).float().mean().item()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    max_batches: int = 0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    set_lsso_diagnostics_enabled(model, not training)
    total_loss = 0.0
    total_acc = 0.0
    total_count = 0
    desc = "train" if training else "eval"

    try:
        for step, (input_ids, target) in enumerate(tqdm(loader, desc=desc, leave=False), start=1):
            input_ids = input_ids.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with torch.set_grad_enabled(training):
                if training:
                    optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(input_ids)
                    loss = criterion(logits, target)
                if training:
                    assert scaler is not None
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            batch = input_ids.shape[0]
            total_loss += loss.item() * batch
            total_acc += accuracy(logits.detach(), target) * batch
            total_count += batch
            if max_batches and step >= max_batches:
                break
    finally:
        set_lsso_diagnostics_enabled(model, False)

    metrics = {"loss": total_loss / total_count, "acc": total_acc / total_count}
    if not training:
        metrics.update(collect_lsso_diagnostics(model))
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    use_amp = args.amp and device.type == "cuda"

    train_loader, test_loader, vocab_size, num_classes = build_loaders(args)
    model = TextEncoder(
        vocab_size=vocab_size,
        num_classes=num_classes,
        max_len=args.max_len,
        dim=args.dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mixer=args.mixer,
        rank=args.rank,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        gamma_max=args.gamma_max,
        theta_gamma_init=args.theta_gamma_init,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run_name = (
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_"
        f"{args.dataset}_{args.mixer}_r{args.rank}_g{args.gamma_max}_tgi{args.theta_gamma_init}_"
        f"d{args.dim}_L{args.depth}_h{args.num_heads}_len{args.max_len}_s{args.seed}"
    )
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"{run_name}.jsonl"
    ckpt_path = run_dir / f"{run_name}.pt"

    best_acc = 0.0
    with log_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"args": vars(args), "vocab_size": vocab_size}, sort_keys=True) + "\n")
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                use_amp,
                optimizer=optimizer,
                scaler=scaler,
                max_batches=args.max_train_batches,
            )
            with torch.no_grad():
                eval_metrics = run_epoch(
                    model,
                    test_loader,
                    criterion,
                    device,
                    use_amp,
                    max_batches=args.max_eval_batches,
                )
            row = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"eval_{k}": v for k, v in eval_metrics.items()},
            }
            print(json.dumps(row, sort_keys=True))
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()

            if eval_metrics["acc"] > best_acc:
                best_acc = eval_metrics["acc"]
                torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, ckpt_path)

    print(f"best_acc={best_acc:.4f}")
    print(f"log={log_path}")
    print(f"checkpoint={ckpt_path}")


if __name__ == "__main__":
    main()
