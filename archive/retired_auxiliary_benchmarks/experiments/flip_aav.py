"""Archived FLIP AAV fitness-regression experiment."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import BatchSampler, DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.models import ProteinFitnessModel, SequenceMixerEncoder


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYBXZJUO"


class ProteinTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.lookup = {token: index + 2 for index, token in enumerate(AMINO_ACIDS)}
        self.vocab_size = len(self.lookup) + 2

    def encode(self, sequence: str, max_length: int) -> torch.Tensor:
        values = [
            self.lookup.get(token, self.unk_token_id)
            for token in sequence.upper()[:max_length]
        ]
        return torch.tensor(values, dtype=torch.int32)

    def batch(self, sequences, max_length: int):
        return collate_proteins(
            [(self.encode(sequence, max_length), 0.0) for sequence in sequences],
            self.pad_token_id,
        )[:2]


class FitnessDataset(Dataset):
    def __init__(self, rows, sequence_key: str, target_key: str,
                 tokenizer: ProteinTokenizer, max_length: int) -> None:
        self.tokens = [
            tokenizer.encode(str(sequence), max_length) for sequence in rows[sequence_key]
        ]
        self.targets = [float(value) for value in rows[target_key]]
        self.lengths = [len(tokens) for tokens in self.tokens]

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        return self.tokens[index], self.targets[index]


def collate_proteins(rows, pad_token_id: int = 0):
    tokens, targets = zip(*rows)
    ids = pad_sequence(
        [row.long() for row in tokens], batch_first=True, padding_value=pad_token_id
    )
    return ids, ids.ne(pad_token_id), torch.tensor(targets, dtype=torch.float32)


class LengthBucketBatchSampler(BatchSampler):
    """Shuffle batches while keeping similarly sized proteins together."""

    def __init__(self, lengths, batch_size: int, seed: int, drop_last: bool = False):
        self.lengths = list(lengths)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        ordered = sorted(range(len(self.lengths)), key=self.lengths.__getitem__)
        batches = [ordered[start:start + self.batch_size]
                   for start in range(0, len(ordered), self.batch_size)]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches.pop()
        rng = random.Random(self.seed + self.epoch)
        rng.shuffle(batches)
        for batch in batches:
            rng.shuffle(batch)
            yield batch
        self.epoch += 1

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="AI4Protein/FLIP_AAV_two-vs-rest")
    p.add_argument("--mixer", choices=("mha", "lsso", "rrlsso"), default="rrlsso")
    p.add_argument("--output", default="runs/auxiliary/flip-aav-rrlsso-r32")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-eval-samples", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_column(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise KeyError(f"none of {candidates} found in dataset columns {columns}")


def atomic_save(state: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


@torch.inference_mode()
def evaluate(model, loader, mean, std, device):
    from scipy.stats import spearmanr

    model.eval()
    predictions, targets = [], []
    for ids, mask, labels in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            normalized = model(
                ids.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            )
        predictions.extend((normalized.float().cpu() * std + mean).tolist())
        targets.extend(labels.tolist())
    pred = np.asarray(predictions)
    target = np.asarray(targets)
    if np.ptp(target) == 0 or np.ptp(pred) == 0:
        correlation = 0.0
    else:
        correlation = float(spearmanr(target, pred).statistic)
    return {
        "spearman": correlation,
        "mse": float(np.mean((target - pred) ** 2)),
    }


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    from datasets import load_dataset

    splits = load_dataset(args.dataset)
    train_rows = splits["train"]
    valid_rows = splits["validation"] if "validation" in splits else splits["valid"]
    test_rows = splits["test"]
    sequence_key = find_column(
        train_rows.column_names, ("sequence", "aa_seq", "seq", "protein")
    )
    target_key = find_column(train_rows.column_names, ("target", "label", "fitness", "score"))
    if args.max_train_samples:
        train_rows = train_rows.select(range(min(args.max_train_samples, len(train_rows))))
    if args.max_eval_samples:
        valid_rows = valid_rows.select(range(min(args.max_eval_samples, len(valid_rows))))
        test_rows = test_rows.select(range(min(args.max_eval_samples, len(test_rows))))
    train_targets = np.asarray(train_rows[target_key], dtype=np.float32)
    target_mean, target_std = float(train_targets.mean()), float(train_targets.std() + 1e-8)
    tokenizer = ProteinTokenizer()
    datasets = [
        FitnessDataset(rows, sequence_key, target_key, tokenizer, args.max_length)
        for rows in (train_rows, valid_rows, test_rows)
    ]
    encoder = SequenceMixerEncoder(
        tokenizer.vocab_size, max_length=args.max_length, pad_token_id=tokenizer.pad_token_id,
        dim=args.dim, depth=args.depth, num_heads=args.heads, mixer=args.mixer, rank=args.rank,
    )
    model = ProteinFitnessModel(encoder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader_kwargs = dict(
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0, collate_fn=collate_proteins,
    )
    loaders = [
        DataLoader(
            datasets[0],
            batch_sampler=LengthBucketBatchSampler(
                datasets[0].lengths, args.batch_size, args.seed
            ),
            **loader_kwargs,
        ),
        DataLoader(datasets[1], batch_size=args.eval_batch_size, **loader_kwargs),
        DataLoader(datasets[2], batch_size=args.eval_batch_size, **loader_kwargs),
    ]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))
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
        for ids, mask, labels in loaders[0]:
            normalized_targets = (
                labels.to(device, non_blocking=True) - target_mean
            ) / target_std
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                predictions = model(
                    ids.to(device, non_blocking=True), mask.to(device, non_blocking=True)
                )
                loss = F.mse_loss(predictions, normalized_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item()
        scheduler.step()
        validation = evaluate(model, loaders[1], target_mean, target_std, device)
        metrics = dict(epoch=epoch, train_loss=loss_sum / max(1, len(loaders[0])), **validation)
        print(json.dumps(metrics, sort_keys=True), flush=True)
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics, sort_keys=True) + "\n")
        score = validation["spearman"]
        improved = score > best or not (output / "best.pt").exists()
        best = max(best, score)
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(), epoch=epoch, best=best,
                     target_mean=target_mean, target_std=target_std, metrics=metrics)
        atomic_save(state, last)
        if improved:
            atomic_save(state, output / "best.pt")
    best_state = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best_state["model"])
    test = evaluate(model, loaders[2], target_mean, target_std, device)
    (output / "test_metrics.json").write_text(json.dumps(test, indent=2, sort_keys=True))
    print(json.dumps({"test": test}, sort_keys=True))


if __name__ == "__main__":
    main()
