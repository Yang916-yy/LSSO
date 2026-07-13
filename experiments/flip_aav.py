"""FLIP AAV 2-vs-rest fitness regression with the shared sequence mixer."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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

    def batch(self, sequences, max_length: int):
        length = min(max(len(sequence) for sequence in sequences), max_length)
        ids = torch.zeros(len(sequences), length, dtype=torch.long)
        mask = torch.zeros(len(sequences), length, dtype=torch.bool)
        for row, sequence in enumerate(sequences):
            values = [self.lookup.get(token, self.unk_token_id) for token in sequence.upper()[:length]]
            ids[row, :len(values)] = torch.tensor(values)
            mask[row, :len(values)] = True
        return ids, mask


class FitnessDataset(Dataset):
    def __init__(self, rows, sequence_key: str, target_key: str) -> None:
        self.rows, self.sequence_key, self.target_key = rows, sequence_key, target_key

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return str(row[self.sequence_key]), float(row[self.target_key])


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
def evaluate(model, loader, tokenizer, max_length, mean, std, device):
    from scipy.stats import spearmanr

    model.eval()
    predictions, targets = [], []
    for sequences, labels in loader:
        ids, mask = tokenizer.batch(sequences, max_length)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            normalized = model(ids.to(device), mask.to(device))
        predictions.extend((normalized.float().cpu() * std + mean).tolist())
        targets.extend(labels.tolist())
    pred = np.asarray(predictions)
    target = np.asarray(targets)
    correlation = float(spearmanr(target, pred).statistic)
    if not np.isfinite(correlation):
        correlation = 0.0
    return {
        "spearman": correlation,
        "mse": float(np.mean((target - pred) ** 2)),
    }


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
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
    datasets = [FitnessDataset(rows, sequence_key, target_key) for rows in (train_rows, valid_rows, test_rows)]
    loaders = [
        DataLoader(data, batch_size=batch, shuffle=shuffle, num_workers=args.workers)
        for data, batch, shuffle in zip(
            datasets, (args.batch_size, args.eval_batch_size, args.eval_batch_size), (True, False, False)
        )
    ]
    encoder = SequenceMixerEncoder(
        tokenizer.vocab_size, max_length=args.max_length, pad_token_id=tokenizer.pad_token_id,
        dim=args.dim, depth=args.depth, num_heads=args.heads, mixer=args.mixer, rank=args.rank,
    )
    model = ProteinFitnessModel(encoder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
        for sequences, labels in loaders[0]:
            ids, mask = tokenizer.batch(sequences, args.max_length)
            normalized_targets = (labels.float().to(device) - target_mean) / target_std
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                predictions = model(ids.to(device), mask.to(device))
                loss = F.mse_loss(predictions, normalized_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item()
        scheduler.step()
        validation = evaluate(model, loaders[1], tokenizer, args.max_length,
                              target_mean, target_std, device)
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
    test = evaluate(model, loaders[2], tokenizer, args.max_length,
                    target_mean, target_std, device)
    (output / "test_metrics.json").write_text(json.dumps(test, indent=2, sort_keys=True))
    print(json.dumps({"test": test}, sort_keys=True))


if __name__ == "__main__":
    main()
