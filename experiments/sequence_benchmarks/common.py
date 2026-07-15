from __future__ import annotations

import json
import math
import os
import random
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler


@dataclass
class TrainingConfig:
    output: str
    epochs: int = 40
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.0
    grad_clip: float = 1.0
    label_smoothing: float = 0.0
    patience: int = 10
    seed: int = 0
    resume: bool = True
    max_train_batches: int = 0
    max_eval_batches: int = 0
    max_parameters: int = 0


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stratified_split_indices(
    labels: Sequence[int], validation_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Split training indices without consuming the benchmark test set."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    groups: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        groups[int(label)].append(index)
    rng = random.Random(seed)
    train, validation = [], []
    for indices in groups.values():
        rng.shuffle(indices)
        if len(indices) <= 1:
            train.extend(indices)
            continue
        count = min(len(indices) - 1, max(1, round(len(indices) * validation_fraction)))
        validation.extend(indices[:count])
        train.extend(indices[count:])
    if not validation:
        raise ValueError("cannot form a validation split from singleton classes")
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def stratified_subset_indices(
    labels: Sequence[int], count: int, seed: int
) -> list[int]:
    """Choose an exact-size, deterministic subset without dataset-order bias."""
    size = len(labels)
    if count <= 0 or count >= size:
        return list(range(size))
    groups: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        groups[int(label)].append(index)
    rng = random.Random(seed)
    for indices in groups.values():
        rng.shuffle(indices)

    quotas = {label: count * len(indices) / size for label, indices in groups.items()}
    minimum = 1 if count >= len(groups) else 0
    allocated = {
        label: min(len(groups[label]), max(minimum, int(quota)))
        for label, quota in quotas.items()
    }
    while sum(allocated.values()) > count:
        candidates = [label for label, value in allocated.items() if value > minimum]
        label = min(candidates, key=lambda item: (quotas[item] - allocated[item], item))
        allocated[label] -= 1
    while sum(allocated.values()) < count:
        candidates = [
            label for label, value in allocated.items() if value < len(groups[label])
        ]
        label = max(candidates, key=lambda item: (quotas[item] - allocated[item], -item))
        allocated[label] += 1

    selected = [
        index
        for label, indices in groups.items()
        for index in indices[: allocated[label]]
    ]
    rng.shuffle(selected)
    return selected


class IndexDataset(Dataset):
    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = list(indices)
        source_lengths = getattr(dataset, "lengths", None)
        self.lengths = (
            [int(source_lengths[index]) for index in self.indices]
            if source_lengths is not None
            else None
        )
        source_labels = getattr(dataset, "labels", None)
        self.labels = (
            [int(source_labels[index]) for index in self.indices]
            if source_labels is not None
            else None
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]


class LengthBucketBatchSampler(BatchSampler):
    """Shuffle length-sorted mega-buckets to reduce padding without fixed order."""

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        seed: int,
        *,
        bucket_multiplier: int = 20,
        drop_last: bool = False,
    ) -> None:
        self.lengths = [int(length) for length in lengths]
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.bucket_size = max(self.batch_size, self.batch_size * bucket_multiplier)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        rng.shuffle(indices)
        batches: list[list[int]] = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            bucket.sort(key=self.lengths.__getitem__)
            batches.extend(
                bucket[offset : offset + self.batch_size]
                for offset in range(0, len(bucket), self.batch_size)
            )
        if self.drop_last:
            batches = [batch for batch in batches if len(batch) == self.batch_size]
        rng.shuffle(batches)
        self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return math.ceil(len(self.lengths) / self.batch_size)

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.epoch = int(state.get("epoch", 0))


class FixedOrderSampler(Sampler[int]):
    def __init__(self, size: int) -> None:
        self.size = int(size)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.size))

    def __len__(self) -> int:
        return self.size


def collate_tokens(rows: Sequence[tuple[torch.Tensor, int]], pad_token_id: int = 0):
    sequences, labels = zip(*rows)
    inputs = pad_sequence(
        [sequence.long() for sequence in sequences],
        batch_first=True,
        padding_value=pad_token_id,
    )
    return {
        "inputs": inputs,
        "mask": inputs.ne(pad_token_id),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def collate_token_pairs(
    rows: Sequence[tuple[torch.Tensor, torch.Tensor, int]], pad_token_id: int = 0
):
    first, second, labels = zip(*rows)
    first_ids = pad_sequence(
        [tokens.long() for tokens in first], batch_first=True, padding_value=pad_token_id
    )
    second_ids = pad_sequence(
        [tokens.long() for tokens in second], batch_first=True, padding_value=pad_token_id
    )
    return {
        "first": first_ids,
        "first_mask": first_ids.ne(pad_token_id),
        "second": second_ids,
        "second_mask": second_ids.ne(pad_token_id),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def collate_values(rows: Sequence[tuple[torch.Tensor, int]]):
    values, labels = zip(*rows)
    lengths = torch.tensor([row.shape[0] for row in values], dtype=torch.long)
    inputs = pad_sequence([row.float() for row in values], batch_first=True)
    positions = torch.arange(inputs.shape[1]).unsqueeze(0)
    return {
        "inputs": inputs,
        "mask": positions < lengths.unsqueeze(1),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    workers: int,
    device: torch.device,
    collate_fn,
    train: bool,
    seed: int,
) -> DataLoader:
    kwargs = dict(
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        collate_fn=collate_fn,
    )
    lengths = getattr(dataset, "lengths", None)
    if train and lengths is not None:
        return DataLoader(
            dataset,
            batch_sampler=LengthBucketBatchSampler(lengths, batch_size, seed),
            **kwargs,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        generator=torch.Generator().manual_seed(seed) if train else None,
        **kwargs,
    )


def atomic_save(state: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _loader_state(loader: DataLoader) -> dict:
    state = {}
    sampler = loader.batch_sampler
    if hasattr(sampler, "state_dict"):
        state["batch_sampler"] = sampler.state_dict()
    if loader.generator is not None:
        state["generator"] = loader.generator.get_state()
    return state


def _restore_loader_state(loader: DataLoader, state: dict | None) -> None:
    if not state:
        return
    sampler = loader.batch_sampler
    if "batch_sampler" in state and hasattr(sampler, "load_state_dict"):
        sampler.load_state_dict(state["batch_sampler"])
    if "generator" in state and loader.generator is not None:
        loader.generator.set_state(state["generator"])


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _forward(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    if "first" in batch:
        return model(
            batch["first"], batch["first_mask"], batch["second"], batch["second_mask"]
        )
    return model(batch["inputs"], batch["mask"])


def _classification_metrics(
    predictions: Iterable[int], targets: Iterable[int], num_classes: int
) -> dict[str, float]:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        confusion[int(target), int(prediction)] += 1
    total = int(confusion.sum())
    accuracy = float(np.trace(confusion) / max(total, 1))
    f1_values = []
    for label in range(num_classes):
        tp = int(confusion[label, label])
        fp = int(confusion[:, label].sum() - tp)
        fn = int(confusion[label, :].sum() - tp)
        denominator = 2 * tp + fp + fn
        f1_values.append(0.0 if denominator == 0 else 2 * tp / denominator)
    true_totals = confusion.sum(axis=1, dtype=np.float64)
    predicted_totals = confusion.sum(axis=0, dtype=np.float64)
    correct = float(np.trace(confusion))
    covariance = correct * total - float(np.dot(true_totals, predicted_totals))
    denominator = math.sqrt(
        max(0.0, total * total - float(np.dot(true_totals, true_totals)))
        * max(0.0, total * total - float(np.dot(predicted_totals, predicted_totals)))
    )
    matthews_correlation = 0.0 if denominator == 0.0 else covariance / denominator
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)),
        "matthews_correlation": float(matthews_correlation),
    }


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    max_batches: int = 0,
) -> dict[str, float]:
    model.eval()
    predictions, targets = [], []
    loss_sum, examples = 0.0, 0
    for index, raw_batch in enumerate(loader):
        if max_batches and index >= max_batches:
            break
        batch = _move_batch(raw_batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = _forward(model, batch)
            loss = F.cross_entropy(logits, batch["labels"])
        batch_size = batch["labels"].numel()
        loss_sum += float(loss) * batch_size
        examples += batch_size
        predictions.extend(logits.argmax(dim=-1).cpu().tolist())
        targets.extend(batch["labels"].cpu().tolist())
    metrics = _classification_metrics(predictions, targets, num_classes)
    metrics["loss"] = loss_sum / max(examples, 1)
    metrics["examples"] = float(examples)
    return metrics


def _scheduler_lambda(
    step: int, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.0
) -> float:
    if warmup_steps and step < warmup_steps:
        return max(1e-8, (step + 1) / warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def train_classifier(
    model: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader | None,
    *,
    num_classes: int,
    config: TrainingConfig,
    metadata: dict,
) -> dict[str, float]:
    seed_all(config.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    named_parameters = list(model.named_parameters())
    parameter_breakdown = {
        "total": sum(parameter.numel() for _, parameter in named_parameters),
        "position": sum(
            parameter.numel() for name, parameter in named_parameters
            if "position_embedding" in name or "position_projection" in name
        ),
        "mixer": sum(
            parameter.numel() for name, parameter in named_parameters if ".mixer." in name
        ),
        "mlp": sum(
            parameter.numel() for name, parameter in named_parameters if ".mlp." in name
        ),
        "local_motif_stem": sum(
            parameter.numel() for name, parameter in named_parameters
            if "local_motif_stem" in name
        ),
        "pooling": sum(
            parameter.numel() for name, parameter in named_parameters
            if "pool_projection" in name
        ),
    }
    if config.max_parameters and parameter_breakdown["total"] > config.max_parameters:
        raise ValueError(
            f"model has {parameter_breakdown['total']:,} parameters, exceeding "
            f"the configured budget of {config.max_parameters:,}"
        )
    run_config = {
        **metadata,
        **asdict(config),
        "parameters": parameter_breakdown["total"],
        "parameter_breakdown": parameter_breakdown,
    }
    (output / "config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8"
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        fused=device.type == "cuda",
    )
    batches_per_epoch = len(train_loader)
    if config.max_train_batches:
        batches_per_epoch = min(batches_per_epoch, config.max_train_batches)
    total_steps = max(1, config.epochs * batches_per_epoch)
    warmup_steps = round(total_steps * config.warmup_ratio)
    if not 0.0 <= config.min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be between zero and one")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _scheduler_lambda(
            step, warmup_steps, total_steps, config.min_lr_ratio
        ),
    )
    start_epoch, best, stale_epochs = 0, float("-inf"), 0
    last_path = output / "last.pt"
    if config.resume and last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        best = float(state["best"])
        stale_epochs = int(state.get("stale_epochs", 0))
        _restore_loader_state(train_loader, state.get("train_loader"))
        _restore_rng_state(state.get("rng"))
    metrics_path = output / "metrics.jsonl"
    for epoch in range(start_epoch, config.epochs):
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        train_loss, examples = 0.0, 0
        for batch_index, raw_batch in enumerate(train_loader):
            if config.max_train_batches and batch_index >= config.max_train_batches:
                break
            batch = _move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = _forward(model, batch)
                loss = F.cross_entropy(
                    logits,
                    batch["labels"],
                    label_smoothing=config.label_smoothing,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()
            batch_size = batch["labels"].numel()
            train_loss += float(loss.detach()) * batch_size
            examples += batch_size
        train_finished = time.perf_counter()
        validation = evaluate(
            model,
            validation_loader,
            device,
            num_classes,
            config.max_eval_batches,
        )
        score = validation["accuracy"]
        improved = score > best or not (output / "best.pt").exists()
        if improved:
            best, stale_epochs = score, 0
        else:
            stale_epochs += 1
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss / max(examples, 1),
            "val_loss": validation["loss"],
            "val_accuracy": validation["accuracy"],
            "val_macro_f1": validation["macro_f1"],
            "val_matthews_correlation": validation["matthews_correlation"],
            "seconds": time.perf_counter() - started,
            "train_examples": examples,
            "examples_per_second": examples / max(train_finished - started, 1e-9),
            "peak_gb": (
                torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
            ),
            "lr": optimizer.param_groups[0]["lr"],
        }
        print(json.dumps(metrics, sort_keys=True), flush=True)
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, sort_keys=True) + "\n")
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best": best,
            "stale_epochs": stale_epochs,
            "metrics": metrics,
            "rng": _rng_state(),
            "train_loader": _loader_state(train_loader),
        }
        atomic_save(state, last_path)
        if improved:
            atomic_save(state, output / "best.pt")
        if config.patience and stale_epochs >= config.patience:
            break
    best_state = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best_state["model"])
    evaluation_loader = validation_loader if test_loader is None else test_loader
    final_metrics = evaluate(
        model, evaluation_loader, device, num_classes, config.max_eval_batches
    )
    result = {
        "accuracy": final_metrics["accuracy"],
        "macro_f1": final_metrics["macro_f1"],
        "matthews_correlation": final_metrics["matthews_correlation"],
        "loss": final_metrics["loss"],
        "selected_epoch": int(best_state["epoch"]),
        "parameters": run_config["parameters"],
    }
    history = []
    if metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            try:
                history.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    by_epoch = {int(row["epoch"]): row for row in history}
    result["train_seconds"] = sum(float(row.get("seconds", 0.0)) for row in by_epoch.values())
    result["peak_gb"] = max((float(row.get("peak_gb", 0.0)) for row in by_epoch.values()), default=0.0)
    result["mean_examples_per_second"] = float(
        np.mean([row["examples_per_second"] for row in by_epoch.values()])
    ) if by_epoch else 0.0
    result_name = "validation_metrics.json" if test_loader is None else "test_metrics.json"
    (output / result_name).write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    result_label = "validation" if test_loader is None else "test"
    print(json.dumps({result_label: result}, sort_keys=True), flush=True)
    return result
