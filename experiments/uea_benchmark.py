"""UEA-30 multivariate time-series classification with RRLSSO."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.models import SequenceClassifier, SequenceValueEncoder
from experiments.sequence_benchmarks.common import (
    IndexDataset,
    TrainingConfig,
    collate_values,
    make_loader,
    seed_all,
    stratified_split_indices,
    stratified_subset_indices,
    train_classifier,
)


UEA_30 = (
    "ArticularyWordRecognition",
    "AtrialFibrillation",
    "BasicMotions",
    "CharacterTrajectories",
    "Cricket",
    "DuckDuckGeese",
    "EigenWorms",
    "Epilepsy",
    "ERing",
    "EthanolConcentration",
    "FaceDetection",
    "FingerMovements",
    "HandMovementDirection",
    "Handwriting",
    "Heartbeat",
    "InsectWingbeat",
    "JapaneseVowels",
    "Libras",
    "LSST",
    "MotorImagery",
    "NATOPS",
    "PEMS-SF",
    "PenDigits",
    "PhonemeSpectra",
    "RacketSports",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "SpokenArabicDigits",
    "StandWalkJump",
    "UWaveGestureLibrary",
)


def _sample(collection, index: int) -> np.ndarray:
    values = np.asarray(collection[index], dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"UEA sample must be [channels, time], received {values.shape}")
    return values


def fit_channel_normalizer(collection, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    first = _sample(collection, indices[0])
    sums = np.zeros(first.shape[0], dtype=np.float64)
    squares = np.zeros(first.shape[0], dtype=np.float64)
    counts = np.zeros(first.shape[0], dtype=np.int64)
    for index in indices:
        values = _sample(collection, index)
        if values.shape[0] != len(sums):
            raise ValueError("channel count changes within a UEA dataset")
        finite = np.isfinite(values)
        safe = np.where(finite, values, 0.0)
        sums += safe.sum(axis=1, dtype=np.float64)
        squares += np.square(safe.astype(np.float64, copy=False)).sum(axis=1)
        counts += finite.sum(axis=1)
    counts = np.maximum(counts, 1)
    mean = sums / counts
    variance = np.maximum(squares / counts - mean**2, 1e-8)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


class UEACollectionDataset(Dataset):
    def __init__(
        self,
        collection,
        labels: list[int],
        mean: np.ndarray,
        std: np.ndarray,
        max_length: int,
    ) -> None:
        self.collection = collection
        self.labels = [int(label) for label in labels]
        self.mean = mean[:, None]
        self.std = std[:, None]
        self.max_length = int(max_length)
        self.lengths = [min(_sample(collection, i).shape[1], max_length) for i in range(len(labels))]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        values = _sample(self.collection, index)[:, : self.max_length]
        values = np.nan_to_num((values - self.mean) / self.std, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(np.ascontiguousarray(values.T)), self.labels[index]


def encode_labels(train_labels, test_labels):
    classes = sorted(set(str(label) for label in train_labels))
    mapping = {label: index for index, label in enumerate(classes)}
    try:
        train = [mapping[str(label)] for label in train_labels]
        test = [mapping[str(label)] for label in test_labels]
    except KeyError as error:
        raise ValueError(f"test label absent from training classes: {error}") from error
    return train, test, classes


def load_uea(dataset_name: str, data_root: str):
    try:
        from aeon.datasets import load_classification
    except ImportError as error:
        raise RuntimeError('install the sequence extra: pip install -e ".[sequence]"') from error
    if data_root:
        Path(data_root).mkdir(parents=True, exist_ok=True)
    kwargs = dict(extract_path=data_root or None, return_metadata=False)
    train_x, train_y = load_classification(dataset_name, split="train", **kwargs)
    test_x, test_y = load_classification(dataset_name, split="test", **kwargs)
    return train_x, train_y, test_x, test_y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=UEA_30, default="BasicMotions")
    parser.add_argument("--data-root", default="data/uea")
    parser.add_argument("--output", default="")
    parser.add_argument("--mixer", choices=("mha", "lsso", "rrlsso"), default="rrlsso")
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--position-rank", type=int, default=16)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--max-parameters", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    train_x, train_y, test_x, test_y = load_uea(args.dataset, args.data_root)
    train_labels, test_labels, classes = encode_labels(train_y, test_y)
    train_indices, validation_indices = stratified_split_indices(
        train_labels, args.validation_fraction, args.seed
    )
    if args.max_train_samples:
        selected = stratified_subset_indices(
            [train_labels[index] for index in train_indices], args.max_train_samples, args.seed + 1
        )
        train_indices = [train_indices[index] for index in selected]
    if args.max_eval_samples:
        selected = stratified_subset_indices(
            [train_labels[index] for index in validation_indices],
            args.max_eval_samples,
            args.seed + 2,
        )
        validation_indices = [validation_indices[index] for index in selected]
    mean, std = fit_channel_normalizer(train_x, train_indices)
    observed_max = max(
        max(_sample(train_x, index).shape[1] for index in range(len(train_labels))),
        max(_sample(test_x, index).shape[1] for index in range(len(test_labels))),
    )
    max_length = args.max_length or observed_max
    full_train = UEACollectionDataset(train_x, train_labels, mean, std, max_length)
    test = UEACollectionDataset(test_x, test_labels, mean, std, max_length)
    train = IndexDataset(full_train, train_indices)
    validation = IndexDataset(full_train, validation_indices)
    if args.max_eval_samples:
        test = IndexDataset(
            test,
            stratified_subset_indices(test_labels, args.max_eval_samples, args.seed + 3),
        )
    input_dim = _sample(train_x, train_indices[0]).shape[0]
    encoder = SequenceValueEncoder(
        input_dim,
        max_length=max_length,
        dim=args.dim,
        depth=args.depth,
        num_heads=args.heads,
        mixer=args.mixer,
        rank=args.rank,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        position_rank=args.position_rank,
    )
    model = SequenceClassifier(encoder, len(classes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = [
        make_loader(
            dataset,
            batch_size=args.batch_size if index == 0 else args.eval_batch_size,
            workers=args.workers if index == 0 else args.eval_workers,
            device=device,
            collate_fn=collate_values,
            train=index == 0,
            seed=args.seed,
        )
        for index, dataset in enumerate((train, validation, test))
    ]
    output = args.output or f"runs/sequence/uea-{args.dataset}-{args.mixer}-s{args.seed}"
    train_classifier(
        model,
        *loaders,
        num_classes=len(classes),
        config=TrainingConfig(
            output=output,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            grad_accum=args.grad_accum,
            patience=args.patience,
            seed=args.seed,
            resume=args.resume,
            max_train_batches=args.max_train_batches,
            max_eval_batches=args.max_eval_batches,
            max_parameters=args.max_parameters,
        ),
        metadata={
            "suite": "uea",
            "dataset": args.dataset,
            "mixer": args.mixer,
            "rank": args.rank,
            "dim": args.dim,
            "depth": args.depth,
            "heads": args.heads,
            "max_length": max_length,
            "position_rank": args.position_rank,
            "input_channels": input_dim,
            "classes": classes,
            "validation_fraction": args.validation_fraction,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "workers": args.workers,
            "eval_workers": args.eval_workers,
            "grad_accum": args.grad_accum,
            "effective_batch_size": args.batch_size * args.grad_accum,
            "max_train_samples": args.max_train_samples,
            "max_eval_samples": args.max_eval_samples,
            "split_sizes": {
                "train": len(train),
                "validation": len(validation),
                "test": len(test),
            },
            "archive_split": "official-train-test",
            "data_provenance": {
                "loader": "aeon",
                "aeon_version": importlib.metadata.version("aeon"),
                "extract_path": str(Path(args.data_root).resolve()),
            },
        },
    )


if __name__ == "__main__":
    main()
