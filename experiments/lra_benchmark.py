"""Modern PyTorch runner for four Long Range Arena tasks."""

from __future__ import annotations

import argparse
import functools
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.models import (
    SequenceClassifier,
    SequenceMixerEncoder,
    SequencePairClassifier,
    SequenceValueEncoder,
)
from experiments.sequence_benchmarks.common import (
    IndexDataset,
    TrainingConfig,
    collate_token_pairs,
    collate_tokens,
    collate_values,
    make_loader,
    seed_all,
    stratified_split_indices,
    stratified_subset_indices,
    train_classifier,
)
from experiments.sequence_benchmarks.lra_data import (
    CharacterVocabulary,
    PathfinderDataset,
    build_packed_pairs,
    build_packed_tokens,
    download_kaggle_lra,
    iter_aan,
    iter_listops,
    resolve_listops_files,
    resolve_pathfinder_directory,
    source_signature,
)


TASK_DEFAULTS = {
    "listops": {
        "max_length": 2048, "epochs": 40, "batch_size": 50, "classes": 10,
        "patience": 0,
    },
    "text": {
        "max_length": 4096, "epochs": 32, "batch_size": 32, "classes": 2,
        "patience": 0,
    },
    "retrieval": {
        "max_length": 4000, "epochs": 20, "batch_size": 64, "classes": 2,
        "patience": 0,
    },
    # Match the official optimization scale without requiring a physical batch
    # of 512: 64 samples x 8 accumulation steps, one-epoch warmup, no early stop.
    "pathfinder": {
        "max_length": 1024, "epochs": 200, "batch_size": 64, "classes": 2,
        "grad_accum": 8, "lr": 1e-3, "warmup_ratio": 1 / 200, "patience": 0,
    },
}


def _data_source_identity(task: str, data_root: Path) -> dict[str, object]:
    if task == "text":
        return {"repository": "stanfordnlp/imdb", "transport": "huggingface-datasets"}
    if task == "retrieval":
        return {"repository": "OpenNLPLab/lra", "subset": "data/aan"}
    manifest_path = data_root / "source-manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        key = "listops" if task == "listops" else "pathfinder32_hard"
        return {
            "community_mirror": manifest.get("community_mirror"),
            "upstream_definition": manifest.get("upstream_definition"),
            "audit": manifest.get(key),
        }
    return {
        "repository": "explicit-local-input",
        "upstream_definition": "google-research/long-range-arena@cd31e5c6",
    }


def _limit(dataset, count: int, seed: int):
    if not count or len(dataset) <= count:
        return dataset
    labels = getattr(dataset, "labels", None)
    if labels is None:
        indices = list(range(len(dataset)))
        random.Random(seed).shuffle(indices)
        indices = indices[:count]
    else:
        indices = stratified_subset_indices(labels, count, seed)
    return IndexDataset(dataset, indices)


def _find_directory(root: Path, alternatives: tuple[str, ...]) -> Path:
    for relative in alternatives:
        candidate = root / relative
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"none of {[str(root / item) for item in alternatives]} exists"
    )


def _load_or_build_vocab(path: Path, builder) -> CharacterVocabulary:
    if path.exists():
        return CharacterVocabulary.load(path)
    vocabulary = builder()
    vocabulary.save(path)
    return vocabulary


def prepare_listops(data_root: Path, cache_root: Path, max_length: int):
    files = resolve_listops_files(data_root)
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    vocab_path = cache_root / "listops" / "vocab.json"
    vocabulary = _load_or_build_vocab(
        vocab_path,
        lambda: CharacterVocabulary.from_listops(text for text, _ in iter_listops(files["train"])),
    )
    datasets = {}
    for split, path in files.items():
        prefix = cache_root / "listops" / f"{split}-l{max_length}"
        datasets[split] = build_packed_tokens(
            prefix,
            iter_listops(path),
            lambda text: vocabulary.encode_listops(text, max_length),
            manifest={
                "schema": 1,
                "task": "listops",
                "split": split,
                "max_length": max_length,
                "vocabulary": vocabulary.fingerprint,
                "source": source_signature(path),
            },
        )
    empty_splits = [split for split, dataset in datasets.items() if len(dataset) == 0]
    if empty_splits:
        raise RuntimeError(
            f"AAN preprocessing produced empty splits: {empty_splits}; "
            "check the source label and column format"
        )
    return datasets["train"], datasets["val"], datasets["test"], vocabulary


def prepare_text(cache_root: Path, max_length: int, seed: int, validation_fraction: float):
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("install the auxiliary extra to load LRA Text") from error
    rows = load_dataset("stanfordnlp/imdb", cache_dir=str(cache_root / "hf"))
    vocab_path = cache_root / "text" / "vocab.json"
    vocabulary = _load_or_build_vocab(
        vocab_path,
        lambda: CharacterVocabulary.from_texts(rows["train"]["text"], min_frequency=15),
    )
    full_train = build_packed_tokens(
        cache_root / "text" / f"train-l{max_length}",
        ((row["text"], int(row["label"])) for row in rows["train"]),
        lambda text: vocabulary.encode_chars(text, max_length),
        manifest={
            "schema": 1,
            "task": "text",
            "split": "train",
            "max_length": max_length,
            "vocabulary": vocabulary.fingerprint,
            "source_fingerprint": rows["train"]._fingerprint,
        },
    )
    test = build_packed_tokens(
        cache_root / "text" / f"test-l{max_length}",
        ((row["text"], int(row["label"])) for row in rows["test"]),
        lambda text: vocabulary.encode_chars(text, max_length),
        manifest={
            "schema": 1,
            "task": "text",
            "split": "test",
            "max_length": max_length,
            "vocabulary": vocabulary.fingerprint,
            "source_fingerprint": rows["test"]._fingerprint,
        },
    )
    train_indices, validation_indices = stratified_split_indices(
        full_train.labels, validation_fraction, seed
    )
    return (
        IndexDataset(full_train, train_indices),
        IndexDataset(full_train, validation_indices),
        test,
        vocabulary,
    )


def _download_aan(data_root: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as error:
        raise RuntimeError("install huggingface-hub to download the OpenNLPLab AAN mirror") from error
    local = data_root / "opennlplab"
    filenames = (
        "data/aan/new_aan_pairs.train.tsv",
        "data/aan/new_aan_pairs.eval.tsv",
        "data/aan/new_aan_pairs.test.tsv",
    )
    snapshot_download(
        repo_id="OpenNLPLab/lra",
        repo_type="dataset",
        local_dir=local,
        allow_patterns=list(filenames),
        # These are multi-gigabyte LFS objects. Serial downloads avoid four
        # concurrent sparse temporary files and make progress recoverable.
        max_workers=1,
    )
    # A killed local-dir snapshot can leave a cached tree manifest beside
    # incomplete LFS objects. In that state snapshot_download may return even
    # though the materialized files are absent, so verify and resume each one.
    for filename in filenames:
        if not (local / filename).is_file():
            hf_hub_download(
                repo_id="OpenNLPLab/lra",
                repo_type="dataset",
                filename=filename,
                local_dir=local,
            )
    return local / "data" / "aan"


def prepare_retrieval(
    data_root: Path, cache_root: Path, max_length: int, download: bool
):
    try:
        source = _find_directory(data_root, ("aan", "tsv_data", "opennlplab/data/aan"))
    except FileNotFoundError:
        if not download:
            raise FileNotFoundError(
                "AAN data is missing; pass --download-aan to fetch the OpenNLPLab HF mirror"
            )
        source = _download_aan(data_root)
    files = {
        "train": source / "new_aan_pairs.train.tsv",
        "val": source / "new_aan_pairs.eval.tsv",
        "test": source / "new_aan_pairs.test.tsv",
    }
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    vocab_path = cache_root / "retrieval" / "vocab.json"

    # Retrieval is a byte-level LRA task. A fixed vocabulary avoids an extra
    # character-counting pass over the 8.5 GB training TSV and is independent
    # of which split happens to be scanned first.
    expected_vocabulary = CharacterVocabulary.from_bytes()
    vocabulary = (
        CharacterVocabulary.load(vocab_path)
        if vocab_path.exists()
        else expected_vocabulary
    )
    if vocabulary.fingerprint != expected_vocabulary.fingerprint:
        vocabulary = expected_vocabulary
    vocabulary.save(vocab_path)
    datasets = {}
    for split, path in files.items():
        datasets[split] = build_packed_pairs(
            cache_root / "retrieval" / f"{split}-l{max_length}",
            iter_aan(path),
            lambda text: vocabulary.encode_bytes(text, max_length),
            manifest={
                "schema": 2,
                "task": "retrieval",
                "split": split,
                "max_length": max_length,
                "vocabulary": vocabulary.fingerprint,
                "source": source_signature(path),
            },
        )
    return datasets["train"], datasets["val"], datasets["test"], vocabulary


def prepare_pathfinder(data_root: Path, resolution: int, seed: int):
    source = resolve_pathfinder_directory(data_root, resolution)
    full = PathfinderDataset(source)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(full), generator=generator).tolist()
    validation_count = int(0.1 * len(full))
    test_count = int(0.1 * len(full))
    validation = IndexDataset(full, order[:validation_count])
    test = IndexDataset(full, order[validation_count : validation_count + test_count])
    train = IndexDataset(full, order[validation_count + test_count :])
    return train, validation, test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=tuple(TASK_DEFAULTS), default="listops")
    parser.add_argument("--data-root", default="data/lra")
    parser.add_argument("--cache-dir", default="data/lra_cache")
    parser.add_argument("--download-aan", action="store_true")
    parser.add_argument(
        "--download-lra",
        action="store_true",
        help="download pinned Kaggle mirror v1 for missing ListOps/Pathfinder data",
    )
    parser.add_argument("--pathfinder-resolution", type=int, default=32)
    parser.add_argument(
        "--pathfinder-local-kernel", type=int, default=0,
        help="odd 2-D depthwise kernel interleaved before each Pathfinder mixer",
    )
    parser.add_argument(
        "--pathfinder-local-dilations", type=int, nargs="*", default=(),
        help="one dilation per encoder block; empty disables the local branch",
    )
    parser.add_argument("--pathfinder-local-layer-scale", type=float, default=1e-3)
    parser.add_argument(
        "--pathfinder-local-lr-multiplier", type=float, default=1.0,
        help="learning-rate multiplier for Pathfinder local spatial blocks",
    )
    parser.add_argument(
        "--split-seed", type=int, default=0,
        help="fixed data-partition seed; independent of model initialization seed",
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--mixer", choices=("mha", "lsso", "rrlsso"), default="rrlsso")
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--position-rank", type=int, default=0)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--pooling", choices=("mean", "max", "meanmax"), default="mean",
        help="sequence readout; meanmax is useful for sparse visual markers",
    )
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--grad-accum", type=int, default=0, help="0 selects the task default")
    parser.add_argument("--lr", type=float, default=0.0, help="0 selects the task default")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=-1.0, help="negative selects the task default")
    parser.add_argument("--patience", type=int, default=-1, help="negative selects the task default")
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
    defaults = TASK_DEFAULTS[args.task]
    max_length = args.max_length or defaults["max_length"]
    epochs = args.epochs or defaults["epochs"]
    batch_size = args.batch_size or defaults["batch_size"]
    eval_batch_size = args.eval_batch_size or batch_size
    grad_accum = args.grad_accum or defaults.get("grad_accum", 1)
    lr = args.lr or defaults.get("lr", 3e-4)
    warmup_ratio = (
        args.warmup_ratio if args.warmup_ratio >= 0 else defaults.get("warmup_ratio", 0.05)
    )
    patience = args.patience if args.patience >= 0 else defaults.get("patience", 10)
    output = args.output or f"runs/sequence/lra-{args.task}-{args.mixer}-s{args.seed}"
    data_root, cache_root = Path(args.data_root), Path(args.cache_dir)
    if args.download_lra and args.task in {"listops", "pathfinder"}:
        download_kaggle_lra(data_root)
    pair_task = False
    if args.task == "listops":
        train, validation, test, vocabulary = prepare_listops(data_root, cache_root, max_length)
        collate = functools.partial(collate_tokens, pad_token_id=vocabulary.pad_token_id)
    elif args.task == "text":
        train, validation, test, vocabulary = prepare_text(
            cache_root, max_length, args.seed, args.validation_fraction
        )
        collate = functools.partial(collate_tokens, pad_token_id=vocabulary.pad_token_id)
    elif args.task == "retrieval":
        train, validation, test, vocabulary = prepare_retrieval(
            data_root, cache_root, max_length, args.download_aan
        )
        collate = functools.partial(collate_token_pairs, pad_token_id=vocabulary.pad_token_id)
        pair_task = True
    else:
        train, validation, test = prepare_pathfinder(
            data_root, args.pathfinder_resolution, args.split_seed
        )
        max_length = args.pathfinder_resolution**2
        vocabulary = None
        collate = collate_values
    train = _limit(train, args.max_train_samples, args.seed + 1)
    validation = _limit(validation, args.max_eval_samples, args.seed + 2)
    test = _limit(test, args.max_eval_samples, args.seed + 3)
    if args.task == "pathfinder":
        encoder = SequenceValueEncoder(
            1, max_length=max_length, dim=args.dim, depth=args.depth,
            num_heads=args.heads, mixer=args.mixer, rank=args.rank,
            mlp_ratio=args.mlp_ratio, dropout=args.dropout,
            position_rank=args.position_rank,
            pooling=args.pooling,
            spatial_shape=(args.pathfinder_resolution, args.pathfinder_resolution),
            local_spatial_kernel=args.pathfinder_local_kernel,
            local_spatial_dilations=tuple(args.pathfinder_local_dilations),
            local_spatial_layer_scale=args.pathfinder_local_layer_scale,
        )
        model = SequenceClassifier(encoder, defaults["classes"])
    else:
        encoder = SequenceMixerEncoder(
            vocabulary.vocab_size, max_length=max_length,
            pad_token_id=vocabulary.pad_token_id, dim=args.dim, depth=args.depth,
            num_heads=args.heads, mixer=args.mixer, rank=args.rank,
            mlp_ratio=args.mlp_ratio, dropout=args.dropout,
            position_rank=args.position_rank, pooling=args.pooling,
        )
        model = (
            SequencePairClassifier(encoder, defaults["classes"])
            if pair_task
            else SequenceClassifier(encoder, defaults["classes"])
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = [
        make_loader(
            dataset,
            batch_size=batch_size if index == 0 else eval_batch_size,
            workers=args.workers if index == 0 else args.eval_workers,
            device=device,
            collate_fn=collate,
            train=index == 0,
            seed=args.seed,
        )
        for index, dataset in enumerate((train, validation, test))
    ]
    train_classifier(
        model,
        *loaders,
        num_classes=defaults["classes"],
        config=TrainingConfig(
            output=output,
            epochs=epochs,
            lr=lr,
            weight_decay=args.weight_decay,
            warmup_ratio=warmup_ratio,
            local_spatial_lr_multiplier=args.pathfinder_local_lr_multiplier,
            grad_accum=grad_accum,
            patience=patience,
            seed=args.seed,
            resume=args.resume,
            max_train_batches=args.max_train_batches,
            max_eval_batches=args.max_eval_batches,
            max_parameters=args.max_parameters,
        ),
        metadata={
            "suite": "lra",
            "dataset": args.task,
            "mixer": args.mixer,
            "rank": args.rank,
            "dim": args.dim,
            "depth": args.depth,
            "heads": args.heads,
            "max_length": max_length,
            "position_rank": args.position_rank,
            "pooling": args.pooling,
            "validation_fraction": args.validation_fraction,
            "batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "workers": args.workers,
            "eval_workers": args.eval_workers,
            "grad_accum": grad_accum,
            "effective_batch_size": batch_size * grad_accum,
            "lr": lr,
            "warmup_ratio": warmup_ratio,
            "patience": patience,
            "spatial_positioning": (
                "1d-rope" if args.mixer == "mha" else
                "rank-rotary" if args.mixer == "rrlsso" else "none"
            ) if args.task == "pathfinder" else (
                "1d-rope" if args.mixer == "mha" else
                "1d-rank-rotary" if args.mixer == "rrlsso" else "none"
            ),
            "pathfinder_local_kernel": args.pathfinder_local_kernel,
            "pathfinder_local_dilations": list(args.pathfinder_local_dilations),
            "pathfinder_local_layer_scale": args.pathfinder_local_layer_scale,
            "pathfinder_local_lr_multiplier": args.pathfinder_local_lr_multiplier,
            "max_train_samples": args.max_train_samples,
            "max_eval_samples": args.max_eval_samples,
            "split_sizes": {
                "train": len(train),
                "validation": len(validation),
                "test": len(test),
            },
            "protocol": "native-pytorch-reported-baseline",
            "data_definition": "google-research/long-range-arena@cd31e5c6",
            "data_source": _data_source_identity(args.task, data_root),
            "split_seed": args.split_seed,
        },
    )


if __name__ == "__main__":
    main()
