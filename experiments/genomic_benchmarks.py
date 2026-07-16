"""GenomicBenchmarks sequence classification with the shared RRLSSO encoder."""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.models import ReverseComplementSequenceClassifier, SequenceMixerEncoder
from experiments.sequence_benchmarks.common import (
    IndexDataset,
    TrainingConfig,
    collate_tokens,
    evaluate,
    make_loader,
    seed_all,
    stratified_split_indices,
    stratified_subset_indices,
    train_classifier,
)


GENOMIC_BENCHMARKS = (
    "dummy_mouse_enhancers_ensembl",
    "demo_coding_vs_intergenomic_seqs",
    "demo_human_or_worm",
    "human_enhancers_cohn",
    "human_enhancers_ensembl",
    "human_ensembl_regulatory",
    "human_nontata_promoters",
    "human_ocr_ensembl",
)


class NucleotideTokenizer:
    alphabet = "ACGTN"

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.lookup = {token: index + 2 for index, token in enumerate(self.alphabet)}
        self.vocab_size = len(self.lookup) + 2

    @property
    def complement_ids(self) -> torch.Tensor:
        complement = torch.arange(self.vocab_size, dtype=torch.long)
        for first, second in (("A", "T"), ("C", "G"), ("N", "N")):
            complement[self.lookup[first]] = self.lookup[second]
            complement[self.lookup[second]] = self.lookup[first]
        return complement

    def encode(self, sequence: str, max_length: int) -> torch.Tensor:
        sequence = str(sequence).upper()[:max_length]
        return torch.tensor(
            [self.lookup.get(token, self.unk_token_id) for token in sequence],
            dtype=torch.uint8,
        )


class StringSequenceDataset(Dataset):
    def __init__(
        self,
        rows,
        tokenizer: NucleotideTokenizer,
        max_length: int,
        *,
        sequence_key: str = "seq",
        label_key: str = "label",
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.sequence_key = sequence_key
        self.label_key = label_key
        try:
            sequences = rows[sequence_key]
            self.lengths = [min(len(str(sequence)), max_length) for sequence in sequences]
            self.labels = [int(label) for label in rows[label_key]]
        except (TypeError, KeyError):
            self.lengths, self.labels = [], []
            for index in range(len(rows)):
                sequence, label = self._raw(index)
                self.lengths.append(min(len(sequence), max_length))
                self.labels.append(label)

    def _raw(self, index: int) -> tuple[str, int]:
        row = self.rows[index]
        if isinstance(row, dict):
            return str(row[self.sequence_key]), int(row[self.label_key])
        sequence, label = row
        return str(sequence), int(label)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        sequence, label = self._raw(index)
        return self.tokenizer.encode(sequence, self.max_length), label


class FolderRows(Dataset):
    def __init__(self, split_dir: Path, class_to_label: dict[str, int]) -> None:
        self.rows: list[tuple[Path, int]] = []
        for class_name, label in class_to_label.items():
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"missing genomic class directory: {class_dir}")
            self.rows.extend((path, label) for path in sorted(class_dir.iterdir()) if path.is_file())

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        path, label = self.rows[index]
        return path.read_text(encoding="utf-8").strip(), label


def load_splits(
    dataset_name: str, data_root: str, cache_dir: str, revision: str = ""
):
    if data_root:
        base = Path(data_root) / dataset_name
        train_dir, test_dir = base / "train", base / "test"
        if not train_dir.is_dir() or not test_dir.is_dir():
            raise FileNotFoundError(f"expected {train_dir} and {test_dir}")
        classes = sorted(path.name for path in train_dir.iterdir() if path.is_dir())
        mapping = {name: index for index, name in enumerate(classes)}
        provenance = {"source": "local", "path": str(base.resolve())}
        return (
            FolderRows(train_dir, mapping), FolderRows(test_dir, mapping),
            "sequence", "label", provenance,
        )
    # ``datasets.load_dataset`` contacts the Hub even when the Arrow payload is
    # already complete. Formal sweeps should remain restartable during a Hub
    # outage, so prefer a matching on-disk revision before making any request.
    if cache_dir:
        dataset_cache = (
            Path(cache_dir)
            / f"katarinagresova___genomic_benchmarks_{dataset_name}"
            / "default"
            / "0.0.0"
        )
        candidates = []
        if revision:
            candidates.append(dataset_cache / revision)
        elif dataset_cache.is_dir():
            candidates.extend(
                sorted(
                    (path for path in dataset_cache.iterdir() if path.is_dir()),
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )
            )
        for candidate in candidates:
            train_files = tuple(candidate.glob("*-train.arrow"))
            test_files = tuple(candidate.glob("*-test.arrow"))
            if len(train_files) != 1 or len(test_files) != 1:
                continue
            from datasets import Dataset

            train_rows = Dataset.from_file(str(train_files[0]))
            test_rows = Dataset.from_file(str(test_files[0]))
            sequence_key = (
                "seq" if "seq" in train_rows.column_names else "sequence"
            )
            provenance = {
                "source": "huggingface-cache",
                "repository": f"katarinagresova/Genomic_Benchmarks_{dataset_name}",
                "revision": candidate.name,
                "train_fingerprint": train_rows._fingerprint,
                "test_fingerprint": test_rows._fingerprint,
            }
            return train_rows, test_rows, sequence_key, "label", provenance
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi

        repository = f"katarinagresova/Genomic_Benchmarks_{dataset_name}"
        resolved_revision = revision or HfApi().dataset_info(repository).sha
        rows = load_dataset(
            repository, revision=resolved_revision, cache_dir=cache_dir or None
        )
        sequence_key = "seq" if "seq" in rows["train"].column_names else "sequence"
        provenance = {
            "source": "huggingface",
            "repository": repository,
            "revision": resolved_revision,
            "train_fingerprint": rows["train"]._fingerprint,
            "test_fingerprint": rows["test"]._fingerprint,
        }
        return rows["train"], rows["test"], sequence_key, "label", provenance
    except Exception as hf_error:
        try:
            from genomic_benchmarks.dataset_getters.pytorch_datasets import get_dataset

            return (
                get_dataset(dataset_name, "train"),
                get_dataset(dataset_name, "test"),
                "sequence",
                "label",
                {"source": "genomic-benchmarks-package"},
            )
        except Exception as official_error:
            raise RuntimeError(
                "failed to load GenomicBenchmarks from both the author HF mirror and "
                "the official genomic-benchmarks package"
            ) from ExceptionGroup("dataset loading failures", [hf_error, official_error])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-profile",
        choices=(
            "standard",
            "hyenadna-flavor",
            "hyenadna-flavor-rc",
            "hyenadna-flavor-rc-mutation",
        ),
        default="standard",
    )
    parser.add_argument("--dataset", choices=GENOMIC_BENCHMARKS, default="human_enhancers_cohn")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--cache-dir", default="data/genomic_benchmarks")
    parser.add_argument("--dataset-revision", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--mixer", choices=("mha", "lsso", "rrlsso"), default="rrlsso")
    parser.add_argument(
        "--max-length",
        type=int,
        default=0,
        help="sequence truncation length; zero preserves the longest example in the task",
    )
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--position-rank", type=int, default=0)
    parser.add_argument(
        "--local-motif-kernel",
        type=int,
        default=0,
        help="odd depthwise motif-stem kernel; zero disables the local stem",
    )
    parser.add_argument(
        "--local-motif-dilations",
        type=int,
        nargs="*",
        default=(),
        help="enable gated kernel-9 local blocks with these dilation rates",
    )
    parser.add_argument("--local-motif-layer-scale", type=float, default=1e-3)
    parser.add_argument("--pooling", choices=("mean", "max", "meanmax"), default="mean")
    parser.add_argument("--reverse-complement-probability", type=float, default=0.0)
    parser.add_argument("--reverse-complement-eval", action="store_true")
    parser.add_argument("--posthoc-rc-eval", action="store_true")
    parser.add_argument("--mutation-probability", type=float, default=0.0)
    parser.add_argument("--mutation-clean-epochs", type=int, default=0)
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="select and report the best checkpoint without evaluating the benchmark test split",
    )
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--embedding-dropout",
        type=float,
        default=None,
        help="embedding-only dropout; defaults to --dropout when omitted",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--min-lr-ratio", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--max-parameters", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evaluate-checkpoint", default="")
    return parser.parse_args()


def apply_training_profile(args: argparse.Namespace) -> None:
    if args.training_profile not in {
        "hyenadna-flavor",
        "hyenadna-flavor-rc",
        "hyenadna-flavor-rc-mutation",
    }:
        return
    # Transfer the architecture-agnostic portion of the official HyenaDNA
    # scratch recipe. Keep RRLSSO's milder weight decay because Hyena's
    # layer-specific decay policy does not map cleanly to solve parameters.
    args.epochs = 100
    args.patience = 100
    # Preserve the effective batch of 128 while avoiding long-sequence
    # activation pressure on 16 GiB GPUs.
    args.batch_size = 64
    args.grad_accum = 2
    args.eval_batch_size = 256
    args.lr = 6e-4
    args.weight_decay = 0.01
    args.warmup_ratio = 0.01
    args.min_lr_ratio = 0.1
    args.dropout = 0.0
    args.embedding_dropout = 0.1
    args.pooling = "mean"
    args.reverse_complement_probability = (
        0.5 if args.training_profile in {
            "hyenadna-flavor-rc", "hyenadna-flavor-rc-mutation"
        } else 0.0
    )
    args.reverse_complement_eval = False
    if args.training_profile == "hyenadna-flavor-rc-mutation":
        args.mutation_probability = 0.002
        args.mutation_clean_epochs = 20


def main() -> None:
    args = parse_args()
    apply_training_profile(args)
    seed_all(args.seed)
    train_rows, test_rows, sequence_key, label_key, provenance = load_splits(
        args.dataset, args.data_root, args.cache_dir, args.dataset_revision
    )
    # Architecture selection must not inspect the held-out benchmark test set.
    max_length = args.max_length or max(
        len(str(sequence)) for sequence in train_rows[sequence_key]
    )
    tokenizer = NucleotideTokenizer()
    full_train = StringSequenceDataset(
        train_rows, tokenizer, max_length, sequence_key=sequence_key, label_key=label_key
    )
    test = None if args.validation_only else StringSequenceDataset(
        test_rows, tokenizer, max_length, sequence_key=sequence_key, label_key=label_key
    )
    train_indices, validation_indices = stratified_split_indices(
        full_train.labels, args.validation_fraction, args.seed
    )
    if args.max_train_samples:
        selected = stratified_subset_indices(
            [full_train.labels[index] for index in train_indices],
            args.max_train_samples,
            args.seed + 1,
        )
        train_indices = [train_indices[index] for index in selected]
    if args.max_eval_samples:
        selected = stratified_subset_indices(
            [full_train.labels[index] for index in validation_indices],
            args.max_eval_samples,
            args.seed + 2,
        )
        validation_indices = [validation_indices[index] for index in selected]
        if test is not None:
            test = IndexDataset(
                test,
                stratified_subset_indices(test.labels, args.max_eval_samples, args.seed + 3),
            )
    train = IndexDataset(full_train, train_indices)
    validation = IndexDataset(full_train, validation_indices)
    num_classes = max(full_train.labels) + 1
    encoder = SequenceMixerEncoder(
        tokenizer.vocab_size,
        max_length=max_length,
        pad_token_id=tokenizer.pad_token_id,
        dim=args.dim,
        depth=args.depth,
        num_heads=args.heads,
        mixer=args.mixer,
        rank=args.rank,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        embedding_dropout=args.embedding_dropout,
        position_rank=args.position_rank,
        pooling=args.pooling,
        local_motif_kernel=args.local_motif_kernel,
        local_motif_dilations=tuple(args.local_motif_dilations),
        local_motif_layer_scale=args.local_motif_layer_scale,
    )
    model = ReverseComplementSequenceClassifier(
        encoder,
        num_classes,
        complement_ids=tokenizer.complement_ids,
        reverse_complement_probability=args.reverse_complement_probability,
        reverse_complement_eval=args.reverse_complement_eval,
        mutation_probability=args.mutation_probability,
        mutation_stop_epoch=max(0, args.epochs - args.mutation_clean_epochs),
    )
    output = args.output or f"runs/sequence/genomic-{args.dataset}-{args.mixer}-s{args.seed}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    collate = functools.partial(collate_tokens, pad_token_id=tokenizer.pad_token_id)
    train_loader = make_loader(
        train, batch_size=args.batch_size, workers=args.workers, device=device,
        collate_fn=collate, train=True, seed=args.seed,
    )
    validation_loader = make_loader(
        validation, batch_size=args.eval_batch_size, workers=args.workers, device=device,
        collate_fn=collate, train=False, seed=args.seed,
    )
    test_loader = None if test is None else make_loader(
        test, batch_size=args.eval_batch_size, workers=args.workers, device=device,
        collate_fn=collate, train=False, seed=args.seed,
    )
    if args.evaluate_checkpoint:
        state = torch.load(args.evaluate_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        if args.posthoc_rc_eval:
            model.reverse_complement_eval = True
        model.to(device)
        started = time.perf_counter()
        metrics = evaluate(model, validation_loader, device, num_classes)
        result = {
            **metrics,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "selected_epoch": int(state.get("epoch", -1)),
            "evaluation_seconds": time.perf_counter() - started,
            "checkpoint": str(Path(args.evaluate_checkpoint).resolve()),
            "posthoc_rc_eval": args.posthoc_rc_eval,
        }
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        temporary = output_path / "validation_metrics.json.tmp"
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(output_path / "validation_metrics.json")
        print(json.dumps({"validation": result}, sort_keys=True), flush=True)
        return
    train_classifier(
        model,
        train_loader,
        validation_loader,
        test_loader,
        num_classes=num_classes,
        config=TrainingConfig(
            output=output,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            min_lr_ratio=args.min_lr_ratio,
            posthoc_rc_eval=args.posthoc_rc_eval,
            grad_accum=args.grad_accum,
            patience=args.patience,
            seed=args.seed,
            resume=args.resume,
            max_train_batches=args.max_train_batches,
            max_eval_batches=args.max_eval_batches,
            max_parameters=args.max_parameters,
        ),
        metadata={
            "suite": "genomic",
            "training_profile": args.training_profile,
            "dataset": args.dataset,
            "mixer": args.mixer,
            "rank": args.rank,
            "dim": args.dim,
            "depth": args.depth,
            "heads": args.heads,
            "max_length": max_length,
            "requested_max_length": args.max_length,
            "position_rank": args.position_rank,
            "embedding_dropout": args.embedding_dropout,
            "local_motif_kernel": args.local_motif_kernel,
            "local_motif_dilations": list(args.local_motif_dilations),
            "local_motif_layer_scale": args.local_motif_layer_scale,
            "pooling": args.pooling,
            "reverse_complement_probability": args.reverse_complement_probability,
            "reverse_complement_eval": args.reverse_complement_eval,
            "posthoc_rc_eval": args.posthoc_rc_eval,
            "mutation_probability": args.mutation_probability,
            "mutation_clean_epochs": args.mutation_clean_epochs,
            "validation_only": args.validation_only,
            "validation_fraction": args.validation_fraction,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch_size": args.batch_size * args.grad_accum,
            "eval_batch_size": args.eval_batch_size,
            "workers": args.workers,
            "max_train_samples": args.max_train_samples,
            "max_eval_samples": args.max_eval_samples,
            "split_sizes": {
                "train": len(train),
                "validation": len(validation),
                "test": None if test is None else len(test),
            },
            "pretraining": "none",
            "data_provenance": provenance,
        },
    )


if __name__ == "__main__":
    main()
