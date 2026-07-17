"""Compare MHA and optimized RRLSSO on the formal GenomicBenchmarks recipe.

The benchmark reports analytic mixer MACs and times one complete training epoch
with the same dynamic length buckets, augmentation, BF16 autocast, gradient
accumulation, clipping, and fused AdamW used by the formal DNA experiments.
"""

from __future__ import annotations

import argparse
import functools
import gc
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.models import ReverseComplementSequenceClassifier, SequenceMixerEncoder
from experiments.genomic_benchmarks import (
    GENOMIC_BENCHMARKS,
    NucleotideTokenizer,
    StringSequenceDataset,
    load_splits,
)
from experiments.sequence_benchmarks.common import (
    IndexDataset,
    collate_tokens,
    make_loader,
    seed_all,
    stratified_split_indices,
)
from lsso.mathdx_backend import is_mathdx_available, mathdx_load_error


DIM = 256
DEPTH = 2
HEADS = 8
RANK = 32
BATCH_SIZE = 64
GRAD_ACCUM = 2


def mixer_macs(
    mixer: str,
    sequence: int,
    *,
    dim: int = DIM,
    heads: int = HEADS,
    rank: int = RANK,
) -> int:
    """Return per-sample, per-layer multiply-accumulates.

    MHA counts QKV/output projections, QK^T, and AV. RRLSSO counts its
    joint U/C projection, output projection, U^T U, U^T C, U readout, and
    dense-equivalent Cholesky/triangular-solve work. Elementwise operations,
    normalization, rotary transforms, softmax, and activation functions are
    deliberately excluded from both sides.
    """
    n, d, h, r = int(sequence), int(dim), int(heads), int(rank)
    if mixer == "mha":
        return 4 * n * d * d + 2 * n * n * d
    if mixer != "rrlsso":
        raise ValueError(f"unsupported mixer: {mixer}")
    projection = n * d * (2 * d + h * r)
    statistics_and_readout = h * n * r * r + 2 * n * r * d
    triangular_solves = d * r * r
    cholesky = round(h * r**3 / 6)
    return projection + statistics_and_readout + triangular_solves + cholesky


def _make_model(
    mixer: str,
    *,
    vocab_size: int,
    pad_token_id: int,
    complement_ids: torch.Tensor,
    max_length: int,
    num_classes: int,
) -> ReverseComplementSequenceClassifier:
    encoder = SequenceMixerEncoder(
        vocab_size,
        max_length=max_length,
        pad_token_id=pad_token_id,
        dim=DIM,
        depth=DEPTH,
        num_heads=HEADS,
        mixer=mixer,
        rank=RANK,
        mlp_ratio=4.0,
        dropout=0.0,
        embedding_dropout=0.1,
        pooling="mean",
        local_motif_kernel=7,
    )
    return ReverseComplementSequenceClassifier(
        encoder,
        num_classes,
        complement_ids=complement_ids,
        reverse_complement_probability=0.5,
        reverse_complement_eval=False,
        mutation_probability=0.002,
        mutation_stop_epoch=80,
    )


def _step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    raw_batch: dict[str, torch.Tensor | float],
    device: torch.device,
    *,
    update: bool,
    accumulated_examples: int,
) -> tuple[int, int, int]:
    inputs = raw_batch["inputs"].to(device, non_blocking=True)
    mask = raw_batch["mask"].to(device, non_blocking=True)
    labels = raw_batch["labels"].to(device, non_blocking=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(
            inputs,
            mask,
            padding_ratio_hint=float(raw_batch["padding_ratio"]),
        )
        loss = F.cross_entropy(logits, labels)
    batch_examples = labels.numel()
    (loss * batch_examples).backward()
    accumulated_examples += batch_examples
    if update:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(accumulated_examples)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        accumulated_examples = 0
    valid_tokens = int(mask.sum().item())
    return accumulated_examples, batch_examples, valid_tokens


def benchmark_epoch(
    mixer: str,
    train: IndexDataset,
    tokenizer: NucleotideTokenizer,
    *,
    max_length: int,
    num_classes: int,
    workers: int,
    warmup_steps: int,
    seed: int,
) -> dict[str, float | int | str]:
    device = torch.device("cuda")
    seed_all(seed)
    model = _make_model(
        mixer,
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        complement_ids=tokenizer.complement_ids,
        max_length=max_length,
        num_classes=num_classes,
    ).to(device)
    model.train()
    model.set_augmentation_epoch(0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=6e-4, weight_decay=0.01, fused=True
    )
    collate = functools.partial(collate_tokens, pad_token_id=tokenizer.pad_token_id)
    loader = make_loader(
        train,
        batch_size=BATCH_SIZE,
        workers=workers,
        device=device,
        collate_fn=collate,
        train=True,
        seed=seed,
    )

    # Allocate optimizer state, CUDA workspaces, rotary caches, and compiled
    # backend state before resetting the measured peak and timers.
    warmup_iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    accumulated = 0
    for index in range(warmup_steps):
        try:
            batch = next(warmup_iterator)
        except StopIteration:
            break
        accumulated, _, _ = _step(
            model,
            optimizer,
            batch,
            device,
            update=(index + 1) % GRAD_ACCUM == 0,
            accumulated_examples=accumulated,
        )
    if accumulated:
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    optimizer.zero_grad(set_to_none=True)
    accumulated = examples = valid_tokens = 0
    batches = len(loader)
    gpu_start = torch.cuda.Event(enable_timing=True)
    gpu_end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    gpu_start.record()
    for index, batch in enumerate(loader):
        update = (index + 1) % GRAD_ACCUM == 0 or index + 1 == batches
        accumulated, batch_examples, batch_tokens = _step(
            model,
            optimizer,
            batch,
            device,
            update=update,
            accumulated_examples=accumulated,
        )
        examples += batch_examples
        valid_tokens += batch_tokens
    gpu_end.record()
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    gpu_seconds = gpu_start.elapsed_time(gpu_end) / 1000.0
    result: dict[str, float | int | str] = {
        "mixer": mixer,
        "batches": batches,
        "examples": examples,
        "valid_tokens": valid_tokens,
        "mean_valid_length": valid_tokens / max(examples, 1),
        "gpu_seconds": gpu_seconds,
        "wall_seconds": wall_seconds,
        "examples_per_second": examples / wall_seconds,
        "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 2**30,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    del warmup_iterator, loader, optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="data/genomic_benchmarks")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--datasets", nargs="*", default=list(GENOMIC_BENCHMARKS))
    parser.add_argument(
        "--output", default="runs/benchmarks/genomic_mixer_training.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_float32_matmul_precision("high")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for dataset_name in args.datasets:
        train_rows, _test_rows, sequence_key, label_key, _provenance = load_splits(
            dataset_name, "", args.cache_dir
        )
        max_length = max(len(str(sequence)) for sequence in train_rows[sequence_key])
        tokenizer = NucleotideTokenizer()
        full_train = StringSequenceDataset(
            train_rows,
            tokenizer,
            max_length,
            sequence_key=sequence_key,
            label_key=label_key,
        )
        train_indices, _ = stratified_split_indices(
            full_train.labels, 0.1, args.seed
        )
        train = IndexDataset(full_train, train_indices)
        num_classes = max(full_train.labels) + 1
        row = {
            "dataset": dataset_name,
            "max_length": max_length,
            "train_examples": len(train),
            "mixer_macs_per_sample": {
                mixer: DEPTH * mixer_macs(mixer, max_length)
                for mixer in ("rrlsso", "mha")
            },
            "timings": [],
        }
        for mixer in ("rrlsso", "mha"):
            timing = benchmark_epoch(
                mixer,
                train,
                tokenizer,
                max_length=max_length,
                num_classes=num_classes,
                workers=args.workers,
                warmup_steps=args.warmup_steps,
                seed=args.seed,
            )
            row["timings"].append(timing)
            print(json.dumps({"dataset": dataset_name, **timing}), flush=True)
        results.append(row)
        output.write_text(
            json.dumps(
                {
                    "device": torch.cuda.get_device_name(),
                    "mathdx_available": is_mathdx_available(),
                    "mathdx_error": None if is_mathdx_available() else str(mathdx_load_error()),
                    "mac_convention": "one multiply-accumulate is one MAC; elementwise ops excluded",
                    "results": results,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
