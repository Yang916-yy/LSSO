from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.models.bertstyle import BertStyleEncoder
from examples.models.vit import VisionEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot trained LSSO diagnostics by layer.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-type", choices=["retrieval", "cifar100"], required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("paper_results/operator_diagnostics"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def build_model(payload: dict, task_type: str):
    args = payload["args"]
    if task_type == "retrieval":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args["tokenizer_name"], use_fast=True, local_files_only=True
        )
        model = BertStyleEncoder(
            vocab_size=len(tokenizer),
            num_classes=2,
            max_len=max(args["max_query_len"], args["max_doc_len"]),
            dim=args["dim"],
            depth=args["depth"],
            num_heads=args["num_heads"],
            mixer="lsso",
            rank=args["rank"],
            mlp_ratio=args["mlp_ratio"],
            dropout=args["dropout"],
            gamma_max=args["gamma_max"],
            theta_gamma_init=args["theta_gamma_init"],
            pad_id=tokenizer.pad_token_id or 0,
        )
        return model, tokenizer

    model = VisionEncoder(
        image_size=args["image_size"],
        patch_size=args["patch_size"],
        num_classes=100,
        dim=args["dim"],
        depth=args["depth"],
        num_heads=args["num_heads"],
        mixer="lsso",
        rank=args["rank"],
        mlp_ratio=args["mlp_ratio"],
        dropout=args["dropout"],
        gamma_max=args["gamma_max"],
        theta_gamma_init=args["theta_gamma_init"],
        normalize_u=not args.get("no_u_rms_norm", False),
    )
    return model, None


def retrieval_batches(task: str, tokenizer, max_len: int, batch_size: int, max_batches: int):
    from train_bertstyle_retrieval import join_title_text, load_beir_dataset

    corpus_ds, queries_ds, _, _ = load_beir_dataset(task, offline=True)
    texts = [join_title_text(row.get("title", ""), row.get("text", "")) for row in corpus_ds]
    limit = batch_size * max_batches
    for start in range(0, min(len(texts), limit), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        yield encoded["input_ids"]


def cifar_batches(data_dir: Path, batch_size: int, max_batches: int):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408),
                (0.2675, 0.2565, 0.2761),
            ),
        ]
    )
    dataset = datasets.CIFAR100(data_dir, train=False, transform=transform, download=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for step, (images, _) in enumerate(loader):
        if step >= max_batches:
            break
        yield images


def collect_layer_values(model) -> list[dict[str, float]]:
    rows = []
    for layer_idx, layer in enumerate(model.lsso_layers(), start=1):
        diag = layer.last_diagnostics
        if diag is None:
            raise RuntimeError(f"missing diagnostics for layer {layer_idx}")
        rows.append(
            {
                "layer": layer_idx,
                "gamma_over_mu": float(diag.gamma_over_mu.mean()),
                "correction_ratio": float(diag.correction_ratio.mean()),
                "effective_rank": float(diag.effective_rank.mean()),
            }
        )
    return rows


def plot(rows: list[dict[str, float]], task: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7), constrained_layout=True)
    metrics = [
        ("gamma_over_mu", r"$\gamma / \mu$"),
        ("correction_ratio", "Correction ratio"),
        ("effective_rank", "Effective rank"),
    ]
    for ax, (metric, label) in zip(axes, metrics):
        ax.plot(
            [row["layer"] for row in rows],
            [row[metric] for row in rows],
            marker="o",
            linewidth=2,
            color="#2457a7",
        )
        ax.set_xlabel("Layer")
        ax.set_ylabel(label)
        ax.set_xticks([row["layer"] for row in rows])
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"LSSO operator diagnostics: {task}", fontsize=13)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, tokenizer = build_model(payload, args.task_type)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    for layer in model.lsso_layers():
        layer.record_diagnostics = True

    accum: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    if args.task_type == "retrieval":
        batches = retrieval_batches(
            args.task,
            tokenizer,
            payload["args"]["max_doc_len"],
            args.batch_size,
            args.max_batches,
        )
    else:
        batches = cifar_batches(args.data_dir, args.batch_size, args.max_batches)

    count = 0
    with torch.no_grad():
        for batch in batches:
            batch = batch.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.bfloat16):
                if args.task_type == "retrieval":
                    model.forward_features(batch)
                else:
                    model(batch)
            for row in collect_layer_values(model):
                layer = row.pop("layer")
                for metric, value in row.items():
                    accum[layer][metric].append(value)
            count += 1

    if count == 0:
        raise RuntimeError("no diagnostic batches were produced")
    rows = []
    for layer in sorted(accum):
        rows.append(
            {
                "task": args.task,
                "checkpoint": args.checkpoint.name,
                "batches": count,
                "layer": layer,
                **{
                    metric: sum(values) / len(values)
                    for metric, values in accum[layer].items()
                },
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir / f"{args.task}_layer_diagnostics.tsv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, args.task, args.output_dir / f"{args.task}_layer_diagnostics.png")
    print(json.dumps({"task": args.task, "batches": count, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
