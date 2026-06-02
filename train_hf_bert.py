from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from examples.models.bert import iter_bert_lsso_layers, replace_bert_self_attention_with_lsso
from train_text import download_ag_news, download_imdb, read_ag_news, read_imdb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BERT or BERT-LSSO on text classification.")
    parser.add_argument("--dataset", choices=["ag_news", "imdb"], default="ag_news")
    parser.add_argument("--data-source", choices=["local", "hf"], default="local")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--mixer", choices=["mha", "lsso", "lsso-no-global"], default="lsso")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--gamma-max", type=float, default=0.3)
    parser.add_argument("--theta-gamma-init", type=float, default=-4.0)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-eval-examples", type=int, default=0)
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    num_labels = 4 if args.dataset == "ag_news" else 2
    tokenizer_name = args.tokenizer_name or args.model_name
    print("loading tokenizer", tokenizer_name, flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=True,
            local_files_only=args.local_files_only,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=False,
            local_files_only=args.local_files_only,
        )

    print("building dataset", args.data_source, args.dataset, flush=True)
    if args.data_source == "local":
        train_dataset, eval_dataset = build_local_datasets(args, tokenizer)
    else:
        train_dataset, eval_dataset = build_hf_datasets(args, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print("loading model", args.model_name, flush=True)
    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        config=config,
        local_files_only=args.local_files_only,
    )
    if args.mixer in {"lsso", "lsso-no-global"}:
        print("replacing BERT self-attention with LSSO", flush=True)
        replace_bert_self_attention_with_lsso(
            model,
            rank=args.rank,
            gamma_max=args.gamma_max,
            theta_gamma_init=args.theta_gamma_init,
            no_global=args.mixer == "lsso-no-global",
        )

    device = torch.device(args.device)
    model.to(device)
    print("model ready", device, flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    run_name = (
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_hfbert_{args.dataset}_{args.mixer}_"
        f"r{args.rank}_g{args.gamma_max}_tgi{args.theta_gamma_init}_len{args.max_len}_s{args.seed}"
    )
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"{run_name}.jsonl"
    ckpt_path = run_dir / f"{run_name}.pt"

    best_acc = 0.0
    with log_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"args": vars(args)}, sort_keys=True) + "\n")
        f.flush()
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
                use_amp=args.amp and device.type == "cuda",
                max_batches=args.max_train_batches,
                training=True,
            )
            with torch.no_grad():
                eval_metrics = run_epoch(
                    model,
                    eval_loader,
                    optimizer=None,
                    scaler=None,
                    device=device,
                    use_amp=args.amp and device.type == "cuda",
                    max_batches=args.max_eval_batches,
                    training=False,
                )
            eval_metrics.update(collect_lsso_diagnostics(model))
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


def build_hf_datasets(args: argparse.Namespace, tokenizer):
    from datasets import DatasetDict, load_dataset

    if args.dataset == "ag_news":
        raw = load_dataset("ag_news")
    elif args.dataset == "imdb":
        dataset = load_dataset("imdb")
        raw = DatasetDict(train=dataset["train"], test=dataset["test"])
    else:
        raise ValueError(f"unknown dataset: {args.dataset}")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_len,
        )

    tokenized = raw.map(tokenize, batched=True, remove_columns=["text"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")
    return tokenized["train"], tokenized["test"]


def build_local_datasets(args: argparse.Namespace, tokenizer):
    data_root = Path(args.data_dir)
    if args.dataset == "ag_news":
        train_path, test_path = download_ag_news(data_root)
        train_rows = read_ag_news(train_path)
        eval_rows = read_ag_news(test_path)
        cache_dir = data_root / "ag_news" / "hfbert"
    elif args.dataset == "imdb":
        root = download_imdb(data_root)
        train_rows = read_imdb(root, "train")
        eval_rows = read_imdb(root, "test")
        cache_dir = data_root / "imdb" / "hfbert"
    else:
        raise ValueError(f"unknown dataset: {args.dataset}")

    if args.max_train_examples:
        train_rows = train_rows[: args.max_train_examples]
    if args.max_eval_examples:
        eval_rows = eval_rows[: args.max_eval_examples]
    if args.max_train_batches and not args.max_train_examples:
        train_rows = train_rows[: args.max_train_batches * args.batch_size]
    if args.max_eval_batches and not args.max_eval_examples:
        eval_rows = eval_rows[: args.max_eval_batches * args.batch_size]

    tag = (
        f"{args.dataset}_{args.model_name.replace('/', '_')}_"
        f"{(args.tokenizer_name or args.model_name).replace('/', '_')}_"
        f"len{args.max_len}_train{len(train_rows)}_eval{len(eval_rows)}"
    )
    train_cache = cache_dir / f"{tag}_train.pt"
    eval_cache = cache_dir / f"{tag}_eval.pt"
    return (
        HFBertLocalDataset(train_rows, tokenizer, args.max_len, train_cache),
        HFBertLocalDataset(eval_rows, tokenizer, args.max_len, eval_cache),
    )


class HFBertLocalDataset(Dataset):
    def __init__(self, rows, tokenizer, max_len: int, cache_path: Path) -> None:
        self.cache_path = cache_path
        if cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu")
            self.encodings = {
                "input_ids": cached["input_ids"],
                "attention_mask": cached["attention_mask"],
            }
            self.labels = cached["labels"]
            return

        print("tokenizing", cache_path.name, len(rows), flush=True)
        texts = [text for text, _ in rows]
        labels = [label for _, label in rows]
        encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.encodings = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
        }
        self.labels = torch.tensor(labels, dtype=torch.long)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "input_ids": self.encodings["input_ids"],
                "attention_mask": self.encodings["attention_mask"],
                "labels": self.labels,
            },
            cache_path,
        )

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return {
            "input_ids": self.encodings["input_ids"][index],
            "attention_mask": self.encodings["attention_mask"][index],
            "labels": self.labels[index],
        }


def run_epoch(
    model,
    loader: DataLoader,
    optimizer,
    scaler,
    device: torch.device,
    use_amp: bool,
    max_batches: int,
    training: bool,
) -> dict[str, float]:
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    desc = "train" if training else "eval"

    for step, batch in enumerate(tqdm(loader, desc=desc, leave=False), start=1):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(**batch)
                loss = outputs.loss
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        logits = outputs.logits.detach()
        labels = batch["labels"]
        total_loss += loss.item() * labels.shape[0]
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += labels.shape[0]
        if max_batches and step >= max_batches:
            break

    return {"loss": total_loss / total_count, "acc": total_correct / total_count}


def collect_lsso_diagnostics(model) -> dict[str, float]:
    layers = iter_bert_lsso_layers(model)
    if not layers:
        return {}

    gamma_over_mu = []
    effective_rank = []
    correction_ratio = []
    for layer in layers:
        diag = layer.last_diagnostics
        if diag is None:
            continue
        gamma_over_mu.append(diag.gamma_over_mu.flatten())
        effective_rank.append(diag.effective_rank.flatten())
        correction_ratio.append(diag.correction_ratio.flatten())

    if not gamma_over_mu:
        return {}

    return {
        "diag_gamma_over_mu": torch.cat(gamma_over_mu).mean().item(),
        "diag_effective_rank": torch.cat(effective_rank).mean().item(),
        "diag_correction_ratio": torch.cat(correction_ratio).mean().item(),
    }


if __name__ == "__main__":
    main()
