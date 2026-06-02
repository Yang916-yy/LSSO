from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from examples.models.bertstyle import BertStyleEncoder
from train_cifar import collect_lsso_diagnostics, set_lsso_diagnostics_enabled
from train_text import build_loaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a hand-written BERT-style MHA/LSSO classifier.")
    parser.add_argument("--dataset", choices=["ag_news", "imdb", "yahoo_answers"], default="ag_news")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument("--mixer", choices=["mha", "lsso", "lsso-no-global"], default="lsso")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--max-vocab", type=int, default=50000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gamma-max", type=float, default=0.3)
    parser.add_argument("--theta-gamma-init", type=float, default=-4.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
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


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == target).float().mean().item()


def run_epoch(
    model: nn.Module,
    loader,
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
    model = BertStyleEncoder(
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
    params = sum(p.numel() for p in model.parameters())
    print(
        f"dataset={args.dataset} train={len(train_loader.dataset)} eval={len(test_loader.dataset)} "
        f"vocab={vocab_size} classes={num_classes}",
        flush=True,
    )
    print(f"model params={params:,} device={device} amp={use_amp}", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run_name = (
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_bertstyle_"
        f"{args.dataset}_{args.mixer}_r{args.rank}_g{args.gamma_max}_tgi{args.theta_gamma_init}_"
        f"d{args.dim}_L{args.depth}_h{args.num_heads}_len{args.max_len}_s{args.seed}"
    )
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"{run_name}.jsonl"
    ckpt_path = run_dir / f"{run_name}.pt"

    best_acc = 0.0
    with log_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"args": vars(args), "vocab_size": vocab_size, "params": params}, sort_keys=True) + "\n")
        for epoch in range(1, args.epochs + 1):
            print(f"epoch {epoch}/{args.epochs} train", flush=True)
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
            print(f"epoch {epoch}/{args.epochs} eval", flush=True)
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
