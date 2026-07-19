"""Controlled CIFAR-100 search for trace-normalized RRLSSO strength.

The sweep changes only the per-head ``alpha`` initialization.  Every child
process starts from the same seed, gain, model, data recipe, and scheduler so
the ten-epoch curves are directly paired.  Completed runs are reused.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "experiments" / "cv_vit_rrlsso_cifar100.py"


def _tag(value: float) -> str:
    return f"{value:.8g}".replace("-", "m").replace(".", "p")


def _read_result(path: Path, epochs: int) -> dict[str, float] | None:
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < epochs or int(rows[epochs - 1]["epoch"]) != epochs:
        return None
    rows = rows[:epochs]
    validation = [float(row["val_acc"]) for row in rows]
    return {
        "final_val_acc": validation[-1],
        "best_val_acc": max(validation),
        "mean_val_acc": sum(validation) / len(validation),
        "peak_mem_gb": max(float(row["peak_mem_gb"]) for row in rows),
        "train_seconds": sum(float(row["epoch_sec"]) for row in rows),
        "final_alpha_mean": float(rows[-1]["gamma_over_mu_mean"]),
        "final_alpha_min": float(rows[-1]["gamma_over_mu_min"]),
        "final_alpha_max": float(rows[-1]["gamma_over_mu_max"]),
    }


def _write_summary(path: Path, rows: list[dict[str, float | str]]) -> None:
    ranked = sorted(
        rows,
        key=lambda row: (float(row["final_val_acc"]), float(row["best_val_acc"])),
        reverse=True,
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranked[0]))
        writer.writeheader()
        writer.writerows(ranked)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace-normalized RRLSSO alpha initialization sweep."
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.8, 1.0776072417497349, 1.35, 1.6, 1.85],
    )
    parser.add_argument("--gain", type=float, default=1.4426742274994273)
    parser.add_argument("--alpha-max", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data-dir", default="/root/LSSO-data/cifar100")
    parser.add_argument(
        "--data-archive", default="/mnt/d/LSSO-data/cifar-100-python.tar.gz"
    )
    parser.add_argument(
        "--output-root", default="runs/cifar100_trace_alpha_init_10ep"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.epochs < 10:
        raise ValueError("initialization screening requires at least 10 epochs")
    if args.gain <= 0:
        raise ValueError("gain must be positive")
    if any(not 0 < alpha < args.alpha_max for alpha in args.alphas):
        raise ValueError(f"every alpha must lie in (0, {args.alpha_max})")
    if len(set(args.alphas)) != len(args.alphas):
        raise ValueError("alpha initializations must be unique")

    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "sweep_config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8"
    )
    results: list[dict[str, float | str]] = []
    for alpha in args.alphas:
        run_name = f"alpha_{_tag(alpha)}"
        run_root = output_root / run_name
        metrics = run_root / "rrlsso" / "metrics.csv"
        result = None if args.force else _read_result(metrics, args.epochs)
        if result is None:
            command = [
                sys.executable,
                str(TRAIN),
                "--models", "rrlsso",
                "--dataset", "cifar100",
                "--backbone", "torchvision-vit-b",
                "--data-dir", args.data_dir,
                "--data-archive", args.data_archive,
                "--out-dir", str(run_root),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--eval-batch-size", str(args.eval_batch_size),
                "--num-workers", str(args.workers),
                "--image-size", "32",
                "--patch-size", "4",
                "--rank", "32",
                "--gain-init", str(args.gain),
                "--alpha-init", str(alpha),
                "--alpha-max", str(args.alpha_max),
                "--solve-parameterization", "gain_alpha",
                "--basis-normalization", "trace",
                "--length-normalize",
                "--length-reference", "1.0",
                "--seed", str(args.seed),
                "--no-save-checkpoints",
                "--no-auto-resume",
            ]
            print(
                json.dumps(
                    {"status": "starting", "alpha_init": alpha, "command": command}
                ),
                flush=True,
            )
            subprocess.run(command, cwd=ROOT, check=True)
            result = _read_result(metrics, args.epochs)
            if result is None:
                raise RuntimeError(f"incomplete run: {metrics}")
        row: dict[str, float | str] = {
            "run": run_name,
            "gain_init": args.gain,
            "alpha_init": alpha,
            **result,
        }
        results.append(row)
        _write_summary(output_root / "summary.csv", results)
        print(json.dumps(row, sort_keys=True), flush=True)

    ranked = sorted(
        results,
        key=lambda row: (float(row["final_val_acc"]), float(row["best_val_acc"])),
        reverse=True,
    )
    print(json.dumps({"status": "complete", "ranking": ranked}, indent=2), flush=True)


if __name__ == "__main__":
    main()
