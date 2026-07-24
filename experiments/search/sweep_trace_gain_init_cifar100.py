"""Controlled CIFAR-100 gain-initialization sweep for trace RRLSSO."""

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
        "final_alpha_observed_min": float(rows[-1]["gamma_over_mu_min"]),
        "final_alpha_observed_max": float(rows[-1]["gamma_over_mu_max"]),
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
        description="Trace-normalized RRLSSO gain initialization sweep."
    )
    parser.add_argument(
        "--gains",
        nargs="+",
        type=float,
        default=[0.75, 1.0, 1.25, 1.4426742274994273, 1.75, 2.0, 2.5],
    )
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
        "--output-root", default="runs/cifar100_trace_gain_init_10ep"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.epochs < 10:
        raise ValueError("initialization screening requires at least 10 epochs")
    if any(gain <= 0 for gain in args.gains):
        raise ValueError("every gain must be positive")
    if len(set(args.gains)) != len(args.gains):
        raise ValueError("gain initializations must be unique")

    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "sweep_config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8"
    )
    results: list[dict[str, float | str]] = []
    for gain in args.gains:
        run_name = f"gain_{_tag(gain)}"
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
                "--gain-init", str(gain),
                "--length-normalize",
                "--length-reference", "1.0",
                "--seed", str(args.seed),
                "--no-save-checkpoints",
                "--no-auto-resume",
            ]
            print(
                json.dumps(
                    {"status": "starting", "gain_init": gain, "command": command}
                ),
                flush=True,
            )
            subprocess.run(command, cwd=ROOT, check=True)
            result = _read_result(metrics, args.epochs)
            if result is None:
                raise RuntimeError(f"incomplete run: {metrics}")
        row: dict[str, float | str] = {
            "run": run_name,
            "gain_init": gain,
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
