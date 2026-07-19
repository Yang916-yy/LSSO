from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "experiments" / "cv_vit_rrlsso_cifar100.py"


def _tag(value: float) -> str:
    return f"{value:.6g}".replace("-", "m").replace(".", "p")


def _read_result(path: Path, epochs: int) -> dict[str, float] | None:
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or int(rows[-1]["epoch"]) < epochs:
        return None
    values = [float(row["val_acc"]) for row in rows[:epochs]]
    return {
        "final_val_acc": values[-1],
        "best_val_acc": max(values),
        "mean_val_acc": sum(values) / len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Function-controlled CIFAR-100 gain/alpha initialization grid."
    )
    parser.add_argument("--gains", nargs="+", type=float, default=[1.0, 1.4426742275, 2.0])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.5, 1.0776072417, 1.5])
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
        "--output-root", default="runs/cifar100_solve_initialization_grid"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, float | str]] = []
    for gain, alpha in itertools.product(args.gains, args.alphas):
        if not 0 < alpha < args.alpha_max:
            raise ValueError(f"alpha={alpha} must lie in (0, {args.alpha_max})")
        run_name = f"g{_tag(gain)}_a{_tag(alpha)}"
        run_root = output_root / run_name
        metrics = run_root / "rrlsso" / "metrics.csv"
        result = None if args.force else _read_result(metrics, args.epochs)
        if result is None:
            command = [
                sys.executable,
                str(TRAIN),
                "--models", "rrlsso",
                "--dataset", "cifar100",
                "--backbone", "vision-llama-s",
                "--data-dir", args.data_dir,
                "--data-archive", args.data_archive,
                "--out-dir", str(run_root),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--eval-batch-size", str(args.eval_batch_size),
                "--num-workers", str(args.workers),
                "--rank", "32",
                "--gain-init", str(gain),
                "--alpha-init", str(alpha),
                "--alpha-max", str(args.alpha_max),
                "--solve-parameterization", "gain_alpha",
                "--seed", str(args.seed),
                "--no-save-checkpoints",
            ]
            print(f"\n=== {run_name}: gain={gain:.6g}, alpha={alpha:.6g} ===", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            result = _read_result(metrics, args.epochs)
            if result is None:
                raise RuntimeError(f"incomplete metrics after run: {metrics}")
        results.append({"run": run_name, "gain_init": gain, "alpha_init": alpha, **result})
        print(json.dumps(results[-1], sort_keys=True), flush=True)

    results.sort(key=lambda row: (row["final_val_acc"], row["mean_val_acc"]), reverse=True)
    summary = output_root / "summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nsummary={summary}")
    for row in results:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
