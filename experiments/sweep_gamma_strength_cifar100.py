from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
from pathlib import Path


def _tag(theta: float) -> str:
    return f"theta_{theta:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p")


def _initial_alpha(gamma_max: float, theta: float, eps: float = 1e-5) -> float:
    gamma = gamma_max / (1.0 + math.exp(-theta))
    mu = math.log(2.0) + eps
    return gamma / mu


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep initial gamma/mu after effective-length normalization."
    )
    parser.add_argument("--model", choices=["rrlsso", "grouped-rrlsso"], default="grouped-rrlsso")
    parser.add_argument("--relation-groups", type=int, default=4)
    parser.add_argument("--theta", type=float, nargs="+", default=[-0.5, 0.0, 0.5, 1.0])
    parser.add_argument("--gamma-max", type=float, default=1.2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--steps-per-epoch", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup-epochs", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-dir", default="runs/gamma_strength_sweep/g4_default_sweep")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_root = root / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []

    for theta in args.theta:
        run_dir = out_root / _tag(theta)
        command = [
            sys.executable,
            str(root / "experiments" / "cv_vit_rrlsso_cifar100.py"),
            "--models",
            args.model,
            "--relation-groups",
            str(args.relation_groups),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--eval-batch-size",
            "256",
            "--dtype",
            "bf16",
            "--num-workers",
            "4",
            "--max-train-steps-per-epoch",
            str(args.steps_per_epoch),
            "--warmup-epochs",
            str(args.warmup_epochs),
            "--gamma-max",
            str(args.gamma_max),
            "--theta-gamma-init",
            str(theta),
            "--length-reference",
            "1",
            "--seed",
            str(args.seed),
            "--no-save-checkpoints",
            "--out-dir",
            str(run_dir),
        ]
        print(
            f"\n=== {args.model} G={args.relation_groups} theta={theta:g} "
            f"initial_alpha={_initial_alpha(args.gamma_max, theta):.6f} ===",
            flush=True,
        )
        subprocess.run(command, cwd=root, env=os.environ.copy(), check=True)

        metrics_path = run_dir / args.model / "metrics.csv"
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            metrics = list(csv.DictReader(handle))
        final = metrics[-1]
        best = min(metrics, key=lambda row: float(row["val_loss"]))
        rows.append(
            {
                "model": args.model,
                "relation_groups": args.relation_groups,
                "theta_gamma_init": theta,
                "gamma_max": args.gamma_max,
                "initial_gamma_over_mu": _initial_alpha(args.gamma_max, theta),
                "final_gamma_over_mu_mean": float(final["gamma_over_mu_mean"]),
                "final_gamma_over_mu_min": float(final["gamma_over_mu_min"]),
                "final_gamma_over_mu_max": float(final["gamma_over_mu_max"]),
                "final_train_loss": float(final["train_loss"]),
                "final_train_acc": float(final["train_acc"]),
                "final_val_loss": float(final["val_loss"]),
                "final_val_acc": float(final["val_acc"]),
                "best_val_loss": float(best["val_loss"]),
                "best_val_acc_at_best_loss": float(best["val_acc"]),
                "run_dir": str(run_dir.relative_to(root)),
            }
        )

        summary_path = out_root / "summary.tsv"
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

    ranked = sorted(rows, key=lambda row: float(row["best_val_loss"]))
    print("\n=== ranking by best validation loss ===")
    for row in ranked:
        print(
            f"theta={float(row['theta_gamma_init']):+.2f} "
            f"alpha0={float(row['initial_gamma_over_mu']):.5f} "
            f"alpha_final={float(row['final_gamma_over_mu_mean']):.5f} "
            f"val_loss={float(row['best_val_loss']):.4f} "
            f"val_acc={float(row['best_val_acc_at_best_loss']):.4f}"
        )


if __name__ == "__main__":
    main()
