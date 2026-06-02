from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*", default=["runs/*.jsonl"])
    return parser.parse_args()


def iter_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matched = sorted(Path().glob(pattern))
        if matched:
            paths.extend(matched)
        else:
            path = Path(pattern)
            if path.exists():
                paths.append(path)
    return paths


def main() -> None:
    args = parse_args()
    rows = []
    for path in iter_paths(args.runs):
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if not lines or "args" not in lines[0]:
            continue
        run_args = lines[0]["args"]
        epochs = [row for row in lines[1:] if "epoch" in row]
        if not epochs:
            continue
        best = max(epochs, key=lambda row: row.get("eval_acc", float("-inf")))
        last = epochs[-1]
        rows.append(
            {
                "file": str(path),
                "dataset": run_args.get("dataset"),
                "mixer": run_args.get("mixer"),
                "rank": run_args.get("rank"),
                "gamma_max": run_args.get("gamma_max"),
                "theta_gamma_init": run_args.get("theta_gamma_init"),
                "dim": run_args.get("dim"),
                "depth": run_args.get("depth"),
                "heads": run_args.get("num_heads"),
                "seed": run_args.get("seed"),
                "best_epoch": best["epoch"],
                "best_acc": best.get("eval_acc"),
                "last_acc": last.get("eval_acc"),
                "last_loss": last.get("eval_loss"),
                "last_gamma_over_mu": last.get("eval_diag_gamma_over_mu"),
                "last_correction_ratio": last.get("eval_diag_correction_ratio"),
                "last_effective_rank": last.get("eval_diag_effective_rank"),
            }
        )

    rows.sort(key=lambda row: (row["best_acc"] is not None, row["best_acc"]), reverse=True)
    print(
        "best_acc\tlast_acc\tgamma_max\ttheta\tcorr\teff_rank\tgamma_over_mu\tfile"
    )
    for row in rows:
        print(
            f"{row['best_acc']:.4f}\t"
            f"{row['last_acc']:.4f}\t"
            f"{row['gamma_max']}\t"
            f"{row['theta_gamma_init']}\t"
            f"{fmt(row['last_correction_ratio'])}\t"
            f"{fmt(row['last_effective_rank'])}\t"
            f"{fmt(row['last_gamma_over_mu'])}\t"
            f"{row['file']}"
        )


def fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4g}"


if __name__ == "__main__":
    main()
