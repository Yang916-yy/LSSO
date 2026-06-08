from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine per-task LSSO layer diagnostics into a paper figure."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("paper_results/operator_diagnostics"),
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [
            {
                **row,
                "layer": int(row["layer"]),
                "gamma_over_mu": float(row["gamma_over_mu"]),
                "correction_ratio": float(row["correction_ratio"]),
                "effective_rank": float(row["effective_rank"]),
            }
            for row in csv.DictReader(handle, delimiter="\t")
        ]


def write_summary(
    tasks: dict[str, list[dict[str, object]]],
    path: Path,
) -> None:
    metrics = ("gamma_over_mu", "correction_ratio", "effective_rank")
    rows = []
    for task, task_rows in tasks.items():
        row: dict[str, object] = {
            "task": task,
            "layers": len(task_rows),
            "checkpoint": task_rows[0]["checkpoint"],
            "batches": task_rows[0]["batches"],
        }
        for metric in metrics:
            values = [float(item[metric]) for item in task_rows]
            row[f"{metric}_mean"] = sum(values) / len(values)
            row[f"{metric}_min"] = min(values)
            row[f"{metric}_max"] = max(values)
        rows.append(row)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def plot(tasks: dict[str, list[dict[str, object]]], path: Path) -> None:
    colors = {
        "fiqa": "#2457a7",
        "nfcorpus": "#d1495b",
        "scifact": "#2a9d8f",
        "cifar100": "#8a5fbf",
    }
    metrics = [
        ("gamma_over_mu", r"$\gamma / \mu$"),
        ("correction_ratio", "Correction ratio"),
        ("effective_rank", "Effective rank"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
    for ax, (metric, label) in zip(axes, metrics):
        for task, rows in tasks.items():
            ax.plot(
                [row["layer"] for row in rows],
                [row[metric] for row in rows],
                marker="o",
                linewidth=2,
                markersize=4,
                label=task,
                color=colors.get(task),
            )
        ax.set_xlabel("Layer")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Learned LSSO operator diagnostics", fontsize=14)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_dir.glob("*_layer_diagnostics.tsv"))
    if not paths:
        raise RuntimeError(f"no layer diagnostic tables found in {args.input_dir}")
    tasks = {
        path.name.removesuffix("_layer_diagnostics.tsv"): read_rows(path)
        for path in paths
    }
    write_summary(tasks, args.input_dir / "summary.tsv")
    plot(tasks, args.input_dir / "operator_diagnostics_summary.png")


if __name__ == "__main__":
    main()
