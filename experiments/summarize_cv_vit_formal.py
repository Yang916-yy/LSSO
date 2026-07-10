from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_metrics(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a formal MHA vs RRLSSO CV run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--models", nargs="+", default=["mha", "rrlsso"])
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {model: _read_metrics(run_dir / model / "metrics.csv") for model in args.models}

    labels = {"mha": "MHA", "rrlsso": "RRLSSO-r32"}
    colors = {"mha": "#555555", "rrlsso": "#127a53"}
    rows: list[dict[str, float | int | str]] = []
    for model, series in metrics.items():
        with (run_dir / model / "config.json").open(encoding="utf-8") as handle:
            config = json.load(handle)
        best_acc = max(series, key=lambda row: row["val_acc"])
        best_loss = min(series, key=lambda row: row["val_loss"])
        final = series[-1]
        rows.append(
            {
                "model": model,
                "epochs": len(series),
                "params": int(config["params"]),
                "best_acc_epoch": int(best_acc["epoch"]),
                "best_val_acc": best_acc["val_acc"],
                "best_acc_val_loss": best_acc["val_loss"],
                "best_loss_epoch": int(best_loss["epoch"]),
                "best_val_loss": best_loss["val_loss"],
                "final_val_acc": final["val_acc"],
                "final_val_loss": final["val_loss"],
                "steady_epoch_sec": sum(row["epoch_sec"] for row in series[2:]) / max(1, len(series) - 2),
                "peak_mem_gb": max(row["peak_mem_gb"] for row in series),
                "final_gamma_over_mu_mean": final["gamma_over_mu_mean"],
                "final_gamma_over_mu_min": final["gamma_over_mu_min"],
                "final_gamma_over_mu_max": final["gamma_over_mu_max"],
            }
        )

    with (out_dir / "summary.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), sharex=True)
    for model, series in metrics.items():
        epochs = [row["epoch"] for row in series]
        axes[0].plot(epochs, [100.0 * row["val_acc"] for row in series], label=labels.get(model, model), color=colors.get(model))
        axes[1].plot(epochs, [row["val_loss"] for row in series], label=labels.get(model, model), color=colors.get(model))
    axes[0].set_ylabel("Validation accuracy (%)")
    axes[1].set_ylabel("Validation loss")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curves.png", dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
