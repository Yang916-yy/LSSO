"""Collect benchmark outputs and compute seed-level and dataset-level summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/sequence")
    parser.add_argument("--output", default="runs/sequence/summary")
    parser.add_argument(
        "--reported-baselines",
        default="",
        help="optional CSV with suite,dataset,model,accuracy columns (accuracy in [0,1])",
    )
    parser.add_argument("--expected-seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--strict-completeness", action="store_true")
    return parser.parse_args()


def read_local_runs(root: Path) -> list[dict]:
    rows = []
    for metrics_path in sorted(root.glob("*/test_metrics.json")):
        config_path = metrics_path.parent / "config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "suite": config["suite"],
                "dataset": config["dataset"],
                "model": config["mixer"],
                "seed": int(config["seed"]),
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "matthews_correlation": float(metrics.get("matthews_correlation", 0.0)),
                "parameters": int(metrics["parameters"]),
                "selected_epoch": int(metrics["selected_epoch"]),
                "train_seconds": float(metrics.get("train_seconds", 0.0)),
                "peak_gb": float(metrics.get("peak_gb", 0.0)),
                "examples_per_second": float(metrics.get("mean_examples_per_second", 0.0)),
                "source": "local",
                "run": str(metrics_path.parent),
            }
        )
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["suite"], row["dataset"], row["model"])].append(row)
    result = []
    for (suite, dataset, model), values in sorted(grouped.items()):
        accuracies = [row["accuracy"] for row in values]
        f1s = [row["macro_f1"] for row in values]
        correlations = [row.get("matthews_correlation", 0.0) for row in values]
        result.append(
            {
                "suite": suite,
                "dataset": dataset,
                "model": model,
                "runs": len(values),
                "accuracy_mean": mean(accuracies),
                "accuracy_std": stdev(accuracies) if len(accuracies) > 1 else 0.0,
                "macro_f1_mean": mean(f1s),
                "macro_f1_std": stdev(f1s) if len(f1s) > 1 else 0.0,
                "matthews_correlation_mean": mean(correlations),
                "matthews_correlation_std": (
                    stdev(correlations) if len(correlations) > 1 else 0.0
                ),
                "parameters": values[0]["parameters"],
                "train_seconds_mean": mean(row.get("train_seconds", 0.0) for row in values),
                "peak_gb_max": max(row.get("peak_gb", 0.0) for row in values),
                "examples_per_second_mean": mean(
                    row.get("examples_per_second", 0.0) for row in values
                ),
            }
        )
    return result


def read_reported(path: Path) -> list[dict]:
    if not path:
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "suite", "dataset", "model", "accuracy", "source_url", "source_table"
    }
    if rows and not required.issubset(rows[0]):
        raise ValueError(f"reported baseline CSV needs columns {sorted(required)}")
    result = []
    for row in rows:
        accuracy = float(row["accuracy"])
        if not 0.0 <= accuracy <= 1.0:
            raise ValueError("reported baseline accuracy must be in [0, 1]")
        result.append({**row, "accuracy": accuracy})
    return result


def completeness(rows: list[dict], expected_seeds: set[int]) -> list[dict]:
    grouped = defaultdict(set)
    for row in rows:
        grouped[(row["suite"], row["dataset"], row["model"])].add(row["seed"])
    result = []
    for (suite, dataset, model), seeds in sorted(grouped.items()):
        missing = sorted(expected_seeds - seeds)
        result.append(
            {
                "suite": suite,
                "dataset": dataset,
                "model": model,
                "observed_seeds": " ".join(map(str, sorted(seeds))),
                "missing_seeds": " ".join(map(str, missing)),
                "complete": not missing,
            }
        )
    return result


def pairwise_wins(local: list[dict], reported: list[dict]) -> list[dict]:
    scores = defaultdict(dict)
    for row in local:
        scores[(row["suite"], row["dataset"])][row["model"]] = row["accuracy_mean"]
    for row in reported:
        scores[(row["suite"], row["dataset"])][row["model"]] = row["accuracy"]
    suites = defaultdict(dict)
    for (suite, dataset), values in scores.items():
        suites[suite][dataset] = values
    output = []
    for suite, datasets in sorted(suites.items()):
        models = sorted({model for values in datasets.values() for model in values})
        for first, second in combinations(models, 2):
            common = [values for values in datasets.values() if first in values and second in values]
            wins = sum(values[first] > values[second] + 1e-12 for values in common)
            losses = sum(values[second] > values[first] + 1e-12 for values in common)
            output.append(
                {
                    "suite": suite,
                    "model": first,
                    "opponent": second,
                    "datasets": len(common),
                    "wins": wins,
                    "ties": len(common) - wins - losses,
                    "losses": losses,
                }
            )
    return output


def critical_difference_plots(rank_rows: list[dict], output: Path, alpha: float = 0.05) -> None:
    if not rank_rows:
        return
    import matplotlib.pyplot as plt
    from scipy.stats import studentized_range

    grouped = defaultdict(list)
    for row in rank_rows:
        grouped[row["suite"]].append(row)
    for suite, rows in grouped.items():
        if len(rows) < 2 or rows[0]["complete_datasets"] < 2:
            continue
        count = len(rows)
        datasets = rows[0]["complete_datasets"]
        q_alpha = studentized_range.ppf(1.0 - alpha, count, math.inf) / math.sqrt(2.0)
        cd = q_alpha * math.sqrt(count * (count + 1) / (6.0 * datasets))
        figure, axis = plt.subplots(figsize=(max(6, count * 1.2), 2.8))
        axis.set_xlim(0.8, count + 0.2)
        axis.set_ylim(-0.4, 1.0)
        axis.set_yticks([])
        axis.set_xlabel("mean rank (lower is better)")
        for index, row in enumerate(sorted(rows, key=lambda item: item["mean_rank"])):
            y = 0.55 if index % 2 == 0 else 0.25
            axis.plot(row["mean_rank"], y, "o", color="black")
            axis.text(row["mean_rank"], y + 0.11, row["model"], ha="center", fontsize=9)
        axis.plot([1.0, 1.0 + cd], [0.88, 0.88], color="black", linewidth=2)
        axis.text(1.0 + cd / 2, 0.93, f"CD={cd:.2f}", ha="center", fontsize=9)
        axis.set_title(f"{suite}: Nemenyi critical difference, alpha={alpha}")
        figure.tight_layout()
        figure.savefig(output / f"{suite}-critical-difference.png", dpi=180)
        plt.close(figure)


def average_ranks(local: list[dict], reported: list[dict]) -> list[dict]:
    scores = defaultdict(dict)
    for row in local:
        scores[(row["suite"], row["dataset"])][row["model"]] = row["accuracy_mean"]
    for row in reported:
        key = (row["suite"], row["dataset"])
        if row["model"] in scores[key]:
            raise ValueError(f"duplicate score for {key} and model={row['model']}")
        scores[key][row["model"]] = row["accuracy"]

    by_suite = defaultdict(dict)
    for (suite, dataset), values in scores.items():
        by_suite[suite][dataset] = values
    output = []
    for suite, datasets in sorted(by_suite.items()):
        model_sets = [set(values) for values in datasets.values()]
        complete_models = set.intersection(*model_sets) if model_sets else set()
        ranks = defaultdict(list)
        for values in datasets.values():
            ordered = sorted(complete_models, key=lambda model: values[model], reverse=True)
            position = 1
            while position <= len(ordered):
                end = position
                score = values[ordered[position - 1]]
                while end < len(ordered) and math.isclose(
                    values[ordered[end]], score, rel_tol=0.0, abs_tol=1e-12
                ):
                    end += 1
                rank = (position + end) / 2
                for model in ordered[position - 1 : end]:
                    ranks[model].append(rank)
                position = end + 1
        for model, values in sorted(ranks.items(), key=lambda item: mean(item[1])):
            output.append(
                {
                    "suite": suite,
                    "model": model,
                    "complete_datasets": len(values),
                    "mean_rank": mean(values),
                }
            )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    local = read_local_runs(Path(args.root))
    summary = aggregate(local)
    reported = read_reported(Path(args.reported_baselines)) if args.reported_baselines else []
    ranks = average_ranks(summary, reported)
    completion = completeness(local, set(args.expected_seeds))
    wins = pairwise_wins(summary, reported)
    output = Path(args.output)
    write_csv(output / "runs.csv", local)
    write_csv(output / "datasets.csv", summary)
    write_csv(output / "mean_ranks.csv", ranks)
    write_csv(output / "completeness.csv", completion)
    write_csv(output / "wins_ties_losses.csv", wins)
    critical_difference_plots(ranks, output)
    incomplete = sum(not row["complete"] for row in completion)
    print(json.dumps({
        "runs": len(local), "datasets": len(summary), "incomplete": incomplete,
        "output": str(output),
    }))
    if args.strict_completeness and incomplete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
