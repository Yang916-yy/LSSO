"""Run the frozen Motif-7 GenomicBenchmarks recipe for RRLSSO and MHA.

The launcher reuses existing RRLSSO-DNA formal directories, fills only missing
tasks, and then runs the paired MHA jobs sequentially. Completed
``test_metrics.json`` files are always skipped, so restarting the launcher is
safe. A partial comparison is refreshed after every completed job.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program-root", default="runs/rrlsso_dna_program")
    parser.add_argument("--cache-dir", default="data/genomic_benchmarks")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=(),
        help="run several seeds sequentially; overrides --seed",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--mixers",
        nargs="+",
        choices=("rrlsso", "mha"),
        default=("rrlsso", "mha"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun-complete", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def output_for(root: Path, dataset: str, mixer: str, architecture: str, seed: int) -> Path:
    suffix = f"genomic-{dataset}-{mixer}-dna-{architecture}-s{seed}"
    return root / ("formal" if mixer == "rrlsso" else "formal_mha") / suffix


def command_for(
    args: argparse.Namespace,
    frozen: dict,
    dataset: str,
    mixer: str,
    output: Path,
    seed: int,
) -> list[str]:
    size = frozen["size"]
    recipe = frozen["recipe"]
    command = [
        sys.executable,
        str(ROOT / "experiments" / "genomic_benchmarks.py"),
        "--dataset", dataset,
        "--cache-dir", args.cache_dir,
        "--output", str(output),
        "--mixer", mixer,
        "--workers", str(args.workers),
        "--seed", str(seed),
        "--dim", str(size["dim"]),
        "--depth", str(size["depth"]),
        "--heads", str(size["heads"]),
        "--rank", str(size["rank"]),
        "--max-parameters", str(size["max_parameters"]),
        "--pooling", str(recipe["pooling"]),
        "--reverse-complement-probability",
        str(recipe["reverse_complement_probability"]),
        "--local-motif-kernel", str(frozen["local_motif_kernel"]),
        "--training-profile", str(recipe["training_profile"]),
    ]
    if recipe.get("posthoc_rc_eval", False):
        command.append("--posthoc-rc-eval")
    if args.data_root:
        command += ["--data-root", args.data_root]
    return command


def refresh_summary(args: argparse.Namespace, frozen: dict, seed: int) -> dict:
    root = Path(args.program_root)
    architecture = frozen["architecture"]
    rows = []
    by_task: dict[str, dict[str, dict]] = {}
    for dataset in frozen["formal_tasks"]:
        task = {}
        for mixer in ("rrlsso", "mha"):
            result_path = (
                output_for(root, dataset, mixer, architecture, seed)
                / "test_metrics.json"
            )
            if result_path.is_file():
                task[mixer] = json.loads(result_path.read_text(encoding="utf-8"))
        by_task[dataset] = task
        if "rrlsso" in task and "mha" in task:
            rows.append(
                {
                    "dataset": dataset,
                    "rrlsso_accuracy": task["rrlsso"]["accuracy"],
                    "mha_accuracy": task["mha"]["accuracy"],
                    "accuracy_delta": task["rrlsso"]["accuracy"] - task["mha"]["accuracy"],
                    "rrlsso_macro_f1": task["rrlsso"]["macro_f1"],
                    "mha_macro_f1": task["mha"]["macro_f1"],
                    "rrlsso_mcc": task["rrlsso"]["matthews_correlation"],
                    "mha_mcc": task["mha"]["matthews_correlation"],
                }
            )
    summary = {
        "seed": seed,
        "paired_tasks": len(rows),
        "expected_tasks": len(frozen["formal_tasks"]),
        "mean_rrlsso_accuracy": (
            sum(row["rrlsso_accuracy"] for row in rows) / len(rows) if rows else None
        ),
        "mean_mha_accuracy": (
            sum(row["mha_accuracy"] for row in rows) / len(rows) if rows else None
        ),
        "mean_accuracy_delta": (
            sum(row["accuracy_delta"] for row in rows) / len(rows) if rows else None
        ),
        "tasks": by_task,
        "updated_unix": time.time(),
    }
    atomic_json(root / f"paired_seed{seed}_summary.json", summary)
    csv_path = root / f"paired_seed{seed}_summary.csv"
    if rows:
        temporary = csv_path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(csv_path)
    return summary


def refresh_multiseed_summary(
    args: argparse.Namespace, frozen: dict, seeds: tuple[int, ...]
) -> dict:
    root = Path(args.program_root)
    architecture = frozen["architecture"]
    detailed_rows = []
    task_summary = {}
    for dataset in frozen["formal_tasks"]:
        task_runs = []
        for seed in seeds:
            results = {}
            for mixer in ("rrlsso", "mha"):
                result_path = (
                    output_for(root, dataset, mixer, architecture, seed)
                    / "test_metrics.json"
                )
                if result_path.is_file():
                    results[mixer] = json.loads(result_path.read_text(encoding="utf-8"))
            if "rrlsso" not in results or "mha" not in results:
                continue
            row = {"dataset": dataset, "seed": seed}
            for metric, key in (
                ("accuracy", "accuracy"),
                ("macro_f1", "macro_f1"),
                ("mcc", "matthews_correlation"),
            ):
                for mixer in ("rrlsso", "mha"):
                    row[f"{mixer}_{metric}"] = results[mixer][key]
                row[f"{metric}_delta"] = (
                    row[f"rrlsso_{metric}"] - row[f"mha_{metric}"]
                )
            detailed_rows.append(row)
            task_runs.append(row)
        metrics = {}
        for metric in ("accuracy", "macro_f1", "mcc"):
            for mixer in ("rrlsso", "mha"):
                values = [row[f"{mixer}_{metric}"] for row in task_runs]
                metrics[f"{mixer}_{metric}_mean"] = mean(values) if values else None
                metrics[f"{mixer}_{metric}_std"] = (
                    stdev(values) if len(values) > 1 else (0.0 if values else None)
                )
            deltas = [row[f"{metric}_delta"] for row in task_runs]
            metrics[f"{metric}_delta_mean"] = mean(deltas) if deltas else None
            metrics[f"{metric}_delta_std"] = (
                stdev(deltas) if len(deltas) > 1 else (0.0 if deltas else None)
            )
        task_summary[dataset] = {"paired_seeds": len(task_runs), **metrics}

    summary = {
        "seeds": list(seeds),
        "completed_pairs": len(detailed_rows),
        "expected_pairs": len(frozen["formal_tasks"]) * len(seeds),
        "mean_rrlsso_accuracy": (
            mean(row["rrlsso_accuracy"] for row in detailed_rows)
            if detailed_rows else None
        ),
        "mean_mha_accuracy": (
            mean(row["mha_accuracy"] for row in detailed_rows)
            if detailed_rows else None
        ),
        "mean_accuracy_delta": (
            mean(row["accuracy_delta"] for row in detailed_rows)
            if detailed_rows else None
        ),
        "tasks": task_summary,
        "updated_unix": time.time(),
    }
    seed_label = "_".join(str(seed) for seed in seeds)
    atomic_json(root / f"paired_seeds_{seed_label}_summary.json", summary)
    if detailed_rows:
        csv_path = root / f"paired_seeds_{seed_label}_runs.csv"
        temporary = csv_path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(detailed_rows[0]))
            writer.writeheader()
            writer.writerows(detailed_rows)
        temporary.replace(csv_path)
    return summary


def main() -> None:
    args = parse_args()
    root = Path(args.program_root)
    frozen_path = root / "frozen_config.json"
    if not frozen_path.is_file():
        raise FileNotFoundError(f"missing frozen configuration: {frozen_path}")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    architecture = frozen["architecture"]
    seeds = tuple(dict.fromkeys(args.seeds or (args.seed,)))
    report_seeds = tuple(dict.fromkeys(frozen.get("formal_seeds", seeds)))
    failures = []
    for mixer in args.mixers:
        for seed in seeds:
            for dataset in frozen["formal_tasks"]:
                output = output_for(root, dataset, mixer, architecture, seed)
                result = output / "test_metrics.json"
                if result.is_file() and not args.rerun_complete:
                    print(json.dumps({"status": "skipped-complete", "result": str(result)}), flush=True)
                    continue
                command = command_for(args, frozen, dataset, mixer, output, seed)
                print(json.dumps({"status": "starting", "command": command}), flush=True)
                if args.dry_run:
                    continue
                try:
                    subprocess.run(command, cwd=ROOT, check=True)
                    refresh_summary(args, frozen, seed)
                    refresh_multiseed_summary(args, frozen, report_seeds)
                except subprocess.CalledProcessError as error:
                    failure = {
                        "dataset": dataset,
                        "mixer": mixer,
                        "seed": seed,
                        "returncode": error.returncode,
                        "command": command,
                        "failed_unix": time.time(),
                    }
                    output.mkdir(parents=True, exist_ok=True)
                    atomic_json(output / "failed.json", failure)
                    failures.append(failure)
                    print(json.dumps({"status": "failed", **failure}), flush=True)
    if not args.dry_run:
        for seed in seeds:
            refresh_summary(args, frozen, seed)
        summary = refresh_multiseed_summary(args, frozen, report_seeds)
        print(json.dumps({"status": "summary", **summary}), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
