"""Validation-gated RRLSSO-DNA recipe, scale, and formal experiment program.

The first two stages never evaluate the benchmark test split. They select a
reverse-complement/pooling recipe and record Tiny/Small/Base validation curves.
Only after ``frozen_config.json`` is written does the formal stage evaluate the
eight GenomicBenchmarks test splits.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.genomic_benchmarks import GENOMIC_BENCHMARKS


REPRESENTATIVE_TASKS = (
    "demo_human_or_worm",
    "human_nontata_promoters",
    "human_ocr_ensembl",
    "human_enhancers_cohn",
)


@dataclass(frozen=True)
class ModelSize:
    dim: int
    depth: int
    heads: int
    rank: int
    max_parameters: int


SIZES = {
    "tiny": ModelSize(128, 2, 4, 16, 750_000),
    "small": ModelSize(192, 2, 6, 32, 1_350_000),
    "base": ModelSize(256, 2, 8, 32, 2_200_000),
}

ENHANCEMENTS = {
    "none": {"probability": 0.0, "eval": False},
    "rc_train": {"probability": 0.5, "eval": False},
    "rc_train_eval": {"probability": 0.5, "eval": True},
}
POOLINGS = ("mean", "max", "meanmax")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="runs/rrlsso_dna_program")
    parser.add_argument("--cache-dir", default="data/genomic_benchmarks")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--screen-seed", type=int, default=0)
    parser.add_argument("--scale-seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--formal-seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--tie-margin",
        type=float,
        default=0.0025,
        help="prefer the simpler candidate when validation accuracy is within this margin",
    )
    parser.add_argument(
        "--stop-after", choices=("recipe", "scale", "formal"), default="formal"
    )
    parser.add_argument("--rerun-complete", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def common_command(args: argparse.Namespace, output: Path, dataset: str) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "experiments" / "genomic_benchmarks.py"),
        "--dataset", dataset,
        "--cache-dir", args.cache_dir,
        "--output", str(output),
        "--mixer", "rrlsso",
        "--workers", str(args.workers),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--eval-batch-size", str(args.eval_batch_size),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--warmup-ratio", str(args.warmup_ratio),
        "--dropout", str(args.dropout),
    ]
    if args.data_root:
        command += ["--data-root", args.data_root]
    return command


def model_arguments(size: ModelSize) -> list[str]:
    return [
        "--dim", str(size.dim),
        "--depth", str(size.depth),
        "--heads", str(size.heads),
        "--rank", str(size.rank),
        "--max-parameters", str(size.max_parameters),
    ]


def recipe_arguments(pooling: str, enhancement: dict) -> list[str]:
    arguments = [
        "--pooling", pooling,
        "--reverse-complement-probability", str(enhancement["probability"]),
    ]
    if enhancement["eval"]:
        arguments.append("--reverse-complement-eval")
    return arguments


def run_job(
    args: argparse.Namespace,
    *,
    output: Path,
    dataset: str,
    seed: int,
    size: ModelSize,
    pooling: str,
    enhancement: dict,
    validation_only: bool,
) -> dict:
    result_name = "validation_metrics.json" if validation_only else "test_metrics.json"
    result_path = output / result_name
    if result_path.exists() and not args.rerun_complete:
        print(json.dumps({"status": "skipped-complete", "result": str(result_path)}), flush=True)
        return json.loads(result_path.read_text(encoding="utf-8"))
    command = (
        common_command(args, output, dataset)
        + ["--seed", str(seed)]
        + model_arguments(size)
        + recipe_arguments(pooling, enhancement)
    )
    if validation_only:
        command.append("--validation-only")
    print(json.dumps({"status": "starting", "command": command}), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    return json.loads(result_path.read_text(encoding="utf-8"))


def mean_accuracy(results: dict[str, dict]) -> float:
    return sum(value["accuracy"] for value in results.values()) / len(results)


def choose_with_margin(
    scores: dict[str, float], preference: tuple[str, ...], margin: float
) -> str:
    best = max(scores.values())
    eligible = {name for name, score in scores.items() if score >= best - margin}
    return next(name for name in preference if name in eligible)


def screen_recipe(args: argparse.Namespace, root: Path) -> dict:
    tiny = SIZES["tiny"]
    enhancement_results: dict[str, dict[str, dict]] = {}
    for name, enhancement in ENHANCEMENTS.items():
        by_task = {}
        for dataset in REPRESENTATIVE_TASKS:
            output = root / "recipe" / "enhancement" / name / f"{dataset}-s{args.screen_seed}"
            by_task[dataset] = run_job(
                args, output=output, dataset=dataset, seed=args.screen_seed,
                size=tiny, pooling="mean", enhancement=enhancement,
                validation_only=True,
            )
        enhancement_results[name] = by_task
    enhancement_scores = {
        name: mean_accuracy(results) for name, results in enhancement_results.items()
    }
    selected_enhancement = choose_with_margin(
        enhancement_scores, tuple(ENHANCEMENTS), args.tie_margin
    )

    pooling_results = {"mean": enhancement_results[selected_enhancement]}
    enhancement = ENHANCEMENTS[selected_enhancement]
    for pooling in ("max", "meanmax"):
        by_task = {}
        for dataset in REPRESENTATIVE_TASKS:
            output = root / "recipe" / "pooling" / pooling / f"{dataset}-s{args.screen_seed}"
            by_task[dataset] = run_job(
                args, output=output, dataset=dataset, seed=args.screen_seed,
                size=tiny, pooling=pooling, enhancement=enhancement,
                validation_only=True,
            )
        pooling_results[pooling] = by_task
    pooling_scores = {
        name: mean_accuracy(results) for name, results in pooling_results.items()
    }
    selected_pooling = choose_with_margin(pooling_scores, POOLINGS, args.tie_margin)
    selection = {
        "protocol": {
            "tasks": list(REPRESENTATIVE_TASKS),
            "seed": args.screen_seed,
            "metric": "validation_accuracy_macro_over_tasks",
            "tie_margin": args.tie_margin,
            "test_evaluated": False,
        },
        "enhancement_scores": enhancement_scores,
        "selected_enhancement": selected_enhancement,
        "pooling_scores": pooling_scores,
        "selected_pooling": selected_pooling,
        "selected_recipe": {
            "pooling": selected_pooling,
            **ENHANCEMENTS[selected_enhancement],
        },
    }
    atomic_json(root / "recipe_selection.json", selection)
    print(json.dumps({"recipe_selection": selection}, sort_keys=True), flush=True)
    return selection


def run_scale(args: argparse.Namespace, root: Path, selection: dict) -> dict:
    recipe = selection["selected_recipe"]
    enhancement = {"probability": recipe["probability"], "eval": recipe["eval"]}
    results = {}
    for name, size in SIZES.items():
        by_run = {}
        for dataset in REPRESENTATIVE_TASKS:
            for seed in args.scale_seeds:
                key = f"{dataset}-s{seed}"
                output = root / "scale" / name / key
                by_run[key] = run_job(
                    args, output=output, dataset=dataset, seed=seed, size=size,
                    pooling=recipe["pooling"], enhancement=enhancement,
                    validation_only=True,
                )
        results[name] = by_run
    summary = {
        name: {
            "validation_accuracy": mean_accuracy(values),
            "runs": len(values),
            "size": asdict(SIZES[name]),
        }
        for name, values in results.items()
    }
    scale = {
        "protocol": {
            "tasks": list(REPRESENTATIVE_TASKS),
            "seeds": list(args.scale_seeds),
            "metric": "validation_accuracy_macro_over_task_seed_runs",
            "recipe_frozen": selection["selected_recipe"],
            "test_evaluated": False,
        },
        "summary": summary,
    }
    atomic_json(root / "scale_results.json", scale)
    print(json.dumps({"scale_results": scale}, sort_keys=True), flush=True)
    return scale


def freeze_base(args: argparse.Namespace, root: Path, selection: dict, scale: dict) -> dict:
    frozen = {
        "model": "RRLSSO-DNA-Base",
        "size": asdict(SIZES["base"]),
        "recipe": selection["selected_recipe"],
        "training": {
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "dropout": args.dropout,
        },
        "formal_tasks": list(GENOMIC_BENCHMARKS),
        "formal_seeds": list(args.formal_seeds),
        "selection_source": {
            "recipe": "recipe_selection.json",
            "scale": "scale_results.json",
        },
        "test_evaluated_at_freeze": False,
        "frozen_unix": time.time(),
    }
    path = root / "frozen_config.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable = {key: value for key, value in existing.items() if key != "frozen_unix"}
        requested = {key: value for key, value in frozen.items() if key != "frozen_unix"}
        if comparable != requested:
            raise RuntimeError(
                f"refusing to change the already frozen formal configuration at {path}"
            )
        return existing
    atomic_json(path, frozen)
    return frozen


def run_formal(args: argparse.Namespace, root: Path, frozen: dict) -> None:
    size = ModelSize(**frozen["size"])
    recipe = frozen["recipe"]
    enhancement = {"probability": recipe["probability"], "eval": recipe["eval"]}
    completed = 0
    for dataset in frozen["formal_tasks"]:
        for seed in frozen["formal_seeds"]:
            output = root / "formal" / f"genomic-{dataset}-rrlsso-dna-base-s{seed}"
            run_job(
                args, output=output, dataset=dataset, seed=seed, size=size,
                pooling=recipe["pooling"], enhancement=enhancement,
                validation_only=False,
            )
            completed += 1
    atomic_json(
        root / "formal_complete.json",
        {"completed_runs": completed, "expected_runs": completed, "completed_unix": time.time()},
    )


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    selection_path = root / "recipe_selection.json"
    selection = (
        json.loads(selection_path.read_text(encoding="utf-8"))
        if selection_path.exists() and not args.rerun_complete
        else screen_recipe(args, root)
    )
    if args.stop_after == "recipe":
        return
    scale_path = root / "scale_results.json"
    scale = (
        json.loads(scale_path.read_text(encoding="utf-8"))
        if scale_path.exists() and not args.rerun_complete
        else run_scale(args, root, selection)
    )
    if args.stop_after == "scale":
        return
    frozen = freeze_base(args, root, selection, scale)
    run_formal(args, root, frozen)


if __name__ == "__main__":
    main()
