"""Archived cross-validated UEA refinement runs.

Epoch selection uses only folds of the official training split. The final model
is then reinitialized, trained on the complete official training split for the
median selected epoch count, and evaluated once on the official test split.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_DATASETS = ("AtrialFibrillation", "DuckDuckGeese", "EthanolConcentration")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--mixers", nargs="+", choices=("mha", "rrlsso"), default=("rrlsso", "mha"))
    parser.add_argument("--data-root", default="data/uea")
    parser.add_argument("--output-root", default="runs/uea_refinement")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-workers", type=int, default=0)
    parser.add_argument("--pooling", choices=("mean", "max", "meanmax"), default="meanmax")
    parser.add_argument("--local-stem-kernels", type=int, nargs="+", default=(3, 7, 15))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--wait-for-pid",
        type=int,
        default=0,
        help="wait for an existing benchmark runner before using the accelerator",
    )
    return parser.parse_args()


def run(command: list[str], *, dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    if args.wait_for_pid:
        while True:
            try:
                os.kill(args.wait_for_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(30)
    root = Path(args.output_root)
    script = Path(__file__).with_name("uea_benchmark.py")
    shared = [
        sys.executable,
        str(script),
        "--data-root", args.data_root,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--eval-batch-size", str(args.eval_batch_size),
        "--workers", str(args.workers),
        "--eval-workers", str(args.eval_workers),
        "--pooling", args.pooling,
        "--local-stem-kernels", *map(str, args.local_stem_kernels),
        "--seed", str(args.seed),
    ]
    for dataset in args.datasets:
        for mixer in args.mixers:
            run_root = root / f"{dataset}-{mixer}-s{args.seed}"
            selected_epochs: list[int] = []
            for fold in range(args.folds):
                fold_output = run_root / f"fold-{fold}"
                command = shared + [
                    "--dataset", dataset,
                    "--mixer", mixer,
                    "--output", str(fold_output),
                    "--validation-folds", str(args.folds),
                    "--validation-fold-index", str(fold),
                    "--no-test-evaluation",
                ]
                run(command, dry_run=args.dry_run)
                if not args.dry_run:
                    result = json.loads(
                        (fold_output / "validation_metrics.json").read_text(encoding="utf-8")
                    )
                    selected_epochs.append(int(result["selected_epoch"]) + 1)
            final_epochs = (
                max(1, round(statistics.median(selected_epochs)))
                if selected_epochs
                else args.epochs
            )
            final_output = run_root / "final"
            final_command = shared + [
                "--dataset", dataset,
                "--mixer", mixer,
                "--output", str(final_output),
                "--epochs", str(final_epochs),
                "--train-full",
                "--no-resume",
            ]
            run(final_command, dry_run=args.dry_run)
            if not args.dry_run:
                (run_root / "cv_selection.json").write_text(
                    json.dumps(
                        {
                            "fold_selected_epoch_counts": selected_epochs,
                            "final_epoch_count": final_epochs,
                            "rule": "rounded median of fold-selected epoch counts",
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )


if __name__ == "__main__":
    main()
