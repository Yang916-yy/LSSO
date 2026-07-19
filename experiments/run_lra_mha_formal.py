"""Run the parameter-controlled MHA side of the four-task LRA comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


COMMON = [
    "--mixer", "mha",
    "--seed", "0",
    "--split-seed", "0",
    "--workers", "4",
    "--eval-workers", "0",
    "--position-rank", "0",
]

TASK_ARGUMENTS = {
    "text": [
        "--dim", "256", "--depth", "2", "--heads", "8", "--rank", "32",
        "--mlp-ratio", "4", "--dropout", "0.1", "--pooling", "mean",
        "--epochs", "32", "--batch-size", "32", "--eval-batch-size", "32",
        "--grad-accum", "1", "--lr", "0.0003", "--weight-decay", "0.01",
        "--warmup-ratio", "0.05", "--patience", "0",
    ],
    "listops": [
        "--dim", "256", "--depth", "2", "--heads", "8", "--rank", "32",
        "--mlp-ratio", "4", "--dropout", "0.1", "--pooling", "mean",
        "--epochs", "40", "--batch-size", "50", "--eval-batch-size", "50",
        "--grad-accum", "1", "--lr", "0.0003", "--weight-decay", "0.01",
        "--warmup-ratio", "0.05", "--patience", "0",
    ],
    "retrieval": [
        "--dim", "256", "--depth", "2", "--heads", "8", "--rank", "32",
        "--mlp-ratio", "4", "--dropout", "0.1", "--pooling", "mean",
        "--epochs", "20", "--batch-size", "64", "--eval-batch-size", "64",
        "--grad-accum", "1", "--lr", "0.0003", "--weight-decay", "0.01",
        "--warmup-ratio", "0.05", "--patience", "0",
    ],
    "pathfinder": [
        "--dim", "128", "--depth", "6", "--heads", "4", "--rank", "32",
        "--mlp-ratio", "2", "--dropout", "0", "--pooling", "meanmax",
        "--pathfinder-local-kernel", "3",
        "--pathfinder-local-dilations", "1", "1", "2", "2", "4", "4",
        "--pathfinder-local-layer-scale", "0.1",
        "--pathfinder-local-lr-multiplier", "5",
        "--epochs", "200", "--batch-size", "128", "--eval-batch-size", "256",
        "--grad-accum", "1", "--lr", "0.0002", "--weight-decay", "0.01",
        "--warmup-ratio", "0.05", "--patience", "10",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks", nargs="+", choices=tuple(TASK_ARGUMENTS),
        default=tuple(TASK_ARGUMENTS),
    )
    parser.add_argument("--output-root", default="runs/lra_mha_formal_full")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    for task in args.tasks:
        output = root / f"lra-{task}-mha-s0"
        test_metrics = output / "test_metrics.json"
        if test_metrics.is_file():
            metrics = json.loads(test_metrics.read_text(encoding="utf-8"))
            print(
                json.dumps(
                    {"task": task, "status": "complete", "accuracy": metrics["accuracy"]}
                ),
                flush=True,
            )
            continue
        command = [
            sys.executable,
            "experiments/lra_benchmark.py",
            "--task", task,
            "--output", str(output),
            *COMMON,
            *TASK_ARGUMENTS[task],
        ]
        print(json.dumps({"task": task, "command": command}), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
