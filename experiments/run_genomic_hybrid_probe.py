"""Validation-only gate for the RRLSSO-DNA local/global hybrid backbone."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("dummy_mouse_enhancers_ensembl", "human_nontata_promoters")
ARCHITECTURES = {
    "base_motif7": {
        "batch_size": 64,
        "grad_accum": 2,
        "eval_batch_size": 256,
        "arguments": ["--local-motif-kernel", "7", "--position-rank", "0"],
    },
    "hybrid_gated_dilated": {
        "batch_size": 32,
        "grad_accum": 4,
        "eval_batch_size": 64,
        "arguments": [
            "--local-motif-kernel", "0",
            "--local-motif-dilations", "1", "1", "4", "16", "64",
            "--local-motif-layer-scale", "0.001",
            "--position-rank", "32",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="runs/rrlsso_dna_hybrid_probe")
    parser.add_argument("--cache-dir", default="data/genomic_benchmarks")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    summary: dict[str, dict[str, dict]] = {}
    for task in TASKS:
        summary[task] = {}
        for architecture, architecture_config in ARCHITECTURES.items():
            output = root / architecture / f"{task}-s{args.seed}"
            result_path = output / "validation_metrics.json"
            if args.rerun or not result_path.is_file():
                command = [
                    sys.executable,
                    str(ROOT / "experiments" / "genomic_benchmarks.py"),
                    "--dataset", task,
                    "--cache-dir", args.cache_dir,
                    "--output", str(output),
                    "--mixer", "rrlsso",
                    "--dim", "256",
                    "--depth", "2",
                    "--heads", "8",
                    "--rank", "32",
                    "--pooling", "mean",
                    "--max-parameters", "3000000",
                    "--workers", str(args.workers),
                    "--seed", str(args.seed),
                    "--epochs", "100",
                    "--patience", "100",
                    "--batch-size", str(architecture_config["batch_size"]),
                    "--grad-accum", str(architecture_config["grad_accum"]),
                    "--eval-batch-size", str(architecture_config["eval_batch_size"]),
                    "--lr", "0.0006",
                    "--weight-decay", "0.01",
                    "--warmup-ratio", "0.01",
                    "--min-lr-ratio", "0.1",
                    "--dropout", "0.0",
                    "--embedding-dropout", "0.1",
                    "--reverse-complement-probability", "0.5",
                    "--mutation-probability", "0.002",
                    "--mutation-clean-epochs", "20",
                    "--posthoc-rc-eval",
                    "--validation-only",
                    *architecture_config["arguments"],
                ]
                print(json.dumps({"status": "starting", "architecture": architecture,
                                  "task": task, "command": command}), flush=True)
                subprocess.run(command, cwd=ROOT, check=True)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            summary[task][architecture] = result
            print(json.dumps({"status": "complete", "architecture": architecture,
                              "task": task, "accuracy": result["accuracy"]}), flush=True)
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
