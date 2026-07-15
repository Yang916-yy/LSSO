"""Low-budget HyenaDNA-flavored recipe transfer for RRLSSO-DNA.

This validation-only probe reuses the frozen Base+motif-7 architecture and
touches only the two tasks where the initial recipe trails most clearly.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("human_nontata_promoters", "human_ocr_ensembl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="",
    )
    parser.add_argument("--cache-dir", default="data/genomic_benchmarks")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--rc-augmentation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = "hyenadna-flavor-rc" if args.rc_augmentation else "hyenadna-flavor"
    default_name = "recipe_hyenadna_flavor_rc" if args.rc_augmentation else "recipe_hyenadna_flavor"
    output_root = Path(args.output_root or f"runs/rrlsso_dna_program/{default_name}/base_motif7")
    for task in args.tasks:
        output = output_root / f"{task}-s{args.seed}"
        result = output / "validation_metrics.json"
        if result.is_file():
            print(json.dumps({"status": "skipped-complete", "result": str(result)}))
            continue
        command = [
            sys.executable,
            str(ROOT / "experiments" / "genomic_benchmarks.py"),
            "--dataset", task,
            "--cache-dir", args.cache_dir,
            "--output", str(output),
            "--training-profile", profile,
            "--mixer", "rrlsso",
            "--dim", "256",
            "--depth", "2",
            "--heads", "8",
            "--rank", "32",
            "--local-motif-kernel", "7",
            "--max-parameters", "2200000",
            "--workers", str(args.workers),
            "--seed", str(args.seed),
            "--validation-only",
        ]
        print(json.dumps({"status": "starting", "command": command}), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
