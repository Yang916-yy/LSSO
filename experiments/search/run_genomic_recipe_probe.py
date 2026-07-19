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


ROOT = Path(__file__).resolve().parents[2]
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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--rc-augmentation", action="store_true")
    mode.add_argument("--rc-mutation", action="store_true")
    mode.add_argument("--posthoc-rc-eval", action="store_true")
    parser.add_argument(
        "--checkpoint-root",
        default="runs/rrlsso_dna_program/recipe_hyenadna_flavor_rc/base_motif7",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rc_mutation:
        profile, default_name = (
            "hyenadna-flavor-rc-mutation",
            "recipe_hyenadna_flavor_rc_mutation",
        )
    elif args.rc_augmentation or args.posthoc_rc_eval:
        profile = "hyenadna-flavor-rc"
        default_name = (
            "recipe_hyenadna_flavor_rc_posthoc"
            if args.posthoc_rc_eval
            else "recipe_hyenadna_flavor_rc"
        )
    else:
        profile, default_name = "hyenadna-flavor", "recipe_hyenadna_flavor"
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
        if args.posthoc_rc_eval:
            checkpoint = Path(args.checkpoint_root) / f"{task}-s{args.seed}" / "best.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
            command += [
                "--evaluate-checkpoint", str(checkpoint),
                "--posthoc-rc-eval",
            ]
        print(json.dumps({"status": "starting", "command": command}), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
