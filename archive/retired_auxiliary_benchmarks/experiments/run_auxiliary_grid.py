"""Archived launcher for the retired BEIR/FLIP auxiliary grid."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("beir", "flip-aav", "all"), default="all")
    parser.add_argument("--mixers", nargs="+", default=("mha", "lsso", "rrlsso"))
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--beir-datasets", nargs="+", default=("nfcorpus", "fiqa", "scifact"))
    parser.add_argument("--output-root", default="runs/auxiliary")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    commands: list[list[str]] = []
    if args.task in ("beir", "all"):
        for dataset in args.beir_datasets:
            for mixer in args.mixers:
                for seed in args.seeds:
                    name = f"beir-{dataset}-{mixer}-r32-s{seed}"
                    commands.append([
                        sys.executable, str(ROOT / "experiments/beir_retrieval.py"),
                        "--dataset", dataset, "--mixer", mixer, "--rank", "32",
                        "--seed", str(seed), "--output", str(Path(args.output_root) / name),
                    ])
    if args.task in ("flip-aav", "all"):
        for mixer in args.mixers:
            for seed in args.seeds:
                name = f"flip-aav-{mixer}-r32-s{seed}"
                commands.append([
                    sys.executable, str(ROOT / "experiments/flip_aav.py"),
                    "--mixer", mixer, "--rank", "32", "--seed", str(seed),
                    "--output", str(Path(args.output_root) / name),
                ])
    for index, command in enumerate(commands, 1):
        print(f"[{index}/{len(commands)}]", " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
