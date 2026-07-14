"""Sequential launcher for the formal non-vision benchmark suite.

The launcher intentionally starts one process at a time.  It is safe to stop
and rerun: completed runs are skipped and incomplete runs resume from last.pt.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.genomic_benchmarks import GENOMIC_BENCHMARKS
from experiments.lra_benchmark import TASK_DEFAULTS
from experiments.uea_benchmark import UEA_30


SUITES = {
    "genomic": GENOMIC_BENCHMARKS,
    "lra": tuple(TASK_DEFAULTS),
    "uea": UEA_30,
}
SCRIPTS = {
    "genomic": ROOT / "experiments" / "genomic_benchmarks.py",
    "lra": ROOT / "experiments" / "lra_benchmark.py",
    "uea": ROOT / "experiments" / "uea_benchmark.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("genomic", "lra", "uea", "all"), default="all")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=(),
        help="optional subset; names must belong to the selected suite(s)",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--mixer", choices=("mha", "lsso", "rrlsso"), default="rrlsso")
    parser.add_argument(
        "--mixers",
        choices=("mha", "lsso", "rrlsso"),
        nargs="+",
        default=(),
        help="run several mixers sequentially; overrides --mixer",
    )
    parser.add_argument("--output-root", default="runs/sequence")
    parser.add_argument("--genomic-data-root", default="")
    parser.add_argument("--genomic-cache", default="data/genomic_benchmarks")
    parser.add_argument("--lra-data-root", default="data/lra")
    parser.add_argument("--lra-cache", default="data/lra_cache")
    parser.add_argument("--uea-data-root", default="data/uea")
    parser.add_argument("--download-aan", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="arguments after '--' are forwarded to every child runner",
    )
    return parser.parse_args()


def selected_jobs(args: argparse.Namespace):
    suites = tuple(SUITES) if args.suite == "all" else (args.suite,)
    known = {name for suite in suites for name in SUITES[suite]}
    unknown = set(args.datasets) - known
    if unknown:
        raise ValueError(f"datasets do not belong to the selected suite(s): {sorted(unknown)}")
    selected = set(args.datasets)
    mixers = tuple(args.mixers) or (args.mixer,)
    for suite in suites:
        for dataset in SUITES[suite]:
            if not selected or dataset in selected:
                for mixer in mixers:
                    for seed in args.seeds:
                        yield suite, dataset, mixer, seed


def command_for(args: argparse.Namespace, suite: str, dataset: str, mixer: str, seed: int):
    output = Path(args.output_root) / f"{suite}-{dataset}-{mixer}-s{seed}"
    command = [
        sys.executable,
        str(SCRIPTS[suite]),
        "--mixer",
        mixer,
        "--seed",
        str(seed),
        "--workers",
        str(args.workers),
        "--output",
        str(output),
    ]
    if suite == "genomic":
        command += ["--dataset", dataset, "--cache-dir", args.genomic_cache]
        if args.genomic_data_root:
            command += ["--data-root", args.genomic_data_root]
    elif suite == "lra":
        command += [
            "--task",
            dataset,
            "--data-root",
            args.lra_data_root,
            "--cache-dir",
            args.lra_cache,
        ]
        if args.download_aan:
            command.append("--download-aan")
    else:
        command += ["--dataset", dataset, "--data-root", args.uea_data_root]
    extra = list(args.extra)
    if extra[:1] == ["--"]:
        extra = extra[1:]
    return output, command + extra


def main() -> None:
    args = parse_args()
    failures = []
    for suite, dataset, mixer, seed in selected_jobs(args):
        output, command = command_for(args, suite, dataset, mixer, seed)
        result_path = output / "test_metrics.json"
        if result_path.exists() and not args.rerun_complete:
            print(json.dumps({"status": "skipped-complete", "output": str(output)}), flush=True)
            continue
        print(json.dumps({"status": "starting", "command": command}), flush=True)
        if not args.dry_run:
            started = time.time()
            try:
                subprocess.run(command, cwd=ROOT, check=True)
                (output / "failed.json").unlink(missing_ok=True)
            except subprocess.CalledProcessError as error:
                failure = {
                    "status": "failed",
                    "suite": suite,
                    "dataset": dataset,
                    "seed": seed,
                    "returncode": error.returncode,
                    "command": command,
                    "started_unix": started,
                    "failed_unix": time.time(),
                }
                output.mkdir(parents=True, exist_ok=True)
                (output / "failed.json").write_text(
                    json.dumps(failure, indent=2), encoding="utf-8"
                )
                print(json.dumps(failure), flush=True)
                failures.append(failure)
                if args.fail_fast:
                    raise
    if failures:
        print(json.dumps({"failed_jobs": len(failures)}), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
