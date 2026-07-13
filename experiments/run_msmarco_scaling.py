"""Run fixed-effective-batch MS MARCO scaling trials sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCALES = {
    "small": dict(dim=256, depth=6, heads=4, batch=16, accum=6),
    "base": dict(dim=384, depth=8, heads=6, batch=16, accum=6),
    "large": dict(dim=512, depth=12, heads=8, batch=16, accum=6),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scales", nargs="+", choices=tuple(SCALES), default=tuple(SCALES))
    p.add_argument("--mixer", choices=("mha", "lsso", "rrlsso"), default="rrlsso")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=250)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-root", default="runs/auxiliary/msmarco-scaling")
    p.add_argument("--cache-dir", default="data/auxiliary_cache/huggingface")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    for scale_name in args.scales:
        scale = SCALES[scale_name]
        output = Path(args.output_root) / f"{scale_name}-{args.mixer}-r{args.rank}-s{args.seed}"
        command = [
            sys.executable, str(ROOT / "experiments/msmarco_pretrain.py"),
            "--mixer", args.mixer, "--rank", str(args.rank),
            "--dim", str(scale["dim"]), "--depth", str(scale["depth"]),
            "--heads", str(scale["heads"]), "--batch-size", str(scale["batch"]),
            "--grad-accum", str(scale["accum"]), "--max-steps", str(args.max_steps),
            "--warmup-steps", str(args.warmup_steps), "--save-steps", str(args.save_steps),
            "--workers", str(args.workers), "--seed", str(args.seed),
            "--cache-dir", args.cache_dir, "--output", str(output),
        ]
        print(" ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
