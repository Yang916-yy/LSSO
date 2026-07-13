"""Aggregate completed auxiliary runs into one CSV table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/auxiliary")
    parser.add_argument("--output", default="runs/auxiliary/summary.csv")
    args = parser.parse_args()
    rows = []
    for metrics_path in sorted(Path(args.root).glob("*/test_metrics.json")):
        config_path = metrics_path.parent / "config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        metrics = json.loads(metrics_path.read_text())
        rows.append({
            "run": metrics_path.parent.name,
            "task": "beir" if "dataset" in config and config["dataset"] in
                    ("nfcorpus", "fiqa", "scifact") else "flip-aav",
            "dataset": config.get("dataset", "flip-aav"),
            "mixer": config["mixer"],
            "rank": config["rank"],
            "seed": config["seed"],
            **metrics,
        })
    if not rows:
        raise RuntimeError(f"no completed runs found under {args.root}")
    fields = sorted({key for row in rows for key in row})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} runs to {output}")


if __name__ == "__main__":
    main()
