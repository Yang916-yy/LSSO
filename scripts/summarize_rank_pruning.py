from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*", default=["runs/rank_pruning_main_seed1/rank_prune_*.jsonl"])
    return parser.parse_args()


def iter_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matched = [Path(p) for p in sorted(glob.glob(pattern))]
        if matched:
            paths.extend(matched)
        else:
            path = Path(pattern)
            if path.exists():
                paths.append(path)
    return paths


def fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def main() -> None:
    rows = []
    for path in iter_paths(parse_args().runs):
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            continue
        header = lines[0]
        for row in lines[1:]:
            if "keep_rank" not in row:
                continue
            rows.append(
                {
                    "dataset": header.get("dataset"),
                    "seed": header.get("seed"),
                    "train_rank": header.get("rank"),
                    "keep_rank": row.get("keep_rank"),
                    "r1": row.get("recall@1"),
                    "r10": row.get("recall@10"),
                    "mrr10": row.get("mrr@10"),
                    "compact_ratio": row.get("compact_doc_mixer_macs_ratio"),
                    "dynamic_ratio": row.get("dynamic_doc_mixer_macs_ratio"),
                    "effective_rank": row.get("diag_effective_rank"),
                    "correction_ratio": row.get("diag_correction_ratio"),
                    "eval_seconds": row.get("eval_seconds"),
                    "file": str(path),
                }
            )

    rows.sort(key=lambda r: (r["dataset"] or "", -(r["keep_rank"] or 0)))
    print(
        "dataset\tseed\ttrain_rank\tkeep_rank\tR@1\tR@10\tMRR@10\t"
        "compact_mac_ratio\tdynamic_mac_ratio\teffective_rank\tcorrection_ratio\teval_seconds\tfile"
    )
    for row in rows:
        print(
            f"{row['dataset']}\t{row['seed']}\t{row['train_rank']}\t{row['keep_rank']}\t"
            f"{fmt(row['r1'])}\t{fmt(row['r10'])}\t{fmt(row['mrr10'])}\t"
            f"{fmt(row['compact_ratio'])}\t{fmt(row['dynamic_ratio'])}\t"
            f"{fmt(row['effective_rank'])}\t{fmt(row['correction_ratio'])}\t"
            f"{fmt(row['eval_seconds'], 2)}\t{row['file']}"
        )


if __name__ == "__main__":
    main()
