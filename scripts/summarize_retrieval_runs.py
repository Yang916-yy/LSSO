from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*", default=["runs/*bertstyle_retr*.jsonl"])
    return parser.parse_args()


def iter_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        if any(ch in pattern for ch in "*?[]"):
            matched = [Path(p) for p in sorted(glob.glob(pattern))]
        else:
            matched = []
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
    args = parse_args()
    rows = []
    for path in iter_paths(args.runs):
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if not lines or "args" not in lines[0]:
            continue
        run_args = lines[0]["args"]
        epochs = [row for row in lines[1:] if "epoch" in row]
        if not epochs or "recall@10" not in epochs[0]:
            continue
        best_r10 = max(epochs, key=lambda row: row.get("recall@10", float("-inf")))
        best_mrr = max(epochs, key=lambda row: row.get("mrr@10", float("-inf")))
        last = epochs[-1]
        rows.append(
            {
                "dataset": run_args.get("dataset"),
                "mixer": run_args.get("mixer"),
                "rank": run_args.get("rank"),
                "seed": run_args.get("seed"),
                "params": lines[0].get("params"),
                "query_mixer_macs": (lines[0].get("query_cost") or {}).get("mixer_macs"),
                "doc_mixer_macs": (lines[0].get("doc_cost") or {}).get("mixer_macs"),
                "query_macs": (lines[0].get("query_cost") or {}).get("total_macs"),
                "doc_macs": (lines[0].get("doc_cost") or {}).get("total_macs"),
                "doc_flops": (lines[0].get("doc_cost") or {}).get("total_flops"),
                "best_r10_epoch": best_r10.get("epoch"),
                "best_r10": best_r10.get("recall@10"),
                "best_mrr_epoch": best_mrr.get("epoch"),
                "best_mrr": best_mrr.get("mrr@10"),
                "last_r1": last.get("recall@1"),
                "last_r10": last.get("recall@10"),
                "last_mrr": last.get("mrr@10"),
                "last_loss": last.get("train_loss"),
                "samples_per_sec": last.get("train_samples_per_sec"),
                "file": str(path),
            }
        )

    rows.sort(key=lambda row: (row["dataset"] or "", row["mixer"] or "", row["rank"] or 0, row["seed"] or 0))
    print(
        "dataset\tmixer\trank\tseed\tparams\tquery_mixer_macs\tdoc_mixer_macs\t"
        "query_macs\tdoc_macs\tdoc_flops\t"
        "best_r10@ep\tbest_mrr@ep\tlast_r1\tlast_r10\tlast_mrr\tsamples/s\tfile"
    )
    for row in rows:
        print(
            f"{row['dataset']}\t{row['mixer']}\t{row['rank']}\t{row['seed']}\t{row['params']}\t"
            f"{row['query_mixer_macs']}\t{row['doc_mixer_macs']}\t"
            f"{row['query_macs']}\t{row['doc_macs']}\t{row['doc_flops']}\t"
            f"{fmt(row['best_r10'])}@{row['best_r10_epoch']}\t"
            f"{fmt(row['best_mrr'])}@{row['best_mrr_epoch']}\t"
            f"{fmt(row['last_r1'])}\t{fmt(row['last_r10'])}\t{fmt(row['last_mrr'])}\t"
            f"{fmt(row['samples_per_sec'], 1)}\t{row['file']}"
        )


if __name__ == "__main__":
    main()
