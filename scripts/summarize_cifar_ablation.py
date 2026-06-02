from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", default="runs/cifar10_ablation_10epoch_20260602")
    return parser.parse_args()


def elapsed_seconds(run_dir: Path, mixer: str, rank: int) -> int | None:
    names = [f"{mixer}_r{rank}", f"{mixer}_seed11"]
    if mixer == "lsso-no-global":
        names.insert(0, f"lsso_r{rank}_no_global_seed11")
    elif mixer == "lsso":
        names.insert(0, f"lsso_r{rank}_seed11")
    elif mixer == "mha":
        names.insert(0, "mha_seed11")
    for name in names:
        log = run_dir / "console" / f"{name}.console.log"
        if not log.exists():
            continue
        matches = re.findall(r"ELAPSED_SECONDS=(\d+)", log.read_text(encoding="utf-8", errors="ignore"))
        if matches:
            return int(matches[-1])
    return None


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    rows = []
    for path in sorted(run_dir.glob("*.jsonl")):
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if len(lines) < 2:
            continue
        run_args = lines[0]["args"]
        epochs = [row for row in lines[1:] if "epoch" in row]
        if not epochs:
            continue
        best = max(epochs, key=lambda row: row.get("eval_acc", float("-inf")))
        last = epochs[-1]
        elapsed = elapsed_seconds(run_dir, run_args["mixer"], int(run_args["rank"]))
        rows.append(
            {
                "mixer": run_args["mixer"],
                "rank": int(run_args["rank"]),
                "epochs": len(epochs),
                "elapsed_s": elapsed,
                "best_acc": best["eval_acc"],
                "best_epoch": best["epoch"],
                "last_acc": last["eval_acc"],
                "last_train_acc": last["train_acc"],
                "last_train_loss": last["train_loss"],
                "last_eval_loss": last["eval_loss"],
                "last_corr": last.get("eval_diag_correction_ratio"),
                "last_eff_rank": last.get("eval_diag_effective_rank"),
                "file": str(path),
            }
        )

    order = {"mha": 0, "lsso": 1, "lsso-no-global": 2}
    rows.sort(key=lambda row: (order.get(row["mixer"], 99), row["rank"]))
    print(
        "mixer\trank\tepochs\telapsed_s\tbest_acc@epoch\tlast_acc\tlast_train_acc\t"
        "last_train_loss\tlast_eval_loss\tlast_correction\tlast_eff_rank\tfile"
    )
    for row in rows:
        print(
            f"{row['mixer']}\t{row['rank']}\t{row['epochs']}\t{row['elapsed_s'] or '-'}\t"
            f"{row['best_acc']:.4f}@{row['best_epoch']}\t{row['last_acc']:.4f}\t"
            f"{row['last_train_acc']:.4f}\t{row['last_train_loss']:.4f}\t{row['last_eval_loss']:.4f}\t"
            f"{row['last_corr'] if row['last_corr'] is not None else '-'}\t"
            f"{row['last_eff_rank'] if row['last_eff_rank'] is not None else '-'}\t{row['file']}"
        )


if __name__ == "__main__":
    main()
