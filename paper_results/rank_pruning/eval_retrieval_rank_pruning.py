from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from examples.models.bertstyle import BertStyleEncoder
from train_bertstyle_retrieval import (
    build_eval_sets,
    build_qrels_map,
    estimate_encoder_macs,
    evaluate,
    join_title_text,
    load_beir_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate inference-time LSSO rank pruning.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-dir", default="runs/rank_pruning")
    parser.add_argument("--keep-ranks", default="0,4,8,12,16,24,32")
    parser.add_argument("--max-eval-queries", type=int, default=None)
    parser.add_argument("--max-corpus-docs", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def set_prune_rank(model: torch.nn.Module, keep_rank: int | None) -> None:
    if not hasattr(model, "lsso_layers"):
        return
    for layer in model.lsso_layers():
        layer.prune_rank_keep = keep_rank


def mixer_macs_for_keep(base_args: argparse.Namespace, seq_len: int, keep_rank: int | None) -> int:
    cost_args = SimpleNamespace(**vars(base_args))
    if keep_rank is not None:
        cost_args.rank = keep_rank
    return estimate_encoder_macs(cost_args, seq_len)["mixer_macs"]


def dynamic_pruned_mixer_macs(base_args: argparse.Namespace, seq_len: int, keep_rank: int | None) -> int:
    if keep_rank is None:
        return mixer_macs_for_keep(base_args, seq_len, None)
    D = base_args.dim
    H = base_args.num_heads
    r_full = base_args.rank
    r = keep_rank
    depth = base_args.depth
    linear_macs = seq_len * D * (H * r_full + D) + seq_len * D * D
    solve_macs = H * seq_len * r * r + 2 * seq_len * r * D + D * r * r + H * r * r * r
    return int(depth * (linear_macs + solve_macs))


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    run_args = argparse.Namespace(**ckpt["args"])
    if run_args.mixer not in {"lsso", "lsso-no-global"}:
        raise ValueError(f"rank pruning only applies to LSSO checkpoints, got mixer={run_args.mixer}")

    if args.max_eval_queries is not None:
        run_args.max_eval_queries = args.max_eval_queries
    if args.max_corpus_docs is not None:
        run_args.max_corpus_docs = args.max_corpus_docs
    if args.eval_batch_size is not None:
        run_args.eval_batch_size = args.eval_batch_size

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        run_args.tokenizer_name,
        use_fast=True,
        local_files_only=run_args.local_files_only,
    )
    corpus_ds, queries_ds, _, qrels_test_ds = load_beir_dataset(run_args.dataset, run_args.offline)
    corpus = {str(row["_id"]): join_title_text(row.get("title", ""), row.get("text", "")) for row in corpus_ds}
    queries = {str(row["_id"]): join_title_text(row.get("title", ""), row.get("text", "")) for row in queries_ds}
    qrels_test = build_qrels_map(qrels_test_ds)
    corpus_texts, eval_queries, positives, candidate_indices = build_eval_sets(
        queries,
        corpus,
        qrels_test,
        run_args.max_eval_queries,
        run_args.max_corpus_docs,
    )

    model = BertStyleEncoder(
        vocab_size=len(tokenizer),
        num_classes=2,
        max_len=max(run_args.max_query_len, run_args.max_doc_len),
        dim=run_args.dim,
        depth=run_args.depth,
        num_heads=run_args.num_heads,
        mixer=run_args.mixer,
        rank=run_args.rank,
        mlp_ratio=run_args.mlp_ratio,
        dropout=run_args.dropout,
        gamma_max=run_args.gamma_max,
        theta_gamma_init=run_args.theta_gamma_init,
        pad_id=tokenizer.pad_token_id or 0,
    )
    model.load_state_dict(ckpt["model"])
    device = torch.device(args.device)
    model.to(device)

    keep_ranks = []
    for item in args.keep_ranks.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        keep_ranks.append(None if value <= 0 or value >= run_args.rank else value)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f"rank_prune_{run_args.dataset}_{run_args.mixer}_r{run_args.rank}_s{run_args.seed}.jsonl"

    use_amp = args.amp and device.type == "cuda"
    full_doc_mixer_macs = mixer_macs_for_keep(run_args, run_args.max_doc_len, None)
    completed_keep_ranks = set()
    if args.resume and out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines()[1:]:
            if not line.strip():
                continue
            row = json.loads(line)
            if "keep_rank" in row:
                completed_keep_ranks.add(int(row["keep_rank"]))

    mode = "a" if args.resume and out_path.exists() else "w"
    with out_path.open(mode, encoding="utf-8") as f:
        header = {
            "checkpoint": str(args.checkpoint),
            "dataset": run_args.dataset,
            "rank": run_args.rank,
            "seed": run_args.seed,
            "full_doc_mixer_macs": full_doc_mixer_macs,
            "max_eval_queries": run_args.max_eval_queries,
            "max_corpus_docs": run_args.max_corpus_docs,
        }
        if mode == "w":
            f.write(json.dumps(header, sort_keys=True) + "\n")
        for keep_rank in keep_ranks:
            keep_rank_value = run_args.rank if keep_rank is None else keep_rank
            if keep_rank_value in completed_keep_ranks:
                print(json.dumps({"keep_rank": keep_rank_value, "skipped": "completed"}, sort_keys=True), flush=True)
                continue
            set_prune_rank(model, keep_rank)
            start = time.perf_counter()
            metrics = evaluate(
                model,
                tokenizer,
                corpus_texts,
                eval_queries,
                positives,
                candidate_indices,
                device,
                run_args,
                use_amp,
            )
            elapsed = time.perf_counter() - start
            compact_doc_mixer_macs = mixer_macs_for_keep(run_args, run_args.max_doc_len, keep_rank)
            dynamic_doc_mixer_macs = dynamic_pruned_mixer_macs(run_args, run_args.max_doc_len, keep_rank)
            row = {
                "keep_rank": keep_rank_value,
                "pruned": keep_rank is not None,
                "dynamic_doc_mixer_macs": dynamic_doc_mixer_macs,
                "dynamic_doc_mixer_macs_ratio": dynamic_doc_mixer_macs / full_doc_mixer_macs,
                "compact_doc_mixer_macs": compact_doc_mixer_macs,
                "compact_doc_mixer_macs_ratio": compact_doc_mixer_macs / full_doc_mixer_macs,
                "eval_seconds": elapsed,
                **metrics,
            }
            print(json.dumps(row, sort_keys=True), flush=True)
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()

    print(f"log={out_path}")


if __name__ == "__main__":
    main()
