# Auxiliary sequence experiments

The auxiliary suite tests the same bidirectional MHA/LSSO/RRLSSO replacement
outside vision without inheriting a BERT encoder. Both tasks use
`SequenceMixerEncoder`: token embeddings, learned absolute position
embeddings, pre-norm residual mixer blocks, MLPs, and masked mean pooling.
RRLSSO additionally applies its fixed 1-D rank rotary transform inside the
relation solve. Padding is passed to the masked operator and excluded from
pooling.

## Installation

```bash
python -m pip install -e ".[auxiliary]"
```

## BEIR text retrieval

The entry point downloads the official Hugging Face BEIR corpus, query, and
qrels tables. It trains a shared dual encoder with symmetric in-batch
contrastive loss. Ten percent of training queries form a deterministic
validation set; the test qrels are evaluated only after selecting `best.pt`.
Reported metrics are NDCG@10, MRR@10, and Recall@1/5/10/100.

```bash
python experiments/beir_retrieval.py \
  --dataset scifact --mixer rrlsso --rank 32 \
  --output runs/auxiliary/beir-scifact-rrlsso-r32
```

Tokenization is cached once under `data/auxiliary_cache/`. Evaluation encodes
the corpus in batches and performs exact top-100 retrieval with a bounded GPU
score matrix, so it never materializes the full query-by-corpus matrix.

Run the formal grid over `nfcorpus`, `fiqa`, and `scifact`, mixers
`mha`, `lsso`, and `rrlsso`, and the same seeds. The default tokenizer is the
T5 SentencePiece vocabulary; no T5/BERT encoder weights or block structure
are loaded.

## FLIP AAV protein fitness

The protein route uses the public `AI4Protein/FLIP_AAV_two-vs-rest` dataset,
its provided train/validation/test split, a character-level amino-acid
tokenizer, and a scalar regression head. Targets are standardized using only
the training split. Model selection uses validation Spearman correlation;
the test split is evaluated once from `best.pt`.
Protein strings are encoded once at startup. A length-bucket sampler applies
dynamic per-batch padding so the masked mixers do not read padded tokens.

```bash
python experiments/flip_aav.py \
  --mixer rrlsso --rank 32 \
  --output runs/auxiliary/flip-aav-rrlsso-r32
```

Formal comparisons use identical dimensions, depth, learned position
embeddings, optimizer, split, and seeds for all three mixers. The primary
metric is Spearman correlation and MSE is retained as a secondary metric.

## Smoke tests

Use small limits to validate a new machine without claiming the resulting
numbers:

```bash
python experiments/beir_retrieval.py --dataset scifact --dim 32 --depth 1 \
  --heads 4 --rank 8 --epochs 1 --batch-size 4 --workers 0 \
  --max-train-pairs 8 --max-eval-queries 4 --max-corpus-docs 16 \
  --output /tmp/lsso-beir-smoke --no-resume

python experiments/flip_aav.py --dim 32 --depth 1 --heads 4 --rank 8 \
  --max-length 64 --epochs 1 --batch-size 4 --workers 0 \
  --max-train-samples 8 --max-eval-samples 8 \
  --output /tmp/lsso-flip-smoke --no-resume
```

Every run writes `config.json`, `metrics.jsonl`, atomic `last.pt` and
`best.pt`, and final `test_metrics.json`.

On CUDA, both entries enable BF16 autocast, fused AdamW, pinned DataLoader
memory, persistent workers, and non-blocking host-to-device copies. These are
disabled automatically on CPU.

Run the complete matrix sequentially on one GPU and aggregate completed runs:

```bash
python experiments/run_auxiliary_grid.py --task all --seeds 0 1 2
python experiments/summarize_auxiliary.py
```
