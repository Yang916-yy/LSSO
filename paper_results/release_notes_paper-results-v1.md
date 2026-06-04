# LSSO Paper Results

This release stores all checkpoint archives for completed LSSO paper
experiments currently tracked in the repository. The git repository keeps only
lightweight summaries, manifests, notebooks/scripts, and JSONL logs; this
release stores checkpoint archives and checksums.

Use this single release as the checkpoint source for all `paper_results/`
tables.

## Checkpoint Assets

- `retrieval_main_fiqa_checkpoints.tar`  
  FIQA retrieval main-table checkpoints for MHA, Nystromformer, LSSO-r16, and
  LSSO-r32 across 3 seeds.
- `retrieval_main_nfcorpus_checkpoints.tar`  
  NFCorpus retrieval main-table checkpoints for the same four mixers across 3
  seeds.
- `retrieval_main_scifact_checkpoints.tar`  
  SciFact retrieval main-table checkpoints for the same four mixers across 3
  seeds.
- `retrieval_ablation_fiqa_checkpoints.tar`  
  FIQA LSSO ablation checkpoints: no-global, fixed mu/gamma, no U RMS norm,
  r8, and r4 across 3 seeds.
- `retrieval_ablation_scifact_checkpoints.tar`  
  SciFact LSSO ablation checkpoints for the same ablation set across 3 seeds.
- `cifar100_cv_main_checkpoints.tar`  
  CIFAR-100 CV main-table checkpoints for MHA, Nystromformer, LSSO-r16, and
  LSSO-r32 across 3 seeds.
- `msmarco_beir_transfer_checkpoints.tar`  
  MS MARCO pretraining checkpoints used for 3-seed BEIR zero-shot transfer
  evaluation.
- `imagenet100_cv_main_checkpoints.tar`  
  ImageNet-100 one-seed CV main-table checkpoints.
- `SHA256SUMS`  
  Checksum file for release asset verification.

## Tracked Metadata

The repository tracks the lightweight metadata needed to read, audit, and reuse
the experiments:

- `paper_results/retrieval_main/summary.tsv`
- `paper_results/retrieval_ablation/summary.tsv`
- `paper_results/msmarco_beir_transfer/summary.tsv`
- `paper_results/rank_pruning/summary.tsv`
- `paper_results/cifar100_cv_main/summary.tsv`
- `paper_results/imagenet100_cv_main/summary.tsv`
- `paper_results/*/manifest.tsv`
- source notebooks/scripts and JSONL logs
- `paper_results/release_assets.tsv`

## Verify Downloads

```bash
sha256sum -c SHA256SUMS
tar -xf retrieval_main_fiqa_checkpoints.tar
tar -xf msmarco_beir_transfer_checkpoints.tar
```

See `paper_results/release_assets.tsv` for exact asset sizes and SHA256
checksums.
