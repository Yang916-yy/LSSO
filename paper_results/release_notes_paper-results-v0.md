# LSSO Paper Results v0

This release stores checkpoint archives for completed LSSO paper experiments.
The git repository intentionally keeps only lightweight experiment metadata:
summary tables, manifests, source scripts, and JSONL logs under
`paper_results/`.

Default `git clone` does not download checkpoints. Download only the release
assets needed for reproduction.

## Repository Metadata

Tracked in the repository:

- `paper_results/retrieval_main/summary.tsv`
- `paper_results/retrieval_ablation/summary.tsv`
- `paper_results/rank_pruning/summary.tsv`
- `paper_results/cifar100_cv_main/summary.tsv`
- `paper_results/*/manifest.tsv`
- source scripts and JSONL logs for the completed runs
- `paper_results/release_assets.tsv` with asset names, sizes, SHA256 checksums,
  and descriptions

## Release Assets

Checkpoint archives:

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
- `SHA256SUMS`  
  Checksum file for release asset verification.

## Result Highlights

Retrieval main table uses random-initialized BERT-style encoders with
`dim=256`, `depth=8`, `heads=8`, `max_doc_len=512`, mean pooling, and 3 seeds.
Reported MACs are mixer-only document-side MACs.

- FIQA: LSSO-r32 reaches R@10 0.2387 and MRR@10 0.1150 with 57.7% fewer mixer
  MACs than MHA.
- NFCorpus: LSSO-r32 reaches R@10 0.5841 and MRR@10 0.4333 with 57.7% fewer
  mixer MACs than MHA.
- SciFact: LSSO-r32 reaches R@10 0.7089 and MRR@10 0.6247 with 57.7% fewer
  mixer MACs than MHA.
- LSSO-r16 uses 66.8% fewer mixer MACs than MHA in the retrieval setup.

CIFAR-100 CV main table uses a patch-2 ViT-style encoder with `dim=96`,
`depth=3`, `heads=6`, CLS pooling, RandAugment(2,9), Mixup=0.2, CutMix=0.5,
and 3 seeds.

- LSSO-r16 uses 62.5% fewer mixer MACs than MHA.
- LSSO-r32 uses 42.1% fewer mixer MACs than MHA.
- Full numeric results are tracked in
  `paper_results/cifar100_cv_main/summary.tsv`.

## Verify Downloads

Download the needed archives and verify checksums:

```bash
sha256sum -c SHA256SUMS
```

Extract an archive:

```bash
tar -xf retrieval_main_fiqa_checkpoints.tar
```

The extracted checkpoint paths mirror the corresponding `paper_results/*`
experiment group.
