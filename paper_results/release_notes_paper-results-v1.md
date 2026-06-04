# LSSO Paper Results v1

This release adds checkpoint archives for two additional completed experiments. The git repository tracks the lightweight summaries, manifests, notebooks, and JSONL logs; the release stores only checkpoint archives and checksums.

## Added Experiments

### MS MARCO -> BEIR Transfer

Random-initialized BERT-style retrieval encoders are pretrained on MS MARCO and evaluated zero-shot on BEIR-style datasets across 3 seeds. Models included:

- MHA
- Nystromformer
- LSSO-r16
- LSSO-r32

Tracked metadata:

- `paper_results/msmarco_beir_transfer/summary.tsv`
- `paper_results/msmarco_beir_transfer/beir_zero_shot_summary_s*.csv`
- `paper_results/msmarco_beir_transfer/logs/`
- `paper_results/msmarco_beir_transfer/notebooks/`

Release checkpoint asset:

- `msmarco_beir_transfer_checkpoints.tar`

### ImageNet-100 CV Main Table

One-seed ImageNet-100 CV run with ViT-style encoders at image size 224 and patch size 8. Models included:

- MHA
- LSSO-r16
- LSSO-r32

Tracked metadata:

- `paper_results/imagenet100_cv_main/summary.tsv`
- `paper_results/imagenet100_cv_main/logs/`
- `paper_results/imagenet100_cv_main/notebooks/`

Release checkpoint asset:

- `imagenet100_cv_main_checkpoints.tar`

## Verify Downloads

```bash
sha256sum -c SHA256SUMS
tar -xf msmarco_beir_transfer_checkpoints.tar
tar -xf imagenet100_cv_main_checkpoints.tar
```

See `paper_results/release_assets.tsv` for exact asset sizes and SHA256 checksums.
