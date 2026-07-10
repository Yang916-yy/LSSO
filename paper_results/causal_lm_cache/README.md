# Archived: Causal LM Cache Experiments

> These records are retained for historical reference only. Causal LSSO is not
> a supported package path or an active experimental direction.

This folder keeps the non-paper-main but useful causal LSSO cache checks from
the local IMDB next-token experiments.

Files:

- `summary_3000.tsv`: 3000-step causal LM comparison summary.
- `allmixers_learned_3000.tsv`: learned absolute-position run details.
- `allmixers_none_3000.tsv`: no-position run details.
- `sp_vs_uc_cache_learned_3000.tsv`: UC-cache vs S/P solve-state cache
  inference comparison on the learned-position RoPE/LSSO runs.

The large scratch checkpoints and smoke-test outputs were intentionally left
out; this directory stores only compact tables needed to recover the observed
cache behavior.
