# Experiment output policy

`runs/` is a disposable, Git-ignored working directory. It must not be used as
a permanent result database.

## During a run

Each experiment owns one shallow directory:

```text
runs/<benchmark>/<task>-<mixer>-s<seed>/
  config.json
  metrics.jsonl
  train.log
  last.pt
  best.pt
```

Do not create separate top-level trees for smoke, preflight, retry, diagnostic,
and ablation variants. Encode the variant in the run name and set a short
retention time for unsuccessful runs.

## After a run

1. Export final scalar results to a compact CSV/JSON table under the relevant
   paper or experiment-plan directory.
2. Keep `best.pt` only when it is required for evaluation, visualization, or a
   published reproducibility artifact.
3. Remove `last.pt`, initialization checkpoints, fold checkpoints, PID files,
   and smoke outputs after validation.
4. Store large retained checkpoints outside the source repository and record
   their location plus checksum in the experiment table.

The source package should contain code, configurations, compact final tables,
and plotting scripts—not training state.

## Archived pre-cleanup outputs

The 2026-07-19 snapshot was moved without deletion to:

```text
/root/LSSO-local-archive/runs-20260719-pre-cleanup
```

It is local-only and can be searched or restored if an old result is needed.
