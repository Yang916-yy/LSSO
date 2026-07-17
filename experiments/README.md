# Experiments

This directory contains reproducible task programs, not reusable operator
implementation.  Reusable layers belong in `lsso`; external framework glue
belongs in `integrations`; timing-only programs belong in `benchmarks`.

Experiment entry points should:

1. expose all result-affecting settings through arguments or a checked config;
2. record the source commit, seed, split, model, optimizer, and backend;
3. support an inexpensive smoke mode;
4. write only under an explicit output directory;
5. resume from a checkpoint without silently restarting schedules.

Naming conventions:

- `<task>_benchmark.py`: stable task entry point;
- `run_<program>.py`: orchestrates several stable task runs;
- `diagnose_<topic>.py`: temporary diagnostic, not a formal result;
- `summarize_<program>.py`: reads outputs and emits a compact summary.

One-off probes should be retired or promoted once their decision has been made.
Retirement means deleting the script after its useful conclusion is captured in
`docs/`, not preserving every exploratory file indefinitely.
