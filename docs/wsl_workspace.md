# WSL workspace layout

The canonical working tree for agent and CUDA work is `/root/LSSO` in the
Ubuntu-24.04 WSL filesystem. This avoids routine source and metadata I/O on
the Windows-mounted `/mnt/d` volume.

The former Windows worktree at `D:\LSSO` is a migration source only. Do not
develop independently in both locations: fetch/pull the same Git branch from
the canonical WSL tree, and use GitHub as the synchronization boundary.

Formal checkpoints that are too large for Git live under
`/root/LSSO/artifacts/` (ignored by Git). The concise metrics, figures, and
reproduction scripts belong in `paper_results/` and are tracked.

The native WSL runtime is in `/root/LSSO/.venv`:

```bash
cd /root/LSSO
source .venv/bin/activate
```

Set `LSSO_MATHDX_LIBRARY` to a MathDx shared library built for the active GPU
when running native backend experiments.
