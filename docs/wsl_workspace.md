# WSL development workspace

The native development checkout is `/root/LSSO` in Ubuntu-24.04. It is the
only checkout used for builds, tests, and GPU training. The Windows checkout
is a Git mirror, not an independent experiment tree.

```bash
cd /root/LSSO
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[experiments]'
```

The dataset cache may remain on the Windows disk to avoid duplication. Pass
it explicitly, for example `--data-dir /mnt/d/LSSO-data/torchvision` after the
cache is moved outside the repository.

Build the optional MathDx extension from the native checkout:

```bash
bash tools/build_mathdx_backend.sh
```

Generated checkpoints, run logs, build products, and datasets are ignored by
Git and must not be treated as source files.
