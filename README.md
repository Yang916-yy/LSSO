# LSSO

LSSO is a low-rank global token mixer with one PyTorch reference operator and
one reserved CUDA boundary. It contains no legacy compatibility path.

The default operator builds a QR soft frame, forms one shared compact state,
generates a sample-conditioned accretive generator, and evaluates the resulting
equilibrium directly. It does not materialize a compact transition matrix.

~~~python
import torch
from lsso import LSSO, LSSOConfig

layer = LSSO(LSSOConfig(dim=192, num_heads=3, rank=16)).cuda()
x = torch.randn(8, 65, 192, device="cuda")
y = layer(x)
~~~

The explicit CUDA fast path covers only the complete DYNAMIC + Rank-Rotary
operator, with rank 16, 32, 48, or 64 and any positive practical head dimension.
It accepts the current operator's optional boolean `valid_mask` and shared or
per-sample position IDs without falling back to another implementation. It
targets Turing SM75 and newer supported NVIDIA architectures.
Build its strict per-SM artifacts with `tools/build_cuda.sh`, then load the
artifact for the device before requesting it:

~~~python
from lsso.ball import cuda

cuda.load(device=x.device)
y = layer(x, implementation="cuda")
~~~

Official releases provide a separate runtime wheel containing all eight CUDA
artifacts. It is intentionally exact to the release binary contract:
`torch==2.11.0+cu128`, CUDA `12.8`, native contract `6`, and Linux x86_64.
Install the matching main and runtime wheels, then `cuda.load()` discovers the
device-specific artifact without a local CUDA toolkit or compilation step.

~~~bash
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.11.0+cu128'
python -m pip install \
  ./lsso_operator-0.6.3-py3-none-any.whl \
  ./lsso_cuda_runtime-0.6.3+torch2110cu128-py3-none-linux_x86_64.whl
~~~

Source checkouts still prefer `build/cuda/lib/` for development; an explicit
`LSSO_CUDA_LIBRARY` remains available for a manually built artifact.

The timm adapter keeps the backend explicit for the ImageNet and downstream
workflows.

The only supported ablations are DYNAMIC, STATIC, and ZERO core ownership, plus
the Rank-Rotary on/off switch. See docs/CORE_CONTRACT.md.

GenomicBenchmarks and LRA use the shared PyTorch sequence runner:

~~~bash
python -m experiments.train_transformers --config experiments/configs/genomic.toml \
  --output runs/sequence/genomic-pilot --validation-only
~~~

The runner keeps MHA and LSSO backbones matched, pins split/provenance metadata,
selects checkpoints from validation only, and evaluates test once after
selection. See docs/SEQUENCE_EXPERIMENTS.md for the data contract.

ImageNet-1K uses the official DeiT III S/B/L training recipes with LSSO ranks
32/48/64; see [docs/IMAGENET_DEIT3.md](docs/IMAGENET_DEIT3.md). COCO 2017
Mask R-CNN + FPN 3x and ADE20K UperNet 160k use the shared dense DeiT III
backbone and explicit padded-image masking. Their protocol provenance and
launch commands are in [docs/DOWNSTREAM_PROTOCOLS.md](docs/DOWNSTREAM_PROTOCOLS.md).
