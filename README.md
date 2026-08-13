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

## Sequence results

GenomicBenchmarks and LRA use the same sequence runner, validation-based
checkpoint selection, and one final test evaluation. Formal GenomicBenchmarks
runs compare MHA and LSSO in the same `D128/L4/H4` shell over three seeds. LSSO
is higher on 7 of 8 tasks; its unweighted macro accuracy is 78.53%, versus
75.37% for MHA (+3.16 percentage points).

| GenomicBenchmarks panel | MHA | LSSO |
| --- | ---: | ---: |
| 8-task macro test accuracy | 75.37 | **78.53** |
| Tasks with higher accuracy | 1/8 | **7/8** |

The formal LRA panel reports three-seed, validation-selected test accuracy for
the current task-specific shared shells:

| Task | LSSO accuracy (mean +/- std) |
| --- | ---: |
| ListOps | 37.60 +/- 0.65 |
| Text | 65.19 +/- 0.46 |
| Retrieval | 82.88 +/- 0.08 |
| Pathfinder-32 | 78.79 +/- 0.34 |

Published LRA, Transformer, and S4 numbers are useful external context only.
They use different model shells and training protocols, so they are not an
apples-to-apples comparison with this table and are not pooled or ranked here.
No sequence result above is presented as a state-of-the-art claim.

After placing the datasets at the roots recorded in the config files, the full
three-seed panels can be reproduced on Linux with the CUDA runtime loaded:

~~~bash
DNA_DATA_ROOT=/path/to/genomic_benchmarks
LRA_DATA_ROOT=/path/to/lra
SEQUENCE_CACHE=/path/to/sequence_cache

genomic_tasks=(
  dummy_mouse_enhancers_ensembl
  demo_coding_vs_intergenomic_seqs
  demo_human_or_worm
  human_enhancers_cohn
  human_enhancers_ensembl
  human_ensembl_regulatory
  human_nontata_promoters
  human_ocr_ensembl
)
for task in "${genomic_tasks[@]}"; do
  batch_args=(--batch-size 128 --grad-accum 1)
  if [[ "$task" == dummy_mouse_enhancers_ensembl ]]; then
    batch_args=(--batch-size 64 --grad-accum 2)
  fi
  for mixer in mha lsso; do
    for seed in 0 1 2; do
      python -m experiments.train_transformers \
        --config experiments/configs/genomic.toml \
        --task "$task" --mixer "$mixer" --seed "$seed" --formal \
        "${batch_args[@]}" \
        --data-root "$DNA_DATA_ROOT" --cache-root "$SEQUENCE_CACHE" \
        --output "runs/sequence/genomic/$task/$mixer/s$seed"
    done
  done
done

for task in listops text retrieval pathfinder; do
  for seed in 0 1 2; do
    python -m experiments.train_transformers \
      --config experiments/configs/lra.toml \
      --task "$task" --mixer lsso --seed "$seed" --formal \
      --data-root "$LRA_DATA_ROOT" --cache-root "$SEQUENCE_CACHE" \
      --output "runs/sequence/lra/$task/lsso/s$seed"
  done
done
~~~

The runner pins source and split provenance, keeps the compared DNA backbones
matched, and refuses dirty or mutable formal inputs. See
[docs/SEQUENCE_EXPERIMENTS.md](docs/SEQUENCE_EXPERIMENTS.md) for per-task
results, exact recipes, data layout, and protocol boundaries.

ImageNet-1K uses the official DeiT III S/B/L training recipes with LSSO ranks
32/48/64; see [docs/IMAGENET_DEIT3.md](docs/IMAGENET_DEIT3.md). COCO 2017
Mask R-CNN + FPN 3x and ADE20K UperNet 160k use the shared dense DeiT III
backbone and explicit padded-image masking. Their protocol provenance and
launch commands are in [docs/DOWNSTREAM_PROTOCOLS.md](docs/DOWNSTREAM_PROTOCOLS.md).
