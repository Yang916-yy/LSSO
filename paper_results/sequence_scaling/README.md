# Sequence Scaling Benchmark

Single-layer bidirectional token-mixer benchmark on an NVIDIA GeForce RTX 5070
Ti with PyTorch 2.12.0, CUDA 13.0, and bf16 autocast.

Configuration:

```text
batch=8
dim=256
heads=8
sequence lengths=128, 256, 512, 1024, 2048
Nystromformer landmarks=64
LSSO gamma_max=0.3
LSSO theta_gamma_init=-4.0
```

Latency is the median of 25 CUDA-event measurements after 8 warmup iterations.
Forward+backward excludes the optimizer update. Incremental peak memory is
`torch.cuda.max_memory_allocated()` during one forward+backward step minus the
allocated baseline immediately before that step. It excludes unrelated
processes and reserved-but-unused cache. Mixer MACs are theoretical MACs for
one sample and one mixer layer; timing and memory use batch 8.

## Longest Sequence

| Mixer | Forward (ms) | Forward+backward (ms) | Incremental peak memory (MiB) | Mixer MACs (G) |
| --- | ---: | ---: | ---: | ---: |
| MHA | 0.637 | 2.614 | 121.0 | 2.684 |
| Nystromformer | 2.683 | 7.417 | 263.8 | 0.641 |
| LSSO-r16 | 0.570 | 1.877 | 124.8 | 0.357 |
| LSSO-r32 | 0.636 | 1.946 | 136.9 | 0.454 |

At `N=2048`, LSSO-r16 uses 86.7% fewer theoretical mixer MACs than MHA and is
1.39x faster for forward+backward in this operator benchmark. LSSO-r32 uses
83.1% fewer MACs and is 1.34x faster for forward+backward.

Peak allocated memory is close to optimized MHA rather than dramatically lower.
Current PyTorch MHA uses the memory-efficient Flash/SDPA path, which avoids
materializing the full attention matrix. The main demonstrated advantage here
is arithmetic scaling and the long-sequence latency crossover, not a universal
peak-memory reduction.

Artifacts:

- `sequence_scaling.csv`: all raw benchmark rows and quartiles.
- `sequence_scaling.png`: four-panel paper figure.
- `metadata.json`: software, hardware, arguments, and metric definitions.

Reproduce:

```bash
python benchmarks/benchmark_sequence_scaling.py
```
