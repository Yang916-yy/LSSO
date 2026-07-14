# LSSO/RRLSSO formal experiment plan

## 1. Research questions

The experiments should answer four questions rather than only compare final
accuracy:

1. Can an input-conditioned low-rank solve replace MHA in a controlled plain
   encoder while retaining task quality?
2. Does rank rotation consistently help when position matters?
3. Do the linear-in-token solve path and its signed operator translate into
   measured speed/memory benefits against optimized SDPA, not only lower FLOPs?
4. Does the same operator transfer beyond image classification to dense vision,
   genomic sequences, byte-level language/retrieval/reasoning, and continuous
   multivariate time series?

## 2. Models and fairness rules

### Core variants

- `MHA`: PyTorch scaled-dot-product attention, allowing the fastest available
  Flash/SDPA kernel.
- `LSSO-r32`: per-head LSSO with the current RMS and effective-length
  normalization.
- `RRLSSO-r32`: per-head rank-rotary LSSO. This is the proposed main model.
- External classification-only references: Performer, Nystromformer, and
  WeightFormer when their official implementations can be integrated without
  changing the surrounding backbone.

Grouped variants are excluded. All primary comparisons use the same embedding
width, depth, number of heads, MLP ratio, normalization, residual paths,
stochastic depth, input resolution, optimizer, augmentation, and number of
updates. Parameter count is reported rather than artificially adding unused
parameters to RRLSSO.

### Backbone hierarchy

The main CV tables use **VisionLLaMA (ECCV 2024)** rather than the original
torchvision ViT. It is recent, strong, has official ImageNet/COCO/ADE20K code,
and remains structurally simple: pre-normalized blocks, LayerNorm, SwiGLU, and
2-D rotary position encoding. Replace only its self-attention module with LSSO
or RRLSSO and retain every other component and official recipe.

- ImageNet main table: plain VisionLLaMA-S/B.
- COCO and ADE20K main tables: Pyramid VisionLLaMA-S/B, using its official
  Mask R-CNN and UPerNet configurations.
- Controlled operator table: plain DeiT-III-style ViT-S/B with identical
  width/depth across mixers.
- Mechanism/efficiency table: the minimal torchvision ViT used by the current
  project. This is not used to claim state-of-the-art competitiveness.

This two-level design separates two claims: the controlled DeiT table isolates
the mixer, while the VisionLLaMA table demonstrates competitiveness in a recent
strong backbone.

### Controlled plain-ViT scales

| Scale | Width | Depth | Heads | MLP ratio | Main rank |
|---|---:|---:|---:|---:|---:|
| ViT-S/16 | 384 | 12 | 6 | 4 | 32 |
| ViT-B/16 | 768 | 12 | 12 | 4 | 32 |

The S model is the statistical and ablation model. The B model is the principal
scaling model. The current torchvision ViT-B/16 Food-101 run remains a pilot
and transfer result; it does not replace ImageNet-1K as the main classification
benchmark.

### Position policy

For images and spectrograms, reuse VisionLLaMA's AS2DRoPE coordinate and
frequency construction. MHA applies it to queries/keys; RRLSSO applies the same
2-D rotations to its relation basis. Keep the current flattened 1-D rotation as
an ablation. This avoids row-wrap artifacts and prevents one mixer from seeing
better position information than another.

### Pretraining fairness

- Train a separate ImageNet checkpoint for every mixer.
- Initialize COCO, ADE20K, and AST from the corresponding checkpoint of the
  same mixer; never initialize RRLSSO from an MHA mixer or leave only one mixer
  random.
- Use identical data order and augmentation RNG streams within each paired
  seed when practical.
- Select hyperparameters on ViT-S validation runs, freeze them, then run
  ViT-B. Do not tune one mixer on the test/validation result of another.

## 3. Main CV suite

### 3.1 Image classification

**Dataset:** ImageNet-1K train/validation at 224x224.

**Strong-backbone recipe:** the official plain VisionLLaMA supervised
ImageNet-1K 300-epoch recipe. Use its LayerNorm, SwiGLU, AS2DRoPE, optimizer,
augmentation, stochastic depth, and regularization unchanged. Only the mixer
class and its rank-specific parameters may differ.

**Controlled recipe:** DeiT-III-style supervised training for the plain ViT
table. This table is intentionally separate from the VisionLLaMA results.

**Runs:**

- VisionLLaMA-S: MHA, LSSO-r32, RRLSSO-r32, three seeds each.
- VisionLLaMA-B: MHA and RRLSSO-r32, three seeds for the definitive table; LSSO-r32
  may be one seed if budget is constrained.
- Reproduce WeightFormer-S/B from its official 2026 repository as the most
  direct dynamic-parameter baseline. Also report official Vision Mamba/VMamba
  and Hiera numbers only when pretraining data, resolution, and evaluation
  protocol match; otherwise place them in a non-controlled reference table.

**Metrics:** top-1, top-5, NLL, ECE, parameters, model FLOPs, training images/s,
inference images/s, and peak training memory. Report mean and standard
deviation for repeated runs.

**Secondary transfer:** Food-101, CIFAR-100 at 224, and optionally iNaturalist
2018, all initialized from the corresponding ImageNet checkpoint. Food-101 is
reported as transfer rather than as the primary evidence.

### 3.2 Object detection and instance segmentation

**Dataset:** COCO 2017 train/val.

**Framework:** the official Pyramid VisionLLaMA Mask R-CNN configuration. Keep
its feature pyramid, neck, heads, and training recipe unchanged. ViTDet remains
a supplementary plain-backbone scaling experiment, not the main detector.

**Controlled configuration:** use the official supervised-ImageNet Pyramid
VisionLLaMA 3x COCO recipe, BF16, and effective batch 16. Use the same
stage/window pattern for all mixers.

**Additional global-context configuration:** run RRLSSO with every block global
while retaining the same detector. This is reported separately from the
matched-window comparison; it tests whether the linear token dependence can
buy a larger receptive field in practice.

**Runs:** ViT-S three seeds for MHA/LSSO/RRLSSO; ViT-B one full run for MHA and
RRLSSO, followed by extra seeds only if their difference is within normal COCO
run variance.

**Metrics:** AP-box, AP50, AP75, AP-small/medium/large, AP-mask, train
iterations/s, inference latency, and peak memory. Use single-scale evaluation
as the primary result.

### 3.3 Semantic segmentation

**Dataset:** ADE20K train/val, 150 classes.

**Framework:** the official Pyramid VisionLLaMA UPerNet configuration in
MMSegmentation. Keep all decoder and feature-pyramid settings fixed.

**Recipe:** 512x512 random crops, 160k iterations, AdamW, polynomial decay,
effective batch 16, BF16. Initialize from the corresponding ImageNet-1K
checkpoint.

**Runs:** ViT-S, three seeds for all three core mixers; ViT-B, one run for MHA
and RRLSSO followed by a repeat if the result is close.

**Metrics:** single-scale mIoU and mAcc are primary; multi-scale+flip mIoU is
secondary. Also report throughput and peak memory at 512 and 640 crops.

## 4. Additional modalities

The former MS MARCO/BEIR and FLIP/TAPE/ProteinGym plan is retired. The formal
auxiliary suite now consists of public, fixed-protocol benchmarks for which
published baselines can be cited without reproducing a large operator grid.
Detailed protocol and citation rules are maintained in
`docs/auxiliary_experiments.md`.

### 4.1 GenomicBenchmarks

Run RRLSSO on all eight official GenomicBenchmarks classification tasks with
single-nucleotide tokenization and three fixed seeds. Since the benchmark only
provides train/test splits, make a deterministic stratified 90/10 validation
split from the official training data and never tune on test. Report accuracy,
macro F1 where appropriate, and mean rank across tasks. Compare against
published supervised CNN/Transformer results; show HyenaDNA and Caduceus in a
separate pretrained-model block when their pretraining differs.

### 4.2 Long Range Arena

Run the official ListOps, byte-level Text, AAN Retrieval, and Pathfinder
tasks. Omit sequential CIFAR-10 and Path-X. Match the official model dimensions,
training protocol, tokenization, and parameter tolerance so that the official
Transformer/efficient-attention table and published S4/S5/MEGA results remain
valid external baselines. Report every task and the four-task mean.

### 4.3 UEA-30 multivariate time series

Run all 30 datasets under the official archive splits. Report per-dataset
accuracy, mean rank, average accuracy, and wins/ties/losses against cited DTW,
ROCKET-family, InceptionTime, ConvTran, and Transformer results. Do not import a
baseline into the reported standard-split block when its archive version,
official split, or metric differs; method-specific preprocessing is allowed
but must be documented and is not described as a controlled reproduction.

### 4.4 Minimal reproduction policy

The auxiliary default is RRLSSO-only. Published compatible baselines are cited
with a dagger and source. One MHA anchor per suite is permitted solely to
validate the local pipeline; the full MHA/LSSO/RRLSSO grid remains confined to
the main CV experiments unless later evidence makes another run necessary.

## 5. Required ablations

Run the full ablation suite on ImageNet-1K VisionLLaMA-S unless stated
otherwise. Use one screening seed, then repeat the informative settings with
three seeds.

- Rank: 8, 16, 32, 64.
- Operator: no-global, LSSO, flattened-1D RRLSSO, AS2D-RRLSSO.
- Basis preprocessing: no RMS normalization, RMS only, RMS plus effective
  length normalization.
- Solve scales: fixed versus learned mu/gamma; report layer-wise gamma/mu.
- Exact solve versus shared-weight explicit iteration with K=1,2,4,8 steps.
  Mark runs violating the Neumann contraction diagnostic rather than silently
  treating the series as universally valid.
- Position: absolute-only, MHA-AS2DRoPE, RRLSSO-AS2DRoPE.
- Initialization: paired random initialization and checkpoint-transfer
  sensitivity.
- Resolution/token extrapolation: 224, 384, 512 classification crops and
  synthetic token lengths up to 4096.

Diagnostic plots include effective rank, correction ratio, gamma/mu, signed
positive/negative mass, cancellation ratio, and layer-wise operator spectra.
Signed maps must not be reduced to absolute values.

## 6. Systems evaluation

Benchmark both an isolated mixer layer and complete models on the same A100
80GB, CUDA/PyTorch versions, and clock/power policy. Keep optimized SDPA enabled
for MHA and the native MathDx/CUDA path enabled for LSSO/RRLSSO.

- Token lengths: 197, 577, 1025, 2049, 4097.
- Report forward and forward+backward latency, end-to-end training throughput,
  inference throughput, peak allocated memory, and theoretical FLOPs/MACs.
- Include fixed-batch and maximum-fitting-batch measurements.
- Warm up at least 50 iterations and measure at least 200; report median and
  interquartile range.
- Separate model-only timings from dataloader-inclusive timings and keep data
  on local NVMe/cache to avoid attributing IO stalls to the mixer.
- Plot quality-versus-throughput and quality-versus-memory Pareto frontiers.

## 7. Statistics and reporting

- Pre-register primary metrics: ImageNet top-1, COCO box AP, ADE20K mIoU,
  GenomicBenchmarks eight-task mean rank, LRA four-task mean accuracy, and
  UEA-30 mean rank.
- Report all seeds, not only the best checkpoint. Select checkpoints using a
  validation metric fixed before training.
- Use mean and standard deviation for seed repeats. Use paired per-dataset
  comparisons and a critical-difference analysis for UEA-30 where appropriate.
- Report failures, numerical fallbacks, OOMs, effective batch, accumulation,
  wall-clock time, and total A100-hours.
- Separate reproduced baselines from numbers copied from papers. A copied
  number enters the comparison table only when dataset, pretraining data,
  resolution, and evaluation protocol match exactly.

## 8. A100 80GB execution order and planning budget

The ranges below are planning estimates for one A100 80GB and must be replaced
by extrapolation from a 200-step pilot on the rented instance.

1. Port and validate VisionLLaMA-S on ImageNet-100: 0.5-1 GPU-day.
2. ImageNet-1K VisionLLaMA-S core models, three seeds: roughly 6-12 GPU-days.
3. VisionLLaMA-B MHA/RRLSSO main runs: roughly 6-12 GPU-days for one seed each;
   repeat only after the pipeline is frozen.
4. COCO Pyramid VisionLLaMA-S/B: roughly 4-8 GPU-days per core comparison.
5. ADE20K Pyramid VisionLLaMA-S/B: roughly 2-5 GPU-days per core comparison.
6. Pilot the GenomicBenchmarks eight-task runner and extrapolate its full cost.
7. Run the four fixed LRA tasks after verifying exact official configurations.
8. Run UEA-30 last because its many independent datasets dominate orchestration
   and statistical reporting, even when individual models are small.

Run order is pilot -> VisionLLaMA-S classification -> rank/position ablations ->
VisionLLaMA-B -> COCO/ADE20K -> GenomicBenchmarks -> LRA -> UEA-30. A failed
pilot must stop the corresponding family before long training begins.
