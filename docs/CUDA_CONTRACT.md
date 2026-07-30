# CUDA Contract

Native contract version 6 implements exactly the complete default operator:

- `core_mode=DYNAMIC`;
- `rank_rotary=True`;
- learned per-head one-ULP interiorized tanh complement;
- the soft frame, shared compact state, dynamic accretive generator, and
  direct equilibrium readout defined by `lsso/ball/reference.py`.

STATIC, ZERO, and Rank-Rotary-off are PyTorch-only ablations. CUDA rejects
them explicitly. It never falls back to the reference implementation, retains
an old ABI, or redefines the operator mathematics outside `reference.py`.

## Public And Native Boundaries

`LSSO.forward(..., implementation="cuda")` is an explicit request. It accepts
CUDA activations in `torch.float16` or `torch.float32`. BF16 is deliberately
unsupported on this path. The model's TF32-enabled FP32 `w_bc` projection creates a
contiguous FP32 packed tensor, invokes the native mixer, then the shared
`tensor_core_linear` helper performs `w_o` before the public result is cast
back to `x.dtype`. All projection and dynamic-core parameters remain FP32.

The native mixer is stricter. Its `projected` input must be a contiguous CUDA
FP32 tensor of shape `[B, N, H * R + D]`; it does not accept FP16 or BF16
packed coordinates. `core_base_raw`, `core_drive_weight`, and `eta_raw` must
also be contiguous FP32 CUDA tensors on the same device. Optional
`centered_positions` is contiguous FP32 CUDA metadata of shape `[N]` or
`[B, N]`, not a differentiable input. When the public caller supplies a
`valid_mask`, it zeros masked packed coordinates and supplies contiguous FP32
`valid_counts[B] = max(n_valid, 1)`; this is also metadata and is not
differentiable. The native ABI rejects unsupported dtypes, layouts, shapes,
ranks, and devices explicitly;
`lsso.ball.cuda.fast_mix()` rejects gradient-bearing metadata before dispatch.
It does not interpret a token mask itself: direct `fast_mix()` callers must
zero padded packed coordinates, provide strictly positive finite valid counts,
and mask invalid upstream output gradients.

The supported native shape range is rank in `{16, 32, 48, 64}` and any positive
head dimension within ordinary CUDA allocation and launch limits. Rank and head
dimension are independent; rank may exceed head dimension. The public CUDA path
accepts `valid_mask[B, N]` and position IDs of shape `[N]` or `[B, N]`,
including gapped padding and an
entirely masked sample. Masking retains the one generic physical `[B, N]`
schedule: invalid coordinates are zero and every per-sample normalization uses
`sqrt(n_valid)`. It is therefore a semantic input of the current operator, not
a dense-only scheduling restriction or a fallback path.

The private native ABI exposes:

- `forward_inference`, which returns FP32 pre-output coordinates;
- `forward_train`, which returns those coordinates plus a private FP32 tape and
  one-based `int32` LU pivots, matching MathDx partial-pivot LU factors;
- `backward`, which consumes the tape through the first-order autograd
  boundary owned by `lsso.ball.cuda.fast_mix()`.

`forward_inference` and `forward_train` are not differentiable public
operators. Direct use with autograd-enabled inputs raises an error, and
higher-order gradients are unsupported.

## Numerical Contract

`reference.py` owns the mathematical and canonical numerical definition. The
same reference path run with FP64 tensors is the oracle for CUDA validation.
CUDA uses a mixed TF32/TC16 contract, not an end-to-end bitwise-FP32
implementation:

- `w_bc` uses FP32 operands with TF32 enabled on supported CUDA hardware and
  FP32 output/accumulation; `w_o` and selected compact contractions use FP16
  Tensor Core multiplicands with FP32 accumulation;
- the soft-frame Gram/factorization, Rank-Rotary phase construction,
  one-ULP interiorized complement, LU factorization, and triangular solve remain FP32; the
  accretive `F F^T` factor Gram specifically uses IEEE FP32 FMA;
- selected contractions include compact-state formation, dynamic-coordinate
  generation, and compact readout.

The factor Gram is algebraically still `F F^T`; native CUDA evaluates it with
FP32 FMA rather than quantizing `F` to FP16 first. This removes the dominant
small-matrix factor rounding error without adding a public numerical variant.

The complement's scalar backward reduction reconstructs its frame-content
term in FP32 from the recorded frame. This avoids narrow-head cancellation
from the TC16 compact-state tape; it changes neither the forward result nor
the Tensor Core boundaries of the compact operator.

The accepted CUDA accuracy envelope against the FP64 oracle is relative L2
error at most `5e-3` for forward outputs and at most `1e-2` for input and
parameter gradients; all compared tensors must be finite. Tests do not require
pointwise FP32 equality or finite-difference agreement from the reduced
precision path.

The frame retains the reference detached scaling and identity augmentation, and
the accretive parameterization preserves the normal-training-domain solve.
There is deliberately no recovery fallback or synchronous solver-info poll for
inputs outside that domain.

## Schedule

All supported ranks use one tiled generic workspace schedule. It builds the
FP32 relation/soft-frame state, computes compact tiles, fuses dynamic-coordinate
generation with accretive factor construction and LU factorization, then solves
the equilibrium and performs the TC16/FP32 readout. Default Rank-Rotary phase
tables are cached per device, rank, and sequence length and safely shared
across streams; explicit position metadata uses an invocation-local table.
Training records the state required by the tiled VJP;
inference uses a compact workspace without the training-only frame or
coordinate sections. Head dimensions use runtime 32-column RHS tiles with
zero-filled tails, so they have no second shape whitelist; very small or very
large heads trade throughput or memory for the same operator semantics.

Compact-state tiles are produced independently and reduced in ascending token
order in FP32. The internal producer workspace holds at most sixty-four 32-token
tiles at a time, so long sequences do not make its transient allocation grow
without bound. The QR-frame VJP builds its compact adjoint directly from the
saved sufficient statistics; token-level VJPs retain TC16 operands with FP32
accumulators.

For inputs with at least 32 relation tiles, frame Gram construction uses
independent 32-token FP32 partials and an ascending-token reduction before the
same Cholesky factorization. This improves long-sequence occupancy without a
public scheduling variant. Tape-producing `r=16/32/48` calls reuse the
not-yet-materialized `P[N, R]` tape region for these partials. Other calls use
a temporary lower-triangle workspace of
`B * H * ceil(N / 32) * R * (R + 1) / 2` FP32 values, released before the
cross-state producer workspace. Shorter sequences retain the single-CTA Gram
reduction.

There is no public scheduling switch and no rank-16/rank-32 single-CTA
specialization contract.

## Artifacts

CUDA 12.8 device-LTO with the cuSolverDx fatbin builds one executable image per
device-link invocation. `tools/build_cuda.sh` therefore produces strict
artifacts named `lsso_equilibrium_sm75.so`, `lsso_equilibrium_sm80.so`,
`lsso_equilibrium_sm86.so`, `lsso_equilibrium_sm87.so`,
`lsso_equilibrium_sm89.so`, `lsso_equilibrium_sm90.so`,
`lsso_equilibrium_sm100.so`, and `lsso_equilibrium_sm120.so`, rather than
claiming one universal binary.

SM75 is the minimum supported architecture. The contract uses FP16 Tensor Core
multiplicands with FP32 accumulators for its selected contractions, while QR and
the compact solve remain FP32. Turing has no TF32 path, so its throughput is an
architecture-specific performance question rather than a different numerical
contract.

`lsso.ball.cuda.load(device=...)` selects the artifact matching the requested
device. It serializes loading and binds one native operator implementation per
process, so heterogeneous SMs in one process are rejected explicitly. The
build script removes each selected SM's CMake directory and output artifact
before configuration, preventing a cached Torch_DIR from linking a newly
selected Python environment against an old libtorch. The loaded artifact
verifies both its compiled SM and native contract version before launching
kernels.
