# Core Contract

The operator accepts x[B,N,D], an optional boolean valid_mask[B,N], and
optional position_ids[N] or position_ids[B,N]. It returns y[B,N,D]. Batch size
B and sequence length N must be positive. The reference accepts `float16`,
`bfloat16`, `float32`, and `float64`; the native CUDA implementation accepts
only `float16` and `bfloat16` public inputs. Outputs match the input dtype. A
sequence may be entirely masked; its output is zero.

With Rank-Rotary enabled, position IDs must use an integer dtype, `float32`,
or `float64`. Integer coordinates are differenced before conversion to the
calculation dtype. `float64` coordinates are centered in FP64 before the
relative coordinates are converted to the calculation dtype; `float32`
coordinates use the calculation dtype directly. `float16` and `bfloat16`
position IDs are rejected because their coordinate precision can be lost before
the relative phase is formed.

For each head, let A be the relation coordinates after optional Rank-Rotary and
normalization by sqrt(n_valid), and let C be the masked content coordinates.
The QR soft frame and shared compact state are

~~~
P = qr_soft_frame(A)
Z = P^T C.
~~~

For DYNAMIC mode, the compact coordinates are generated from that same state:

~~~
R = R0 + Z W_drive / sqrt(n_valid).
~~~

STATIC uses R = R0. ZERO owns no compact coordinates and applies the zero
compact-core branch directly:

~~~
Y = eta (C - P Z).
~~~

For DYNAMIC and STATIC, define

~~~
L = tril(R, -1) + Diag(softplus(diag(R) + softplus_inverse(1)))
Omega = triu(R, 1) - triu(R, 1)^T
K = L L^T + Omega
U = solve(I + K, Z).
~~~

The symmetric part of I + K is I + L L^T, so the equilibrium is unique. The
per-head complement and token output before the output projection are

~~~
eps_d = finfo(calculation_dtype).eps
eta = (1 - eps_d) tanh(eta_raw)
Y = eta C + P [2 U - (1 + eta) Z].
~~~

Equivalently, M = 2 (I + K)^-1 - I and

~~~
Y = eta C + P (M - eta I) P^T C.
~~~

At R = 0, L = I, Omega = 0, U = Z / 2, and M = 0. DYNAMIC and STATIC both
start at this compact point; DYNAMIC additionally starts with W_drive = 0.
The direct solve preserves the nonzero initialization gradient without a
zero-value correction branch.

In exact arithmetic, the QR frame, accretive generator, and eta
parameterization make the frozen token mixer contractive. The fixed one-ULP
interior scale keeps the realized FP32 and FP64 complement strictly inside the
unit interval. The reference evaluates `tanh` through sign-specific stable
logistic identities, so its tail gradient remains nonzero when a direct FP32
`tanh` forward would round to `+/-1`, without overflowing in the opposite
inactive branch. Like every finite-precision exponential, this does not claim
meaningful tail gradients for astronomically extreme raw coordinates.

Rank-Rotary acts only on rank-space relation phases. It is an internal spectral
coordinate choice, not an absolute token-position representation.

The frame, compact-state storage, accretive factor Gram `F F^T`, and solve
calculations use FP32 unless the input is FP64. CUDA evaluates the factor Gram
with IEEE FP32 FMA; its small size makes avoiding a second factor quantization
worthwhile. Ordinary projections and eligible compact contractions use BF16
multiplicands with FP32 accumulation. Packed activations and the pre-output
boundary use BF16. Rank-Rotary stores bounded sin/cos in FP16 after FP32
trigonometry, then rotates in FP32. Sensitive eta and solve state stay FP32.
FP16 and BF16 public inputs are accepted by CUDA; the result matches the input
dtype. FP32/FP64 inputs remain available on the reference path. Invalid tokens
are zeroed before every compact statistic.

## Serialized and numerical boundaries

The current model `_extra_state` contract is version **12**. This is separate
from native CUDA ABI **8** and the ImageNet runner envelope format **5**.
Loading requires every saved operator-contract field to match, including model
geometry and ablations. Missing or mismatched contracts fail even under
`strict=False`; older weights need explicit validation before migration.

Biased low-precision CUDA projections add FP32 bias to an FP32 accumulator
before a single FP16/BF16 store. They use a lazily compiled Triton GEMM;
parameter gradients retain the existing FP32 reduction boundaries. Summation
order can change rounding, so algebraic equivalence is not a promise of
bitwise-identical training. See [CUDA implementation details](CUDA_CONTRACT.md).

The contraction statement freezes the input-conditioned frame and generator.
It does not bound the complete input Jacobian, which also differentiates those
quantities. Implementation benchmarks do not establish downstream accuracy.
