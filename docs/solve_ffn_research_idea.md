# Solve-FFN: input-conditioned implicit channel refinement

> Status: independent research direction recorded on 2026-07-14. This idea is
> not part of the current RRLSSO experimental protocol and should first be
> validated as a standalone FFN replacement with the token mixer held fixed.

## 1. Core idea

KAN is only a source of motivation: a conventional MLP with fixed pointwise
activation and a single explicit feed-forward pass is not the only possible
way to transform channels. Solve-FFN does not use B-splines and should not be
presented as a KAN variant.

The proposed block separates two roles:

1. a mature explicit nonlinearity such as GELU, SiLU, or SwiGLU generates a
   candidate hidden state;
2. an input-conditioned convex channel energy refines that candidate, and its
   unique equilibrium is evaluated by a low-rank closed-form solve.

In short:

\[
\boxed{
\text{explicit nonlinear candidate}
\;\longrightarrow\;
\text{implicit channel equilibrium}
\;\longrightarrow\;
\text{output projection}
}
\]

This should be described as latent channel inference, not merely as
"simulating an MLP."

## 2. Correct mathematical formulation

For an input token \(x\in\mathbb{R}^D\), first form an explicit hidden state
\(a\in\mathbb{R}^m\):

\[
a=\phi(W_{\mathrm{in}}x).
\]

Define the channel energy

\[
\boxed{
E(z;x)
=
\frac{1}{2}\lVert z-a\rVert_2^2
+
\frac{\alpha}{2}
\lVert U(x)^\top z-c(x)\rVert_2^2,
\qquad \alpha\ge 0.
}
\]

The second term must have a **positive** sign. A negative sign would produce
the Hessian \(I-\alpha UU^\top\), remove the unconditional SPD guarantee, and
lead to a different solution.

The first-order condition is

\[
(I+\alpha UU^\top)z=a+\alpha Uc.
\]

Therefore

\[
\boxed{
z^\star
=
(I+\alpha UU^\top)^{-1}(a+\alpha Uc).
}
\]

Using Woodbury,

\[
\boxed{
z^\star
=
a+
\alpha U
(I+\alpha U^\top U)^{-1}
(c-U^\top a).
}
\]

The block output is then

\[
y=x+W_{\mathrm{out}}z^\star.
\]

The solve is also the proximal map

\[
z^\star
=
\operatorname{prox}_{\frac{\alpha}{2}
\lVert U(x)^\top(\cdot)-c(x)\rVert^2}(a).
\]

It balances fidelity to the explicit MLP candidate with satisfaction of an
input-dependent low-rank channel target.

## 3. Dynamic low-rank parameterization

A full token-dependent matrix \(U(x)\in\mathbb{R}^{m\times r}\) is too
expensive to generate. Use a shared learned channel basis and token-dependent
gates:

\[
U(x)=P D_g(x),
\qquad
D_g(x)=\operatorname{Diag}(g(x)),
\]

\[
g(x)=\sigma(W_gx),
\qquad
c(x)=W_cx.
\]

With \(G=P^\top P\), the low-rank system becomes

\[
q=
(I+\alpha D_gGD_g)^{-1}
(c-D_gP^\top a),
\]

\[
\boxed{z=a+\alpha P D_gq.}
\]

The same basis is reused across tokens, but each token changes the solve
metric and target through \(g(x)\) and \(c(x)\).

### Why input conditioning is necessary

If \(U\) is fixed and the right-hand side contains only \(a\), then

\[
W_{\mathrm{out}}(I+\alpha UU^\top)^{-1}
\]

can be folded into a new fixed output projection. A fixed \(U\) with a merely
linear \(c(x)\) also largely reduces to a fixed transformed MLP plus a linear
residual branch. The meaningful function-class change therefore requires at
least one of:

- input-dependent \(U(x)\), preferably through \(g(x)\);
- a genuinely nonlinear target \(c(x)\);
- alternating nonlinear and solve operations.

The minimal main candidate is dynamic \(g(x)+c(x)\).

## 4. Infinite-depth interpretation

The raw fixed-point iteration is

\[
z^{(k+1)}
=
a+\alpha Uc-\alpha UU^\top z^{(k)},
\]

which converges when \(\rho(\alpha UU^\top)<1\).

A stronger statement uses Richardson residual iteration:

\[
z^{(k+1)}
=
z^{(k)}
-\eta\left[
(I+\alpha UU^\top)z^{(k)}-(a+\alpha Uc)
\right].
\]

Because the system is SPD, this iteration converges whenever

\[
0<\eta<
\frac{2}{1+\alpha\lambda_{\max}(UU^\top)}.
\]

Thus the closed form is the equilibrium of an infinite-depth, weight-shared
linear residual correction network without requiring the restrictive raw
Neumann condition.

## 5. What it can offer over an ordinary MLP

Solve-FFN does not strictly dominate an arbitrarily wide MLP in universal
approximation. Its possible advantages are architectural and statistical:

1. **Input-dependent post-activation channel operator.** An ordinary FFN has
   a fixed output projection after its pointwise activation. Solve-FFN inserts
   the token-dependent dense low-rank operator
   \((I+\alpha U(x)U(x)^\top)^{-1}\).
2. **Structured joint correction.** It explicitly balances closeness to the
   candidate \(a\) with satisfaction of the low-rank target \(c(x)\), rather
   than leaving all hidden-channel coordination to a fixed projection.
3. **Implicit effective depth.** A closed-form equilibrium may replace some of
   the capacity otherwise obtained by increasing FFN expansion.
4. **Well-posedness.** For fixed conditioning,
   \(I+\alpha UU^\top\succ0\), the solution is unique, and
   \(\lVert(I+\alpha UU^\top)^{-1}\rVert_2\le 1\).
5. **Interpretability.** Gates, solve directions, target residuals, and energy
   reduction can be inspected directly.

The non-expansive bound applies to the conditional map from the right-hand
side to the solution. It does **not** prove that the complete dynamic map
\(x\mapsto z(x)\) is globally non-expansive, because derivatives of \(U(x)\)
and \(c(x)\) also enter its Jacobian.

### Parameter-efficiency hypothesis

A GELU FFN with expansion \(m=4D\) has roughly

\[
8D^2
\]

projection parameters. A Solve-FFN with \(m=2D\) has roughly

\[
4D^2+O(Dr)
\]

parameters. The decisive practical hypothesis is therefore:

\[
\boxed{
\text{Solve-FFN at }2D
\;\ge\;
\text{ordinary MLP/SwiGLU at }4D
}
\]

at lower parameter count and acceptable measured throughput. If Solve-FFN
only beats an equally narrow MLP while remaining slower than a wide MLP, the
practical contribution is weak.

## 6. Structured solve variants

### 6.1 Full-Solve

Learn unconstrained \(P\) and solve the full \(r\times r\) system per token.
This is the expressivity upper bound, but batched Cholesky costs
\(O(BNr^3)\) and may be dominated by many small solves and memory traffic.

### 6.2 Semi-Orthogonal Solve

If

\[
P^\top P=I,
\]

then

\[
q_j=
\frac{c_j-g_jp_j^\top a}{1+\alpha g_j^2},
\]

and

\[
z=a+\alpha P
\left[
g\odot
\frac{c-g\odot(P^\top a)}{1+\alpha g^2}
\right].
\]

This eliminates Cholesky and reduces the solve to projections plus elementwise
rational gates. It is the efficiency bound, but it fixes the solve eigenvectors
and may be viewed as a gated low-rank adapter rather than a genuinely coupled
channel solve.

For static \(U\), semi-orthogonalization can absorb singular values into the
gates without meaningful loss. In the dynamic case, non-orthogonal basis
vectors allow the eigenvectors of \(U(x)U(x)^\top\) to change as the gates
change; strict semi-orthogonality removes this capability and can reduce
accuracy.

### 6.3 Block-Solve (recommended main form)

Use

\[
P=QR,
\qquad Q^\top Q=I,
\]

with block-diagonal \(R\) and block size \(b=4\) or \(8\). Each token solves
several \(b\times b\) SPD systems. The cubic part becomes

\[
O\left(\frac{r}{b}b^3\right)=O(rb^2)
\]

instead of \(O(r^3)\), while preserving within-block dynamic coupling.

The intended hierarchy is:

| Variant | Role |
|---|---|
| Full-Solve | accuracy and expressivity upper bound |
| Block-Solve, \(b=4/8\) | proposed main architecture |
| Semi-Orthogonal Solve | efficiency lower bound |

Chunking controls peak memory but does not remove total solve FLOPs. Kernel
fusion and block structure are therefore more important than chunking alone.

## 7. Relationship to prior work and novelty boundary

The broad ingredients are not individually new:

- [KAN](https://arxiv.org/abs/2404.19756) replaces fixed MLP activations with
  learnable edge functions; Solve-FFN uses mature activations and an implicit
  channel state instead.
- [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377),
  [Implicit Deep Learning](https://epubs.siam.org/doi/10.1137/20M1358517), and
  [monDEQ](https://arxiv.org/abs/2006.08591) use implicit infinite-depth
  representations, generally obtained by iterative root finding.
- [OptNet](https://proceedings.mlr.press/v70/amos17a.html) and
  [Deep Declarative Networks](https://arxiv.org/abs/1909.04866) establish that
  an optimization problem can be used as a differentiable network layer.
- [HyperNetworks](https://research.google/pubs/hypernetworks-2/),
  [Dynamic Filter Networks](https://arxiv.org/abs/1605.09673), and
  [CondConv](https://arxiv.org/abs/1904.04971) establish input-conditioned
  weights and operators.
- [Hopfield Networks is All You Need](https://arxiv.org/abs/2008.02217)
  relates attention to energy-based associative retrieval.
- [Mesa-optimization](https://arxiv.org/abs/2309.05858) places a learned
  least-squares-style algorithm in the forward pass, but operates primarily as
  a sequence/history mixer for in-context learning.

Accordingly, the paper must not claim to be the first implicit network,
optimization layer, equilibrium layer, or dynamically conditioned operator.
The potentially novel intersection is:

\[
\boxed{
\text{a drop-in Transformer FFN replacement based on an}
\\
\text{input-conditioned, low-rank, convex quadratic channel equilibrium,}
\\
\text{with an exact structured closed-form implementation.}
}
\]

This novelty claim requires a dedicated literature review before submission.

## 8. Minimal implementation plan

Keep MHA and the surrounding backbone unchanged. Implement a reference
Solve-FFN with:

- GELU first; SwiGLU only after the base mechanism works;
- expansion \(1\), \(2\), and \(4\);
- rank \(8,16,32,64\);
- positive \(\alpha=\operatorname{softplus}(\hat\alpha)\), initialized near
  zero so the model begins close to the ordinary FFN;
- Full, Block-4, Block-8, and Orthogonal variants;
- a numerically stable FP32 solve under mixed precision;
- a custom backward or recomputation path only after the reference result is
  positive.

The exact implicit backward for the SPD system requires another solve with the
same matrix. It should avoid unrolling the infinite-depth interpretation.

## 9. Required controls and experiments

The first study must isolate the FFN. Do not combine it with RRLSSO.

### Baselines

- MHA + GELU MLP, expansion 4 and 2;
- MHA + SwiGLU with its standard parameter-matched width;
- MHA + dynamic gated low-rank residual without an inverse;
- MHA + one explicit correction step
  \(a+\alpha U(c-U^\top a)\);
- explicit shared correction for 2, 4, and 8 steps;
- closed-form Solve-FFN.

The dynamic low-rank residual baseline is essential: it determines whether any
gain comes from the equilibrium solve or merely from adding \(P,W_g,W_c\).

### Ablations

- static \(U\) versus dynamic \(g(x)\);
- static, linear, and nonlinear \(c(x)\);
- Full versus Block-8 versus Block-4 versus Orthogonal;
- expansion \(4,2,1\);
- rank \(8,16,32,64\);
- GELU versus SiLU versus SwiGLU;
- matched parameters, matched theoretical FLOPs, and matched wall-clock
  training budgets.

### Reported engineering metrics

- model and FFN-only parameters;
- forward and training FLOPs;
- actual samples/tokens per second;
- peak memory;
- solve time versus projection time;
- condition numbers and Cholesky failures;
- energy before and after correction;
- gate and effective-rank statistics.

### Go/no-go criterion

The strongest result would be:

| FFN | Expansion | Expected interpretation |
|---|---:|---|
| MLP/SwiGLU | 4x or parameter-matched | mature baseline |
| Solve-FFN | 2x | matches or exceeds baseline with fewer parameters |
| Solve-FFN | 1x | remains close while substantially reducing FFN cost |

Proceed to a full paper only if the solve offers a favorable accuracy-versus-
parameter or accuracy-versus-wall-clock frontier. Beating only a narrow MLP at
higher runtime is insufficient.

## 10. Relationship to RRLSSO

The two modules act on orthogonal axes:

\[
\boxed{\text{RRLSSO: solve token relationships}}
\]

\[
\boxed{\text{Solve-FFN: solve channel states}}
\]

They should first be established independently:

1. MHA + MLP versus RRLSSO + MLP tests token mixing;
2. MHA + MLP versus MHA + Solve-FFN tests channel transformation;
3. only after both succeed, combine RRLSSO + Solve-FFN into a Full-Solve
   block.

This separation prevents the work from becoming an uninterpretable mixture of
KAN, attention replacement, and implicit FFN changes.

## 11. Open questions

1. Does dynamic channel coupling improve accuracy beyond a parameter-matched
   gated low-rank residual?
2. Can expansion 2 replace expansion 4 without reducing wall-clock speed?
3. Is Block-8 statistically indistinguishable from Full-Solve?
4. Does strict orthogonality regularize or underfit?
5. Should gates be token-wise, sample-wise, or shared over token groups to
   amortize factorization?
6. Does a nonlinear target \(c(x)\) justify its extra projection cost?
7. Can the solve and output projection be fused without materializing
   \(z\)?
8. Does the learned energy expose interpretable channel modes that ordinary
   SwiGLU lacks?

