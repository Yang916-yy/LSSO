# Causal Solve-TTT: unified equilibrium sequence learning

Status: research hypothesis for a separate project after the bidirectional
LSSO/RRLSSO paper. This note records the corrected scope after comparison with
TTT, DeltaNet, Longhorn, MesaNet, Gated KalmaNet, and preconditioned DeltaNet.

## Central thesis

Modern fast-weight sequence mixers can be organized as learned inner
regression problems. They differ primarily in four choices:

1. the temporal support of the inner objective;
2. the prior carried from previous states;
3. the curvature approximation and solver;
4. the bidirectional, recurrent, or chunkwise execution schedule.

The intended contribution is not the first use of Sherman--Morrison, online
ridge regression, residual writing, or an exact single-token update. The goal
is a common solve framework that connects global bidirectional equilibrium,
prefix-optimal causal equilibrium, and order-sensitive recursive equilibrium,
then derives an analytic block-proximal operator with full low-rank geometry.

## The contribution is the idea, not an isolated formula

The desired theory should be general in the same sense that residual learning
is general. A residual connection is algebraically simple, but it changes the
question from learning a complete mapping to learning a state correction. The
corresponding conceptual change here is

\[
\text{compress context into a state}
\quad\longrightarrow\quad
\text{train an internal learner on the context}.
\]

The central claim is that TTT, DeltaNet, Longhorn, MesaNet, and LSSO are not
unrelated memory mechanisms. They are different sequence-internal learning
algorithms. Their recurrent or bidirectional computations specify what data an
inner model sees, how prior knowledge is retained, and how accurately its
objective is solved.

This gives a minimal three-operation abstraction:

\[
\boxed{\text{Write}+\text{Learn}+\text{Read}}.
\]

Given an outer representation `x_t`, a learned feature generator constructs

\[
(U_t,C_t,Q_t,G_t)=\Phi_\theta(x_t),
\]

where `U_t,C_t` define inner training examples, `Q_t` is the read query, and
`G_t` controls weighting, regularization, plasticity, and forgetting. The
state transition and output become

\[
Z_t=\mathcal S(Z_{t-1};U_{\mathcal I_t},C_{\mathcal I_t},G_t),
\qquad
y_t=Q_tZ_{\tau(t)}.
\]

Here `I_t` selects the temporal support of the inner dataset, `S` is an
optimizer or solver, and `tau(t)` distinguishes read-before-write from
write-before-read. This abstraction is intended to be more fundamental than a
particular Sherman--Morrison recurrence.

### Axis one: temporal support of the inner objective

\[
\mathcal I_t=
\begin{cases}
\{t\}, & \text{token-local},\\
\text{current chunk}, & \text{block-local},\\
\{1,\ldots,t\}, & \text{causal prefix},\\
\{1,\ldots,N\}, & \text{bidirectional global}.
\end{cases}
\]

Thus sequence-internal learning spans

\[
\boxed{\text{local}\rightarrow\text{chunk}\rightarrow
\text{prefix}\rightarrow\text{global}}.
\]

### Axis two: accuracy of the inner solver

The state transition may implement

\[
\boxed{
\text{store}\rightarrow
\text{update}\rightarrow
\text{optimize}\rightarrow
\text{solve}
}.
\]

Representative points include additive sufficient-statistic storage, one
explicit gradient or Delta step, finite `K`-step TTT, an implicit proximal
update, an iterative solve to tolerance, and an analytic exact equilibrium.

The proposed theory places both axes in the same map. Temporal support answers
*which experience trains the inner learner*; solver accuracy answers *how far
that learning problem is solved*.

### Unified interpretation of the main families

- Linear attention writes unconditional outer products: storage without
  prediction correction.
- DeltaNet performs prediction-error learning with one residual update.
- Longhorn derives a token-local implicit equilibrium and then adopts a
  practical approximation.
- TTT explicitly treats the hidden state as an inner model trained for a
  finite number of steps.
- MesaNet optimizes a richer in-context objective toward a local optimum with
  an iterative solver.
- Bidirectional LSSO analytically solves a global sample-wise equilibrium.
- Causal Solve-LSSO applies the same learned solve language to token, chunk,
  or prefix objectives.

This perspective changes the architecture-design question. Rather than asking
only what recurrence or gate to invent, it asks:

1. What should the inner model learn?
2. What training example does each token create?
3. What prior or geometry preserves earlier learning?
4. How much of the inner objective should be solved?
5. Which objectives admit efficient recurrent, chunked, and global execution?

The intended analogy to residual learning is therefore conceptual rather than
historical: a simple mathematical primitive can be valuable when it changes
the decomposition of the problem and generates a reusable design language.

### Core paper language

The paper should not center the claim that it invents a new rank-one formula.
The stronger formulation is:

> We introduce a unified equilibrium-learning view of sequence modeling, in
> which the recurrent state is an inner learner, token mixing defines its
> training data, and the state transition specifies an optimizer or solver.

And:

> Attention retrieves from past examples, whereas equilibrium sequence
> learners train a model on those examples and retrieve from the trained
> model.

The shortest statement of the thesis is:

\[
\boxed{
\text{These are not different memory structures; they are different
sequence-internal learning algorithms.}
}
\]

The unified perspective becomes a full method paper when it also yields
strict equivalence and boundary theorems, a missing point in the design map,
and empirical predictions that distinguish objective scope, state prior,
curvature, and solver accuracy.

## A common inner objective

Consider a matrix-valued fast-weight state `Z` and the general update

\[
Z_t=\arg\min_Z\left[
\frac12\|Z-\mathcal F_t(Z_{t-1})\|_{M_t}^2
+\frac12\|U_tZ-C_t\|_{W_t}^2
\right].
\]

The basic scalar-metric proximal instance is

\[
\mathcal J_t(Z;Z_{t-1})=
\frac{\beta_t}{2}\|Z-Z_{t-1}\|_F^2+
\frac{\alpha_t}{2}\|U_tZ-C_t\|_F^2,
\]

with exact update

\[
Z_t=Z_{t-1}+\alpha_t
(\beta_tI+\alpha_tU_t^\top U_t)^{-1}
U_t^\top(C_t-U_tZ_{t-1}).
\]

For one token this reduces to

\[
Z_t=Z_{t-1}+
\frac{\alpha_t}{\beta_t+\alpha_t\|u_t\|^2}
u_t^\top(c_t-u_tZ_{t-1}).
\]

Only the ratio `rho_t = alpha_t / beta_t` is identifiable in this basic
objective. A minimal implementation should predict `rho_t > 0`, or directly
predict the contraction gate

\[
\lambda_t=\frac{\rho_t\|u_t\|^2}{1+\rho_t\|u_t\|^2}\in[0,1).
\]

## Exact relations to prior work

### DeltaNet

After normalizing the write direction, the single-token proximal update is a
Delta-rule fast-weight update. Residual-conditioned writing and the avoidance
of blind additive outer-product accumulation are therefore established
DeltaNet properties, not standalone novelty.

### Longhorn

The single-token objective and exact rank-one solution are isomorphic to the
unapproximated implicit online regression update derived by Longhorn. Longhorn
uses a diagonal approximation in its practical scan-friendly SSM. The proposed
work must not claim the first closed-form causal online learner.

### TTT and MesaNet

TTT treats the hidden state as an inner model updated by finite optimization.
MesaNet already studies locally optimal test-time training and solves its
in-context objective with conjugate gradients. The defensible distinction is
an analytically solvable structured proximal subclass, not the generic claim
of carrying an exact equilibrium.

### Gated KalmaNet and cumulative RLS

Exact cumulative online ridge regression and fading-memory Kalman/RLS updates
are established alternatives. They are an essential causal baseline and a
separate branch of the unified framework.

## Two causalizations of bidirectional LSSO

### Cumulative causal solve

Define the prefix objective

\[
Z_t^{\mathrm{cum}}=\arg\min_Z\left[
\frac\mu2\|Z\|_F^2+
\frac\gamma2\sum_{i\le t}\|u_iZ-c_i\|^2
\right].
\]

With

\[
H_t=\mu I+\gamma\sum_{i\le t}u_i^\top u_i,\qquad
B_t=\gamma\sum_{i\le t}u_i^\top c_i,
\]

we obtain `Z_t = H_t^{-1} B_t`. At the final token,

\[
Z_N^{\mathrm{cum}}=Z_{\mathrm{LSSO}}^{\mathrm{bidirectional}}.
\]

This endpoint consistency gives the strictest bridge between causal prefix
solves and the bidirectional global equilibrium. Its memory includes curvature
or sufficient statistics and its update is closely related to RLS/Mesa/GKA.

### Recursive proximal solve

The state-only proximal recurrence carries only `Z_t`. It is order-sensitive,
prediction-dependent, and generally satisfies

\[
Z_N^{\mathrm{prox}}\ne Z_{\mathrm{LSSO}}^{\mathrm{bidirectional}}.
\]

This is not an approximation to the cumulative endpoint; it defines a
different adaptive-memory semantics related to DeltaNet and Longhorn.

The framework should present these as two complementary memory types:

- set-like prefix-optimal equilibrium memory;
- order-sensitive recursive adaptive memory.

## Block-proximal opportunity

For a chunk of tokens, the exact proximal update retains the complete Gram
matrix `U_t^T U_t` and can reuse the LSSO Woodbury/MathDx machinery. This is
the most promising method-level gap:

- no diagonal curvature approximation;
- analytic low-rank solve rather than iterative CG;
- matrix-valued multi-output fast weights;
- optional RRLSSO rank rotation;
- a common primitive for bidirectional, prefix, and recursive modes.

However, one block solve is not equal to sequential token proximal updates.
The former is permutation invariant inside the block while the latter is
order-sensitive. A block computed from all tokens also leaks future
information to earlier outputs unless a causal intra-chunk dual/readout is
derived. Strict causal chunk semantics are therefore a central open problem,
not an implementation detail.

## Stability facts

For the token update, the state Jacobian is

\[
A_t=I-
\frac{\alpha_t}{\beta_t+\alpha_t\|u_t\|^2}
u_t^\top u_t.
\]

Its eigenvalues are one on the subspace orthogonal to `u_t` and

\[
\frac{\beta_t}{\beta_t+\alpha_t\|u_t\|^2}
\]

along `u_t`. Thus the fixed-feature transition is non-expansive, but it does
not automatically forget untouched subspaces. Bounded state, persistent
excitation, controlled gates, and long-term memory remain empirical and
theoretical requirements.

## Claims to avoid

Do not claim:

- the first exact causal online regression update;
- the first residual-conditioned fast-weight write;
- the first locally optimal TTT layer;
- that a token solve has greater transition expressivity than normalized
  DeltaNet;
- that a block solve is merely a parallel implementation of token recurrence;
- that exact inner optimization necessarily improves task performance;
- that the proximal endpoint equals the bidirectional LSSO solution.

Finite-step iterates converge to the exact state for fixed outer features and
the fixed proximal objective. Task accuracy need not improve monotonically
with the number of inner steps because finite optimization can regularize and
the outer model co-adapts to the chosen solver.

## Proposed paper-level contribution

The strongest prospective claim is:

> Sequence mixers can be described by a learned inner objective, a temporal
> support, a state prior, a curvature model, and a solver. This view unifies
> bidirectional LSSO, finite-step TTT, Delta-style fast weights, Longhorn, and
> Mesa optimization, and motivates a causal block-proximal solve operator that
> evaluates full low-rank equilibria analytically.

The framework becomes a method contribution only if it produces at least one
new verified capability, preferably:

1. strict causal intra-chunk outputs with analytic block solves;
2. a measurable benefit from full non-diagonal Gram geometry over
   DeltaNet/Longhorn approximations;
3. lower compute or better numerical behavior than MesaNet's iterative solve;
4. a rank-rotary causal fast-weight construction with useful relative-position
   and effective-rank properties;
5. a theorem connecting global, prefix, and recursive solve modes.

## Minimum experimental program

1. Verify exact token and block solutions against converged GD.
2. Compare token proximal, sequential DeltaNet, Longhorn exact/diagonal,
   cumulative RLS, and block proximal updates.
3. Test order swaps, repeated writes, conflicting bindings, rule changes, and
   length extrapolation.
4. Evaluate associative recall, selective copying, MQAR, induction, and state
   tracking before language modeling.
5. Report recurrent decode, chunked prefill, backward memory, condition
   numbers, state norms, and effective rank.
6. Treat read-before-write and write-before-read as explicit causal protocols.

## Primary references

- TTT: <https://arxiv.org/abs/2407.04620>
- DeltaNet parallel delta rule: <https://arxiv.org/abs/2406.06484>
- Longhorn: <https://arxiv.org/abs/2407.14207>
- MesaNet: <https://arxiv.org/abs/2506.05233>
- Gated DeltaNet: <https://arxiv.org/abs/2412.06464>
- Gated KalmaNet: <https://arxiv.org/abs/2511.21016>
- Preconditioned DeltaNet: <https://arxiv.org/abs/2604.21100>

## Boundary with the first LSSO paper

The first paper remains focused on bidirectional LSSO/RRLSSO, equilibrium and
energy interpretations, CV and cross-modal evaluation, scaling, and kernels.
The causal framework should appear only as concise future work. A separate
project can ask how global, prefix, and recursively composed equilibria form a
single sequence-learning family.
