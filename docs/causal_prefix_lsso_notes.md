# Causal Prefix-LSSO Notes

This note records a possible causal extension of LSSO. The point is not to
force an `N x N` triangular mask into the non-causal operator. Instead, the
causal problem can be written as a prefix low-rank solve.

## Prefix Form

Non-causal LSSO solves

```text
(mu I + gamma U U^T) Y = C
```

For a causal position `i`, define the output by the prefix system:

```text
y_i = [(mu I + gamma U_{<=i} U_{<=i}^T)^(-1) C_{<=i}]_i
```

Using Woodbury, let

```text
alpha = gamma / mu
S_i = sum_{j<=i} u_j^T u_j      # [r, r]
P_i = sum_{j<=i} u_j^T c_j      # [r, d_h]
```

Then

```text
y_i = (c_i - alpha * u_i (I + alpha S_i)^(-1) P_i) / mu
```

This gives strict causal dependence because `y_i` only uses prefix statistics
from tokens `1..i`. Inclusive prefix corresponds to ordinary causal attention,
where a token can attend to itself. Exclusive prefix can be obtained by shifting
the prefix sums by one position.

## Parallel Training Path

The training/prefill computation can remain parallel:

```python
# U: [B, H, N, r]
# C: [B, H, N, d_h]
# alpha = gamma / mu

S_token = torch.einsum("bhnr,bhns->bhnrs", U, U)
P_token = torch.einsum("bhnr,bhnd->bhnrd", U, C)

S = torch.cumsum(S_token, dim=2)
P = torch.cumsum(P_token, dim=2)

R = I + alpha * S
Z = torch.linalg.solve(R, P)

corr = torch.einsum("bhnr,bhnrd->bhnd", U, Z)
Y = (C - alpha * corr) / mu
```

The only sequence dependency is the prefix sum/scan. The solves are batched
small `r x r` systems across positions.

## Decoding Cache

Autoregressive decoding keeps a fixed-size low-rank state per layer and head:

```text
S_t = S_{t-1} + u_t^T u_t
P_t = P_{t-1} + u_t^T c_t
```

Then compute

```text
y_t = (c_t - alpha * u_t (I + alpha S_t)^(-1) P_t) / mu
```

Cache size per layer/head:

```text
O(r^2 + r d_h)
```

This does not grow linearly with sequence length, unlike a standard KV cache.

## Flash-Style Kernel Analogy

The useful analogy is:

```text
FlashAttention:
Q/K/V block -> SRAM local attention -> online softmax -> write O

Flash Prefix-LSSO:
U/C block -> SRAM prefix S/P update -> small matrix solve -> write Y
```

The cost is not free. Compared with non-causal LSSO, causal prefix-LSSO performs
a small solve per position rather than one solve per sequence. However, the
computation is structured:

```text
1. load U/C block
2. compute block-local outer products u^T u and u^T c
3. scan prefix statistics S/P across blocks
4. solve many small r x r systems
5. write Y
```

This suggests a kernel path similar in spirit to FlashAttention: avoid material
`N x N` masks, keep the recurrent statistics close to SRAM, and write only the
final token states.

## Paper Framing

Suggested language:

```text
Although causal decoding admits a recurrent cache form, causal training need
not be formulated as an explicit triangular masked solve. A causal Prefix-LSSO
can rewrite the masked dependency into prefix low-rank statistics S_i and P_i,
computed by parallel scan, followed by batched small-dimensional solves. This
offers a possible path toward Flash-style causal LSSO kernels with parallel
prefill and fixed-size decoding caches.
```

This should be framed as a future direction, not a completed implementation.
