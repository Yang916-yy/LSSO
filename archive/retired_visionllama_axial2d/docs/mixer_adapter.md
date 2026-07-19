# Unified mixer adapter and 2-D rank rotary

`MixerAdapter` gives backbone code one batch-first interface for MHA, LSSO,
and RRLSSO. A mask is always a boolean `[B,N]` tensor where `True` means a
valid token.

```python
from lsso import MixerAdapter

mixer = MixerAdapter(
    dim=384,
    num_heads=6,
    mixer="rrlsso",  # mha | lsso | rrlsso
    rank=32,
)
y = mixer(
    x,
    valid_mask=valid_mask,
    spatial_shape=(height, width),
    num_prefix_tokens=1,
)
```

For windowed or shifted layouts, pass explicit `(x,y)` coordinates instead:

```python
y = mixer(x, valid_mask=valid_mask, position_coords=coords)
```

`coords` may be `[N,2]` or `[B,N,2]`. MHA applies the 2-D rotation to Q/K;
RRLSSO applies the same coordinate construction to its relation basis; LSSO
does not rotate. Learned absolute positional embeddings remain the
responsibility of the surrounding backbone and are compatible with all three.

The 2-D transform divides the feature dimension into equal x/y subspaces, so
the MHA head dimension or RRLSSO rank must be divisible by four when it is
enabled. Prefix tokens generated with `num_prefix_tokens` receive coordinate
zero and therefore an identity rotation.
