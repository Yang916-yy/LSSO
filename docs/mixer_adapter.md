# Unified mixer adapter

`MixerAdapter` exposes a common batch-first interface for MHA, LSSO, and
RRLSSO. Active RRLSSO uses ordinary one-dimensional Rank Rotary only.

```python
from lsso import MixerAdapter

mixer = MixerAdapter(dim=384, num_heads=6, mixer="rrlsso", rank=32)
y = mixer(x, valid_mask=valid_mask)
```

MHA may optionally use ordinary 1-D RoPE through `rotary_1d=True`. LSSO has no
rotary transform. The retired axial 2-D implementation is not available from
the public API.
