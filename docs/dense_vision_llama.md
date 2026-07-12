# Plain VisionLLaMA for dense prediction

`DenseVisionLLaMA` reuses the exact parameter layout of the classification
backbone while removing the CLS token from dense forwarding. Classification
checkpoints can therefore be loaded after dropping only `head.weight` and
`head.bias`; `cls_token` remains loadable but unused.

The backbone returns a stride-16 BCHW feature map. Most blocks operate on
non-overlapping windows, while `global_block_indices` select the cross-window
propagation blocks. For a 12-block Base model the default indices are
`(2, 5, 8, 11)`. Set `window_size=None` or include every index to run all
blocks globally.

Learned absolute patch embeddings are interpolated over the full input grid.
Local MHA/RRLSSO blocks receive window-local 2-D rotary coordinates and global
blocks receive full-grid coordinates. Padding tokens in partial edge windows
are masked and removed during unpartitioning.

```python
from examples.models import (
    DenseVisionLLaMA,
    SimpleFeaturePyramid,
    load_dense_vision_llama_checkpoint,
)

backbone = DenseVisionLLaMA(
    image_size=224,
    patch_size=16,
    dim=768,
    depth=12,
    num_heads=12,
    mixer="rrlsso",
    rank=32,
    window_size=16,
    global_block_indices=(2, 5, 8, 11),
)
neck = SimpleFeaturePyramid(in_channels=768, out_channels=256)
load_dense_vision_llama_checkpoint(backbone, "best.pt")
features = neck(backbone(images))
```

For an input padded to a multiple of 16, `features` contains P2 through P5 at
strides `(4, 8, 16, 32)`. The same pair is intended for a Mask R-CNN or
UPerNet integration; framework registry wrappers are deliberately kept out of
the backbone implementation.
