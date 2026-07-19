# VisionLLaMA integration

The project contains an independent, optimized implementation of the plain
and four-stage Pyramid VisionLLaMA backbones. It reuses the published
architecture and recipe concepts without importing the upstream copied DeiT,
MMDetection, or MMSegmentation source trees.

## Plain backbone

```python
from examples.models import create_vision_llama

model = create_vision_llama(
    "small",                 # small | base | large
    mixer="rrlsso",          # mha | lsso | rrlsso
    rank=32,
    learned_position=True,
)
```

All variants share patch embedding, RMSNorm, SwiGLU, LayerScale, stochastic
depth, learned absolute position embeddings, classifier, and 2-D coordinate
construction. MHA uses 2-D RoPE on Q/K; RRLSSO uses 2-D rank rotary; LSSO has
no rotary transform.

`load_official_vision_llama_checkpoint` converts the upstream plain-model key
layout. MHA imports attention weights. LSSO/RRLSSO import only the compatible
patch, normalization, SwiGLU, LayerScale, and classifier weights; their mixer
weights must be trained independently for controlled experiments.

## Pyramid backbone

```python
from examples.models import create_pyramid_vision_llama

backbone = create_pyramid_vision_llama(
    "small",
    mixer="rrlsso",
    attention_policy="windowed",
    num_classes=0,
)
features = backbone.forward_features(images)
```

The returned tuple contains strides 4, 8, 16, and 32 and can be connected to
FPN/UPerNet. `windowed` is the controlled protocol: every mixer sees identical
7x7 windows. `alternating-global` alternates 7x7 windows with full global
mixing and is a separate global-context experiment. The latter intentionally
does not claim to reproduce the upstream GSA attention, which spatially
reduces K/V and has no exact LSSO analogue.

Run the real-size CUDA/BF16 check with:

```bash
python tools/smoke_vision_llama.py
```
