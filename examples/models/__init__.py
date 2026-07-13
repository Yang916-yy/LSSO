from .vit import VisionEncoder

__all__ = ["VisionEncoder"]
from .vision_llama import (
    VisionLLaMA,
    VisionLLaMABlock,
    create_vision_llama,
    load_official_vision_llama_checkpoint,
)
from .vision_llama_pyramid import PyramidVisionLLaMA, create_pyramid_vision_llama
from .vision_llama_dense import (
    DenseVisionLLaMA,
    create_dense_vision_llama,
    load_dense_vision_llama_checkpoint,
)
from .simple_feature_pyramid import SimpleFeaturePyramid
from .sequence_encoder import ProteinFitnessModel, SequenceMixerEncoder

__all__ = [
    "VisionLLaMA",
    "VisionLLaMABlock",
    "create_vision_llama",
    "load_official_vision_llama_checkpoint",
    "PyramidVisionLLaMA",
    "create_pyramid_vision_llama",
    "DenseVisionLLaMA",
    "create_dense_vision_llama",
    "load_dense_vision_llama_checkpoint",
    "SimpleFeaturePyramid",
    "SequenceMixerEncoder",
    "ProteinFitnessModel",
]

# Registration is optional for library users but automatic when timm is
# installed, so `timm.create_model(...)` works after importing this package.
try:
    from . import timm_vision_llama as _timm_vision_llama  # noqa: F401
except ImportError:
    _timm_vision_llama = None
