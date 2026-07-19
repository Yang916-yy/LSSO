from .vit import VisionEncoder
from .sequence_encoder import (
    ProteinFitnessModel,
    ReverseComplementSequenceClassifier,
    SequenceClassifier,
    SequenceMixerEncoder,
    SequencePairClassifier,
    SequenceValueEncoder,
)

__all__ = [
    "VisionEncoder",
    "SequenceMixerEncoder",
    "ProteinFitnessModel",
    "ReverseComplementSequenceClassifier",
    "SequenceClassifier",
    "SequencePairClassifier",
    "SequenceValueEncoder",
]

try:
    from .deit3_rrlsso import (
        DEIT3_RRLSSO_MODELS,
        TimmRRLSSOAttention,
        replace_timm_attention_with_rrlsso,
    )
except ModuleNotFoundError as error:  # timm is an optional experiment dependency
    if error.name != "timm":
        raise
else:
    __all__ += [
        "DEIT3_RRLSSO_MODELS",
        "TimmRRLSSOAttention",
        "replace_timm_attention_with_rrlsso",
    ]
