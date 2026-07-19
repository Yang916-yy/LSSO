from __future__ import annotations

import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.models import create_pyramid_vision_llama, create_vision_llama


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal VisionLLaMA smoke test")
    image = torch.randn(1, 3, 224, 224, device="cuda", dtype=torch.bfloat16)
    for mixer in ("mha", "rrlsso"):
        model = create_vision_llama(
            "small", mixer=mixer, num_classes=10
        ).cuda().bfloat16().eval()
        with torch.inference_mode():
            output = model(image)
        print("plain", mixer, tuple(output.shape), bool(torch.isfinite(output).all()))
        del model, output
        torch.cuda.empty_cache()

    model = create_pyramid_vision_llama(
        "small", mixer="rrlsso", num_classes=0
    ).cuda().bfloat16().eval()
    with torch.inference_mode():
        outputs = model.forward_features(image)
    print(
        "pyramid",
        "rrlsso",
        [tuple(item.shape) for item in outputs],
        all(bool(torch.isfinite(item).all()) for item in outputs),
    )


if __name__ == "__main__":
    main()
