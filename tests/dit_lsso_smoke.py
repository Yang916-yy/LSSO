from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.models.dit import LatentDiT


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LatentDiT(
        latent_size=16,
        patch_size=2,
        in_channels=4,
        hidden_size=96,
        depth=2,
        num_heads=6,
        rank=16,
        num_classes=10,
        mixer="lsso",
    ).to(device)
    x = torch.randn(2, 4, 16, 16, device=device, requires_grad=True)
    t = torch.randint(0, 1000, (2,), device=device)
    y = torch.randint(0, 10, (2,), device=device)
    out = model(x, t, y)
    assert out.shape == x.shape
    loss = out.float().square().mean()
    loss.backward()
    assert torch.isfinite(out).all()
    assert torch.isfinite(x.grad).all()
    print("ok", tuple(out.shape), len(model.lsso_layers()))


if __name__ == "__main__":
    main()
