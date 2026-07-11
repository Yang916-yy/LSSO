from __future__ import annotations

import torch

from examples.models.vit import VisionEncoder
from lsso import LSSO


def test_lsso_forward_backward() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 17, 48, requires_grad=True)
    layer = LSSO(dim=48, num_heads=3, rank=8, dropout=0.0)
    layer.record_diagnostics = True
    y = layer(x)
    assert y.shape == x.shape
    loss = y.square().mean()
    loss.backward()
    assert x.grad is not None
    assert layer.last_diagnostics is not None


def test_vit_variants() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 3, 32, 32)
    for mixer in ["mha", "lsso", "lsso-no-global"]:
        model = VisionEncoder(
            num_classes=10,
            dim=48,
            depth=2,
            num_heads=3,
            mixer=mixer,
            rank=8,
        )
        y = model(x)
        assert y.shape == (2, 10)


if __name__ == "__main__":
    test_lsso_forward_backward()
    test_vit_variants()
    print("smoke test passed")
