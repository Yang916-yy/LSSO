from __future__ import annotations

import torch

from examples.models.text import TextEncoder
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


def test_text_variants() -> None:
    torch.manual_seed(0)
    x = torch.tensor([[2, 5, 6, 0, 0], [2, 7, 8, 9, 0]])
    for mixer in ["mha", "lsso", "lsso-no-global"]:
        model = TextEncoder(
            vocab_size=16,
            num_classes=4,
            max_len=5,
            dim=48,
            depth=2,
            num_heads=3,
            mixer=mixer,
            rank=8,
        )
        y = model(x)
        assert y.shape == (2, 4)


if __name__ == "__main__":
    test_lsso_forward_backward()
    test_vit_variants()
    test_text_variants()
    print("smoke test passed")
