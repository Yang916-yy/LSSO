from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from .bidir_lsso_flash_layer import BidirLSSOFlashPrototype, BidirRRLSSOFlashPrototype
    from .flash_bidir_lsso import flash_bidir_lsso
except ImportError:  # Allow direct execution from the prototype directory.
    from bidir_lsso_flash_layer import BidirLSSOFlashPrototype, BidirRRLSSOFlashPrototype
    from flash_bidir_lsso import flash_bidir_lsso
from lsso import LSSO, RRLSSO
from lsso.modules import lsso


def check_core(B: int, H: int, N: int, R: int, DH: int) -> None:
    torch.manual_seed(21000 + B + H + N + R + DH)
    U = torch.randn(B, H, N, R, device="cuda", dtype=torch.float32, requires_grad=True)
    C = torch.randn(B, H, N, DH, device="cuda", dtype=torch.float32, requires_grad=True)
    mu = torch.rand(H, device="cuda", dtype=torch.float32, requires_grad=True) + 0.5
    gamma = torch.rand(H, device="cuda", dtype=torch.float32, requires_grad=True) * 0.1
    grad = torch.randn(B, H, N, DH, device="cuda", dtype=torch.float32)

    y = flash_bidir_lsso(U, C, mu, gamma)
    y_ref = lsso(
        U,
        C,
        mu.view(1, H, 1, 1),
        gamma.view(1, H, 1, 1),
        causal=False,
        length_normalize=False,
    )
    err = (y - y_ref).abs().max().item()
    print(f"core B={B} H={H} N={N} R={R} DH={DH} y_err={err:.3e}")
    torch.testing.assert_close(y, y_ref, atol=5e-5, rtol=5e-5)

    grads = torch.autograd.grad(y, (U, C, mu, gamma), grad, allow_unused=True)
    grads_ref = torch.autograd.grad(y_ref, (U, C, mu, gamma), grad, allow_unused=True)
    for name, actual, expected in zip(("U", "C", "mu", "gamma"), grads, grads_ref):
        assert actual is not None and expected is not None
        gerr = (actual - expected).abs().max().item()
        print(f"{name}_grad_err={gerr:.3e}")
        torch.testing.assert_close(actual, expected, atol=5e-4, rtol=5e-4)


def check_layer(layer_cls, ref_cls, *, position_ids: torch.Tensor | None = None, bias: bool = False) -> None:
    torch.manual_seed(22000 + int(bias) + (0 if position_ids is None else 17))
    B, N, D, H, R = 2, 129, 256, 4, 32
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float32, requires_grad=True)
    grad = torch.randn(B, N, D, device="cuda", dtype=torch.float32)
    mask = torch.ones(B, N, device="cuda", dtype=torch.bool)
    mask[1, -7:] = False

    ref = ref_cls(dim=D, num_heads=H, rank=R, causal=False, bias=bias).cuda()
    flash = layer_cls(dim=D, num_heads=H, rank=R, bias=bias).cuda()
    flash.load_state_dict(ref.state_dict(), strict=False)

    kwargs = {"valid_mask": mask}
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    y = flash(x, **kwargs)
    y_ref = ref(x, **kwargs)
    err = (y - y_ref).abs().max().item()
    print(f"{layer_cls.__name__} bias={bias} y_err={err:.3e}")
    torch.testing.assert_close(y, y_ref, atol=5e-5, rtol=5e-5)

    params = tuple(flash.parameters())
    params_ref = tuple(ref.parameters())
    grads = torch.autograd.grad(y, (x, *params), grad, allow_unused=True)
    grads_ref = torch.autograd.grad(y_ref, (x, *params_ref), grad, allow_unused=True)
    names = ["x", *[name for name, _ in flash.named_parameters()]]
    for name, actual, expected in zip(names, grads, grads_ref):
        if actual is None or expected is None:
            if actual is not None or expected is not None:
                raise AssertionError(f"{name} unused mismatch")
            continue
        gerr = (actual - expected).abs().max().item()
        print(f"{name}_grad_err={gerr:.3e}")
        torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")
    check_core(B=1, H=2, N=33, R=16, DH=32)
    check_core(B=2, H=4, N=130, R=32, DH=64)
    check_layer(BidirLSSOFlashPrototype, LSSO, bias=True)
    pos = torch.arange(129, device="cuda") + 3
    check_layer(BidirRRLSSOFlashPrototype, RRLSSO, position_ids=pos, bias=True)
    print("flash bidir LSSO/RRLSSO numerical tests passed")


if __name__ == "__main__":
    main()
