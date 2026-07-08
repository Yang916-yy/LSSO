from __future__ import annotations

import torch

from lsso import lsso
from lsso.causal_triton import causal_prefix_lsso_triton, triton_available
from lsso.modules import _lsso_prefix_chunked_forward


def _can_run_triton() -> bool:
    return triton_available() and torch.cuda.is_available()


def test_triton_causal_forward_matches_torch_prefix() -> None:
    if not _can_run_triton():
        print("skip: Triton/CUDA unavailable")
        return

    torch.manual_seed(0)
    for rank in (4, 16, 32):
        B, H, N, dh = 2, 2, 17, 32
        U = torch.randn(B, H, N, rank, device="cuda", dtype=torch.float32)
        C = torch.randn(B, H, N, dh, device="cuda", dtype=torch.float32)
        mu = torch.tensor([0.8, 1.1], device="cuda").view(1, H, 1, 1)
        gamma = torch.tensor([0.03, 0.07], device="cuda").view(1, H, 1, 1)

        for exclusive in (False, True):
            expected = _lsso_prefix_chunked_forward(
                U,
                C,
                mu,
                gamma,
                exclusive=exclusive,
                chunk_size=8,
            )
            actual = causal_prefix_lsso_triton(
                U,
                C,
                mu,
                gamma,
                exclusive=exclusive,
                chunk_size=8,
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_triton_backend_backward_fallback_runs() -> None:
    if not _can_run_triton():
        print("skip: Triton/CUDA unavailable")
        return

    torch.manual_seed(1)
    B, H, N, rank, dh = 2, 2, 11, 8, 16
    U = torch.randn(B, H, N, rank, device="cuda", requires_grad=True)
    C = torch.randn(B, H, N, dh, device="cuda", requires_grad=True)
    mu = torch.tensor([0.9, 1.1], device="cuda", requires_grad=True).view(1, H, 1, 1)
    gamma = torch.tensor([0.02, 0.05], device="cuda", requires_grad=True).view(1, H, 1, 1)

    y = lsso(
        U,
        C,
        mu,
        gamma,
        causal=True,
        causal_backend="triton",
        causal_chunk_size=8,
    )
    loss = y.square().mean()
    loss.backward()
    assert U.grad is not None
    assert C.grad is not None
    assert torch.isfinite(U.grad).all()
    assert torch.isfinite(C.grad).all()


def test_triton_backward_fallback_matches_torch_for_linear_loss() -> None:
    if not _can_run_triton():
        print("skip: Triton/CUDA unavailable")
        return

    torch.manual_seed(2)
    B, H, N, rank, dh = 1, 2, 9, 4, 8
    U0 = torch.randn(B, H, N, rank, device="cuda")
    C0 = torch.randn(B, H, N, dh, device="cuda")
    mu0 = torch.tensor([0.9, 1.1], device="cuda").view(1, H, 1, 1)
    gamma0 = torch.tensor([0.02, 0.05], device="cuda").view(1, H, 1, 1)

    U_t = U0.clone().requires_grad_(True)
    C_t = C0.clone().requires_grad_(True)
    mu_t = mu0.clone().requires_grad_(True)
    gamma_t = gamma0.clone().requires_grad_(True)
    y_t = lsso(U_t, C_t, mu_t, gamma_t, causal=True, causal_backend="triton", causal_chunk_size=5)
    grads_t = torch.autograd.grad(y_t.sum(), (U_t, C_t, mu_t, gamma_t))

    U_ref = U0.clone().requires_grad_(True)
    C_ref = C0.clone().requires_grad_(True)
    mu_ref = mu0.clone().requires_grad_(True)
    gamma_ref = gamma0.clone().requires_grad_(True)
    y_ref = lsso(U_ref, C_ref, mu_ref, gamma_ref, causal=True, causal_chunk_size=5)
    grads_ref = torch.autograd.grad(y_ref.sum(), (U_ref, C_ref, mu_ref, gamma_ref))

    for actual, expected in zip(grads_t, grads_ref):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


if __name__ == "__main__":
    test_triton_causal_forward_matches_torch_prefix()
    test_triton_backend_backward_fallback_runs()
    test_triton_backward_fallback_matches_torch_for_linear_loss()
    print("causal triton test passed")
