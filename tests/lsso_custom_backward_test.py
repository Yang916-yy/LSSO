from __future__ import annotations

import torch
from torch.autograd import gradcheck

from lsso.modules import _lsso_woodbury_forward, length_normalize_basis, lsso


def test_lsso_custom_backward_matches_autograd() -> None:
    torch.manual_seed(0)
    B, H, N, r, dh = 2, 3, 7, 4, 5
    U0 = torch.randn(B, H, N, r, dtype=torch.float64)
    C0 = torch.randn(B, H, N, dh, dtype=torch.float64)
    mu0 = torch.full((1, H, 1, 1), 0.9, dtype=torch.float64)
    gamma0 = torch.full((1, H, 1, 1), 0.03, dtype=torch.float64)
    probe = torch.randn(B, H, N, dh, dtype=torch.float64)

    def run_custom():
        U = U0.clone().requires_grad_(True)
        C = C0.clone().requires_grad_(True)
        mu = mu0.clone().requires_grad_(True)
        gamma = gamma0.clone().requires_grad_(True)
        Y = lsso(U, C, mu, gamma)
        loss = (Y * probe).sum()
        loss.backward()
        return Y.detach(), (U.grad, C.grad, mu.grad, gamma.grad)

    def run_reference():
        U = U0.clone().requires_grad_(True)
        C = C0.clone().requires_grad_(True)
        mu = mu0.clone().requires_grad_(True)
        gamma = gamma0.clone().requires_grad_(True)
        Y = _lsso_woodbury_forward(length_normalize_basis(U), C, mu, gamma)
        loss = (Y * probe).sum()
        loss.backward()
        return Y.detach(), (U.grad, C.grad, mu.grad, gamma.grad)

    Y_ref, grads_ref = run_reference()
    Y_custom, grads_custom = run_custom()
    torch.testing.assert_close(Y_custom, Y_ref, atol=1e-10, rtol=1e-10)
    for actual, expected in zip(grads_custom, grads_ref):
        torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-6)


def test_lsso_custom_backward_gradcheck() -> None:
    torch.manual_seed(1)
    B, H, N, r, dh = 1, 2, 5, 3, 4
    U = torch.randn(B, H, N, r, dtype=torch.float64, requires_grad=True)
    C = torch.randn(B, H, N, dh, dtype=torch.float64, requires_grad=True)
    mu = torch.full((1, H, 1, 1), 0.8, dtype=torch.float64, requires_grad=True)
    gamma = torch.full((1, H, 1, 1), 0.02, dtype=torch.float64, requires_grad=True)

    assert gradcheck(
        lambda U, C, mu, gamma: lsso(U, C, mu, gamma),
        (U, C, mu, gamma),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
        fast_mode=True,
    )


def test_lsso_custom_backward_cuda_matches_autograd() -> None:
    if not torch.cuda.is_available():
        return

    torch.manual_seed(2)
    B, H, N, r, dh = 2, 4, 257, 16, 16
    dtype = torch.float16
    device = torch.device("cuda")
    U0 = torch.randn(B, H, N, r, device=device, dtype=dtype)
    C0 = torch.randn(B, H, N, dh, device=device, dtype=dtype)
    mu0 = torch.full((1, H, 1, 1), 0.9, device=device, dtype=dtype)
    gamma0 = torch.full((1, H, 1, 1), 0.03, device=device, dtype=dtype)
    probe = torch.randn(B, H, N, dh, device=device, dtype=dtype)

    def run_custom():
        U = U0.clone().requires_grad_(True)
        C = C0.clone().requires_grad_(True)
        mu = mu0.clone().requires_grad_(True)
        gamma = gamma0.clone().requires_grad_(True)
        Y = lsso(U, C, mu, gamma)
        (Y * probe).sum().backward()
        return Y.detach(), (U.grad, C.grad, mu.grad, gamma.grad)

    def run_reference():
        U = U0.clone().requires_grad_(True)
        C = C0.clone().requires_grad_(True)
        mu = mu0.clone().requires_grad_(True)
        gamma = gamma0.clone().requires_grad_(True)
        Y = _lsso_woodbury_forward(length_normalize_basis(U), C, mu, gamma)
        (Y * probe).sum().backward()
        return Y.detach(), (U.grad, C.grad, mu.grad, gamma.grad)

    Y_ref, grads_ref = run_reference()
    Y_custom, grads_custom = run_custom()
    torch.testing.assert_close(Y_custom.float(), Y_ref.float(), atol=5e-3, rtol=5e-3)
    for actual, expected in zip(grads_custom, grads_ref):
        torch.testing.assert_close(actual.float(), expected.float(), atol=8e-2, rtol=8e-2)


if __name__ == "__main__":
    test_lsso_custom_backward_matches_autograd()
    test_lsso_custom_backward_gradcheck()
    test_lsso_custom_backward_cuda_matches_autograd()
    print("lsso custom backward test passed")
