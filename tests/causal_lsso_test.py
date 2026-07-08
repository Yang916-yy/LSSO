from __future__ import annotations

import torch

from lsso import LSSO, lsso


def _dense_prefix_reference(
    U: torch.Tensor,
    C: torch.Tensor,
    mu: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    B, H, N, _r = U.shape
    dh = C.shape[-1]
    out = torch.empty(B, H, N, dh, dtype=C.dtype, device=C.device)
    for b in range(B):
        for h in range(H):
            mu_h = mu.view(1, H, 1, 1)[0, h, 0, 0]
            gamma_h = gamma.view(1, H, 1, 1)[0, h, 0, 0]
            for i in range(N):
                Up = U[b, h, : i + 1]
                Cp = C[b, h, : i + 1]
                A = mu_h * torch.eye(i + 1, dtype=U.dtype, device=U.device)
                A = A + gamma_h * (Up @ Up.transpose(0, 1))
                out[b, h, i] = torch.linalg.solve(A, Cp)[-1]
    return out


def test_causal_lsso_matches_dense_prefix_solve() -> None:
    torch.manual_seed(0)
    B, H, N, r, dh = 2, 2, 6, 3, 4
    U = torch.randn(B, H, N, r, dtype=torch.float64)
    C = torch.randn(B, H, N, dh, dtype=torch.float64)
    mu = torch.tensor([0.8, 1.1], dtype=torch.float64)
    gamma = torch.tensor([0.03, 0.07], dtype=torch.float64)

    actual = lsso(U, C, mu, gamma, causal=True)
    expected = _dense_prefix_reference(U, C, mu, gamma)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


def test_causal_lsso_future_tokens_do_not_change_past_outputs() -> None:
    torch.manual_seed(1)
    B, H, N, r, dh = 1, 3, 8, 4, 5
    U = torch.randn(B, H, N, r)
    C = torch.randn(B, H, N, dh)
    mu = torch.full((H,), 0.9)
    gamma = torch.full((H,), 0.05)
    cutoff = 4

    U2 = U.clone()
    C2 = C.clone()
    U2[:, :, cutoff + 1 :] = torch.randn_like(U2[:, :, cutoff + 1 :]) * 10.0
    C2[:, :, cutoff + 1 :] = torch.randn_like(C2[:, :, cutoff + 1 :]) * 10.0

    y1 = lsso(U, C, mu, gamma, causal=True)
    y2 = lsso(U2, C2, mu, gamma, causal=True)
    torch.testing.assert_close(y1[:, :, : cutoff + 1], y2[:, :, : cutoff + 1], atol=1e-6, rtol=1e-6)


def test_chunked_causal_lsso_matches_materialized_prefix() -> None:
    torch.manual_seed(3)
    B, H, N, r, dh = 2, 2, 13, 4, 5
    U = torch.randn(B, H, N, r, dtype=torch.float64, requires_grad=True)
    C = torch.randn(B, H, N, dh, dtype=torch.float64, requires_grad=True)
    mu = torch.tensor([0.8, 1.1], dtype=torch.float64, requires_grad=True)
    gamma = torch.tensor([0.03, 0.07], dtype=torch.float64, requires_grad=True)

    y_full = lsso(U, C, mu, gamma, causal=True)
    grads_full = torch.autograd.grad(y_full.square().mean(), (U, C, mu, gamma), retain_graph=False)

    y_chunked = lsso(U, C, mu, gamma, causal=True, causal_chunk_size=5)
    grads_chunked = torch.autograd.grad(y_chunked.square().mean(), (U, C, mu, gamma), retain_graph=False)

    torch.testing.assert_close(y_chunked, y_full, atol=1e-10, rtol=1e-10)
    for actual, expected in zip(grads_chunked, grads_full):
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


def test_causal_lsso_backward_and_diagnostics() -> None:
    torch.manual_seed(2)
    x = torch.randn(2, 11, 48, requires_grad=True)
    layer = LSSO(dim=48, num_heads=3, rank=8, causal=True)
    layer.record_diagnostics = True
    y = layer(x)
    assert y.shape == x.shape
    loss = y.square().mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert layer.last_diagnostics is not None
    assert torch.isfinite(layer.last_diagnostics.effective_rank).all()


if __name__ == "__main__":
    test_causal_lsso_matches_dense_prefix_solve()
    test_causal_lsso_future_tokens_do_not_change_past_outputs()
    test_chunked_causal_lsso_matches_materialized_prefix()
    test_causal_lsso_backward_and_diagnostics()
    print("causal lsso test passed")
