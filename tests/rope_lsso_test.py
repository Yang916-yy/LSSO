from __future__ import annotations

import torch

from lsso import (
    LSSO,
    RoPELSSO,
    apply_rank_rope,
    lsso,
    read_solve_state,
    update_solve_state,
)
from lsso.causal_triton import triton_available


def test_rank_rope_preserves_norm_and_relative_kernel() -> None:
    torch.manual_seed(0)
    B, H, N, r = 2, 3, 9, 8
    U = torch.randn(B, H, N, r, dtype=torch.float64)
    positions = torch.arange(N, dtype=torch.float64)
    U_rope = apply_rank_rope(U, positions)

    torch.testing.assert_close(U_rope.norm(dim=-1), U.norm(dim=-1), atol=1e-12, rtol=1e-12)

    shifted = apply_rank_rope(U, positions + 17)
    kernel = torch.matmul(U_rope, U_rope.transpose(-2, -1))
    shifted_kernel = torch.matmul(shifted, shifted.transpose(-2, -1))
    torch.testing.assert_close(shifted_kernel, kernel, atol=1e-10, rtol=1e-10)


def test_rope_lsso_zero_positions_matches_v1() -> None:
    torch.manual_seed(1)
    B, N, D, H, r = 2, 11, 48, 3, 8
    x = torch.randn(B, N, D)
    v1 = LSSO(dim=D, num_heads=H, rank=r, causal=False)
    v2 = RoPELSSO(dim=D, num_heads=H, rank=r, causal=False)
    v2.load_state_dict(v1.state_dict(), strict=True)
    zeros = torch.zeros(N)

    y1 = v1(x)
    y2 = v2(x, position_ids=zeros)
    torch.testing.assert_close(y2, y1, atol=1e-6, rtol=1e-6)


def test_rope_lsso_bidirectional_backward_and_diagnostics() -> None:
    torch.manual_seed(2)
    x = torch.randn(2, 17, 64, requires_grad=True)
    layer = RoPELSSO(dim=64, num_heads=4, rank=16)
    layer.record_diagnostics = True
    y = layer(x)
    assert y.shape == x.shape
    y.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert layer.last_diagnostics is not None
    assert torch.isfinite(layer.last_diagnostics.effective_rank).all()


def test_rope_causal_lsso_future_tokens_do_not_change_past_outputs() -> None:
    torch.manual_seed(3)
    B, N, D = 2, 13, 48
    cutoff = 6
    x = torch.randn(B, N, D)
    x2 = x.clone()
    x2[:, cutoff + 1 :] = torch.randn_like(x2[:, cutoff + 1 :]) * 10.0

    layer = RoPELSSO(dim=D, num_heads=3, rank=8, causal=True)
    y1 = layer(x)
    y2 = layer(x2)
    torch.testing.assert_close(y1[:, : cutoff + 1], y2[:, : cutoff + 1], atol=1e-5, rtol=1e-5)


def test_rope_causal_chunked_matches_materialized() -> None:
    torch.manual_seed(4)
    B, N, D, H, r = 2, 19, 48, 3, 8
    x = torch.randn(B, N, D, requires_grad=True)
    full = RoPELSSO(dim=D, num_heads=H, rank=r, causal=True)
    chunked = RoPELSSO(dim=D, num_heads=H, rank=r, causal=True, causal_chunk_size=5)
    chunked.load_state_dict(full.state_dict(), strict=True)

    y_full = full(x)
    y_chunked = chunked(x)
    torch.testing.assert_close(y_chunked, y_full, atol=1e-6, rtol=1e-6)

    grad_full = torch.autograd.grad(y_full.square().mean(), x, retain_graph=True)[0]
    grad_chunked = torch.autograd.grad(y_chunked.square().mean(), x, retain_graph=True)[0]
    torch.testing.assert_close(grad_chunked, grad_full, atol=1e-6, rtol=1e-6)


def test_rope_causal_triton_matches_torch_chunked() -> None:
    if not (triton_available() and torch.cuda.is_available()):
        print("skip: Triton/CUDA unavailable")
        return

    torch.manual_seed(5)
    B, N, D, H, r = 2, 17, 64, 4, 16
    x = torch.randn(B, N, D, device="cuda")
    torch_layer = RoPELSSO(dim=D, num_heads=H, rank=r, causal=True, causal_chunk_size=8).cuda()
    triton_layer = RoPELSSO(
        dim=D,
        num_heads=H,
        rank=r,
        causal=True,
        causal_chunk_size=8,
        causal_backend="triton",
    ).cuda()
    triton_layer.load_state_dict(torch_layer.state_dict(), strict=True)

    expected = torch_layer(x)
    actual = triton_layer(x)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_solve_state_read_matches_bidirectional_rope_lsso() -> None:
    torch.manual_seed(7)
    B, H, N, r, dh = 2, 2, 10, 8, 6
    U = torch.randn(B, H, N, r, dtype=torch.float64)
    C = torch.randn(B, H, N, dh, dtype=torch.float64)
    mu = torch.tensor([0.9, 1.1], dtype=torch.float64)
    gamma = torch.tensor([0.02, 0.05], dtype=torch.float64)

    U_rope = apply_rank_rope(U)
    cache = update_solve_state(None, U_rope, C)
    y_cache = read_solve_state(U_rope, C, cache, mu, gamma)
    y_ref = lsso(U_rope, C, mu, gamma)
    torch.testing.assert_close(y_cache, y_ref, atol=1e-10, rtol=1e-10)


def test_solve_state_read_matches_causal_rope_prefix_at_token() -> None:
    torch.manual_seed(8)
    B, H, N, r, dh = 2, 2, 13, 8, 6
    U = torch.randn(B, H, N, r, dtype=torch.float64)
    C = torch.randn(B, H, N, dh, dtype=torch.float64)
    mu = torch.tensor([0.85, 1.15], dtype=torch.float64)
    gamma = torch.tensor([0.03, 0.06], dtype=torch.float64)
    t = 10

    U_rope = apply_rank_rope(U)
    y_ref = lsso(U_rope, C, mu, gamma, causal=True)[:, :, t : t + 1]

    cache = None
    for i in range(t + 1):
        cache = update_solve_state(cache, U_rope[:, :, i : i + 1], C[:, :, i : i + 1])
    y_cache = read_solve_state(
        U_rope[:, :, t : t + 1],
        C[:, :, t : t + 1],
        cache,
        mu,
        gamma,
    )
    torch.testing.assert_close(y_cache, y_ref, atol=1e-10, rtol=1e-10)


if __name__ == "__main__":
    test_rank_rope_preserves_norm_and_relative_kernel()
    test_rope_lsso_zero_positions_matches_v1()
    test_rope_lsso_bidirectional_backward_and_diagnostics()
    test_rope_causal_lsso_future_tokens_do_not_change_past_outputs()
    test_rope_causal_chunked_matches_materialized()
    test_rope_causal_triton_matches_torch_chunked()
    test_solve_state_read_matches_bidirectional_rope_lsso()
    test_solve_state_read_matches_causal_rope_prefix_at_token()
    print("rope lsso test passed")
