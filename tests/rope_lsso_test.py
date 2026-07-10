from __future__ import annotations

import torch

from lsso import (
    LSSO,
    RRLSSO,
    apply_rank_rotary,
    lsso,
    read_solve_state,
    update_solve_state,
)


def test_rank_rope_preserves_norm_and_relative_kernel() -> None:
    torch.manual_seed(0)
    B, H, N, r = 2, 3, 9, 8
    U = torch.randn(B, H, N, r, dtype=torch.float64)
    positions = torch.arange(N, dtype=torch.float64)
    U_rope = apply_rank_rotary(U, positions)

    torch.testing.assert_close(U_rope.norm(dim=-1), U.norm(dim=-1), atol=1e-12, rtol=1e-12)

    shifted = apply_rank_rotary(U, positions + 17)
    kernel = torch.matmul(U_rope, U_rope.transpose(-2, -1))
    shifted_kernel = torch.matmul(shifted, shifted.transpose(-2, -1))
    torch.testing.assert_close(shifted_kernel, kernel, atol=1e-10, rtol=1e-10)


def test_rrlsso_zero_positions_matches_v1() -> None:
    torch.manual_seed(1)
    B, N, D, H, r = 2, 11, 48, 3, 8
    x = torch.randn(B, N, D)
    v1 = LSSO(dim=D, num_heads=H, rank=r, causal=False)
    v2 = RRLSSO(dim=D, num_heads=H, rank=r, causal=False)
    v2.load_state_dict(v1.state_dict(), strict=True)
    zeros = torch.zeros(N)

    y1 = v1(x)
    y2 = v2(x, position_ids=zeros)
    torch.testing.assert_close(y2, y1, atol=1e-6, rtol=1e-6)


def test_rrlsso_bidirectional_backward_and_diagnostics() -> None:
    torch.manual_seed(2)
    x = torch.randn(2, 17, 64, requires_grad=True)
    layer = RRLSSO(dim=64, num_heads=4, rank=16)
    layer.record_diagnostics = True
    y = layer(x)
    assert y.shape == x.shape
    y.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert layer.last_diagnostics is not None
    assert torch.isfinite(layer.last_diagnostics.effective_rank).all()


def test_solve_state_read_matches_bidirectional_rrlsso() -> None:
    torch.manual_seed(7)
    B, H, N, r, dh = 2, 2, 10, 8, 6
    U = torch.randn(B, H, N, r, dtype=torch.float64)
    C = torch.randn(B, H, N, dh, dtype=torch.float64)
    mu = torch.tensor([0.9, 1.1], dtype=torch.float64)
    gamma = torch.tensor([0.02, 0.05], dtype=torch.float64)

    U_rope = apply_rank_rotary(U)
    cache = update_solve_state(None, U_rope, C)
    y_cache = read_solve_state(U_rope, C, cache, mu, gamma)
    y_ref = lsso(U_rope, C, mu, gamma)
    torch.testing.assert_close(y_cache, y_ref, atol=1e-10, rtol=1e-10)


if __name__ == "__main__":
    test_rank_rope_preserves_norm_and_relative_kernel()
    test_rrlsso_zero_positions_matches_v1()
    test_rrlsso_bidirectional_backward_and_diagnostics()
    test_solve_state_read_matches_bidirectional_rrlsso()
    print("rrlsso test passed")
