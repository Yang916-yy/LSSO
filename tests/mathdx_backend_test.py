from __future__ import annotations

import pytest
import torch

from lsso.mathdx_backend import (
    load_mathdx_backend,
    mathdx_load_error,
    solve_spd,
    stats_solve_spd,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("rank", [16, 32])
@pytest.mark.parametrize("rhs_width", [1, 32, 64, 192])
def test_mathdx_solve_spd_matches_torch(rank: int, rhs_width: int) -> None:
    if not load_mathdx_backend():
        pytest.skip(str(mathdx_load_error()))

    torch.manual_seed(0)
    a = torch.randn(7, rank, rank, device="cuda", dtype=torch.float32)
    gram = a @ a.transpose(-1, -2) + 0.5 * torch.eye(rank, device="cuda")
    rhs = torch.randn(7, rank, rhs_width, device="cuda", dtype=torch.float32)

    actual, info = solve_spd(gram.contiguous(), rhs.contiguous())
    expected = torch.linalg.solve(gram, rhs)

    assert torch.count_nonzero(info).item() == 0
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("rank", "sequence", "rhs_width"),
    [(16, 65, 64), (16, 196, 192), (32, 65, 64), (32, 196, 192)],
)
def test_mathdx_fused_stats_solve_matches_torch(
    rank: int,
    sequence: int,
    rhs_width: int,
) -> None:
    if not load_mathdx_backend():
        pytest.skip(str(mathdx_load_error()))

    torch.manual_seed(1)
    u = 0.2 * torch.randn(5, sequence, rank, device="cuda")
    c = torch.randn(5, sequence, rhs_width, device="cuda")
    alpha = 0.05 * torch.rand(5, device="cuda")

    actual, info = stats_solve_spd(u, c, alpha)
    gram = u.transpose(1, 2) @ u
    rhs = u.transpose(1, 2) @ c
    system = torch.eye(rank, device="cuda") + alpha[:, None, None] * gram
    expected = torch.linalg.solve(system, rhs)

    assert torch.count_nonzero(info).item() == 0
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=1e-5)
