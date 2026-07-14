from __future__ import annotations

import pytest
import torch

from lsso.mathdx_backend import (
    load_mathdx_backend,
    mathdx_load_error,
    solve_spd,
    stats_solve_readout,
    stats_solve_spd,
    try_masked_stats_solve_spd,
    try_prepare_basis,
    try_rank_rotary,
)
from lsso.modules_v2 import apply_rank_rotary


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
@pytest.mark.parametrize("rank", [16, 32])
@pytest.mark.parametrize("batches_per_block", [2, 4])
def test_mathdx_multi_system_cholesky_matches_torch(
    rank: int, batches_per_block: int
) -> None:
    if not load_mathdx_backend():
        pytest.skip(str(mathdx_load_error()))

    torch.manual_seed(13)
    systems, rhs_width = 128, 64
    a = torch.randn(systems, rank, rank, device="cuda")
    gram = a @ a.transpose(-1, -2) + torch.eye(rank, device="cuda")
    rhs = torch.randn(systems, rank, rhs_width, device="cuda")
    actual, info = torch.ops.lsso_mathdx.solve_spd_bpb(
        gram.contiguous(), rhs.contiguous(), batches_per_block
    )
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rank", [16, 32])
@pytest.mark.parametrize("rhs_width", [48, 64])
def test_mathdx_stats_solve_readout_matches_torch(
    dtype: torch.dtype, rank: int, rhs_width: int
) -> None:
    if not load_mathdx_backend():
        pytest.skip(str(mathdx_load_error()))

    torch.manual_seed(11)
    batches, sequence = 7, 197
    u = (0.2 * torch.randn(batches, sequence, rank, device="cuda")).to(dtype)
    c = torch.randn(batches, sequence, rhs_width, device="cuda").to(dtype)
    alpha = 0.05 * torch.rand(batches, device="cuda")
    inv_mu = 0.5 + torch.rand(batches, device="cuda")

    actual, info = stats_solve_readout(u, c, alpha, inv_mu)
    u32, c32 = u.float(), c.float()
    gram = u32.transpose(1, 2) @ u32
    rhs = u32.transpose(1, 2) @ c32
    system = torch.eye(rank, device="cuda") + alpha[:, None, None] * gram
    compact = torch.linalg.solve(system, rhs)
    expected = (
        c32 - alpha[:, None, None] * (u32 @ compact)
    ) * inv_mu[:, None, None]

    assert torch.count_nonzero(info).item() == 0
    tolerance = 5e-4 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(
        actual.float(), expected, rtol=tolerance, atol=tolerance
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rank", [16, 32])
def test_mathdx_masked_stats_skips_padding_and_matches_torch(
    dtype: torch.dtype, rank: int
) -> None:
    if not load_mathdx_backend():
        pytest.skip(str(mathdx_load_error()))

    torch.manual_seed(7)
    B, H, N, dh = 3, 4, 129, 48
    u = (0.2 * torch.randn(B, H, N, rank, device="cuda")).to(dtype)
    c = torch.randn(B, H, N, dh, device="cuda").to(dtype)
    lengths = torch.tensor([129, 67, 13], device="cuda")
    mask = torch.arange(N, device="cuda")[None] < lengths[:, None]
    scale = torch.sqrt(129.0 / lengths.float())
    alpha = (0.05 * torch.rand(B * H, device="cuda")).float()

    actual = try_masked_stats_solve_spd(u, c, mask, scale, alpha)
    assert actual is not None
    masked_u = u.float() * mask[:, None, :, None] * scale[:, None, None, None]
    masked_c = c.float() * mask[:, None, :, None]
    u_bh = masked_u.flatten(0, 1)
    c_bh = masked_c.flatten(0, 1)
    gram = u_bh.transpose(1, 2) @ u_bh
    rhs = u_bh.transpose(1, 2) @ c_bh
    system = torch.eye(rank, device="cuda") + alpha[:, None, None] * gram
    expected = torch.linalg.solve(system, rhs)

    tolerance = 5e-4 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mathdx_rank_rotary_matches_torch_and_backward(dtype: torch.dtype) -> None:
    if not load_mathdx_backend():
        pytest.skip(str(mathdx_load_error()))

    torch.manual_seed(2)
    u = torch.randn(3, 4, 65, 32, device="cuda", dtype=dtype, requires_grad=True)
    half = u.shape[-1] // 2
    inv_freq = 10000.0 ** (
        -torch.arange(half, device="cuda", dtype=torch.float32) / half
    )
    angles = torch.arange(u.shape[2], device="cuda", dtype=torch.float32)[:, None] * inv_freq
    cos = angles.cos().to(dtype).contiguous()
    sin = angles.sin().to(dtype).contiguous()

    actual = try_rank_rotary(u, cos, sin)
    assert actual is not None
    expected = apply_rank_rotary(u)
    tolerance = 1e-6 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)

    probe = torch.randn_like(actual)
    actual_grad = torch.autograd.grad((actual * probe).sum(), u, retain_graph=True)[0]
    expected_grad = torch.autograd.grad((expected * probe).sum(), u)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=tolerance, atol=tolerance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("rotary", [False, True])
def test_mathdx_fused_basis_preparation_matches_torch(rotary: bool) -> None:
    if not load_mathdx_backend():
        pytest.skip(str(mathdx_load_error()))

    torch.manual_seed(5)
    u = torch.randn(
        3, 4, 65, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    eps = 1e-5
    length_scale = (1.0 / u.shape[2]) ** 0.5
    normalized = u * torch.rsqrt(torch.mean(u * u, dim=-1, keepdim=True) + eps)
    expected = normalized * length_scale
    cos = sin = None
    if rotary:
        expected = apply_rank_rotary(expected)
        half = u.shape[-1] // 2
        inv_freq = 10000.0 ** (
            -torch.arange(half, device="cuda", dtype=torch.float32) / half
        )
        angles = (
            torch.arange(u.shape[2], device="cuda", dtype=torch.float32)[:, None]
            * inv_freq
        )
        cos = angles.cos().to(u.dtype).contiguous()
        sin = angles.sin().to(u.dtype).contiguous()

    actual = try_prepare_basis(
        u, eps=eps, length_scale=length_scale, cos=cos, sin=sin
    )
    assert actual is not None
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    probe = torch.randn_like(actual)
    actual_grad = torch.autograd.grad((actual * probe).sum(), u, retain_graph=True)[0]
    expected_grad = torch.autograd.grad((expected * probe).sum(), u)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=3e-2, atol=3e-2)
