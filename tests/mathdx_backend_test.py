from __future__ import annotations

import pytest
import torch

from lsso.mathdx_backend import (
    load_mathdx_backend,
    mathdx_load_error,
    solve_spd,
    solve_spd_bpb,
    stats_solve_readout,
    stats_solve_spd,
    try_lsso_backward_fused,
    try_masked_stats_solve_readout,
    try_masked_stats_solve_spd,
    try_prepare_basis,
    try_rank_rotary,
)


@pytest.fixture(autouse=True)
def _isolate_fp32_matmul_precision():
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def test_mathdx_explicit_bpb_rejects_invalid_schedule() -> None:
    gram = torch.empty(1, 16, 16)
    rhs = torch.empty(1, 16, 1)
    with pytest.raises(ValueError, match="batches_per_block"):
        solve_spd_bpb(gram, rhs, 3)
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
@pytest.mark.parametrize(
    ("B", "H", "sequence"),
    [
        (3, 4, 1025),   # long-sequence tiled statistics
        (9, 8, 129),    # B * H > the former 64-system gate
        (9, 8, 1025),   # both paths at once
    ],
)
def test_mathdx_masked_stats_supports_long_sequences_and_large_system_grids(
    B: int, H: int, sequence: int
) -> None:
    if not load_mathdx_backend():
        pytest.skip(str(mathdx_load_error()))

    torch.manual_seed(1701 + sequence + B * H)
    rank, rhs_width = 16, 32
    u = (
        0.05
        * torch.randn(B, H, sequence, rank, device="cuda", dtype=torch.float32)
    ).to(torch.bfloat16)
    c = torch.randn(
        B, H, sequence, rhs_width, device="cuda", dtype=torch.bfloat16
    )
    lengths = torch.linspace(sequence, 1, B, device="cuda").long()
    mask = (torch.arange(sequence, device="cuda")[None] < lengths[:, None]).contiguous()
    scale = torch.sqrt(
        torch.tensor(float(sequence), device="cuda") / lengths.float()
    ).contiguous()
    alpha = (0.03 * torch.rand(B * H, device="cuda")).float()

    # Exercise the public eligibility path, not the operator directly: this
    # catches accidental reintroduction of either historical frontend cap.
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

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mathdx_masked_stats_optional_sequence_policy_cap() -> None:
    if not load_mathdx_backend():
        pytest.skip(str(mathdx_load_error()))

    B, H, N, rank, rhs_width = 1, 2, 513, 16, 16
    u = torch.randn(B, H, N, rank, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(B, H, N, rhs_width, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(B, N, device="cuda", dtype=torch.bool)
    scale = torch.ones(B, device="cuda")
    alpha = torch.full((B * H,), 0.01, device="cuda")

    assert try_masked_stats_solve_spd(u, c, mask, scale, alpha) is not None
    assert (
        try_masked_stats_solve_spd(
            u, c, mask, scale, alpha, max_sequence=512
        )
        is None
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mathdx_masked_stats_zero_sync_padding_hint_dispatch() -> None:
    if not load_mathdx_backend():
        pytest.skip(str(mathdx_load_error()))

    B, H, N, rank, rhs_width = 2, 4, 1025, 16, 32
    u = torch.randn(B, H, N, rank, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(B, H, N, rhs_width, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(B, N, device="cuda", dtype=torch.bool)
    scale = torch.ones(B, device="cuda")
    alpha = torch.full((B * H,), 0.01, device="cuda")

    assert (
        try_masked_stats_solve_spd(
            u, c, mask, scale, alpha, padding_ratio_hint=0.25
        )
        is None
    )
    assert (
        try_masked_stats_solve_spd(
            u, c, mask, scale, alpha, padding_ratio_hint=0.9
        )
        is not None
    )
    with pytest.raises(ValueError, match="padding_ratio_hint"):
        try_masked_stats_solve_spd(
            u, c, mask, scale, alpha, padding_ratio_hint=1.1
        )


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_masked_fused_readout_is_padding_safe(dtype: torch.dtype) -> None:
    if not load_mathdx_backend():
        pytest.skip(str(mathdx_load_error()))

    torch.manual_seed(91)
    B, H, N, rank, width = 3, 2, 97, 16, 48
    lengths = torch.tensor([97, 51, 7], device="cuda")
    mask = torch.arange(N, device="cuda")[None, :] < lengths[:, None]
    scale = torch.sqrt(torch.tensor(float(N), device="cuda") / lengths.float())
    u = (0.1 * torch.randn(B, H, N, rank, device="cuda")).to(dtype)
    c = torch.randn(B, H, N, width, device="cuda").to(dtype)
    # Poison padding rather than merely changing it. Any predicated-load bug
    # now turns valid outputs or compact statistics into NaNs.
    padding = ~mask[:, None, :, None]
    u = torch.where(padding, torch.full_like(u, float("nan")), u)
    c = torch.where(padding, torch.full_like(c, float("nan")), c)
    alpha = 0.02 * torch.rand(B * H, device="cuda")
    inv_mu = 0.5 + torch.rand(B * H, device="cuda")

    with torch.no_grad():
        actual = try_masked_stats_solve_readout(
            u, c, mask, scale, alpha, inv_mu, padding_ratio_hint=0.9
        )
    assert actual is not None
    safe_u = torch.where(padding, torch.zeros_like(u), u).float()
    safe_c = torch.where(padding, torch.zeros_like(c), c).float()
    scaled_u = safe_u * scale[:, None, None, None]
    ubh, cbh = scaled_u.flatten(0, 1), safe_c.flatten(0, 1)
    gram = ubh.transpose(1, 2) @ ubh
    rhs = ubh.transpose(1, 2) @ cbh
    system = torch.eye(rank, device="cuda") + alpha[:, None, None] * gram
    compact = torch.linalg.solve(system, rhs)
    expected = (
        cbh - alpha[:, None, None] * (ubh @ compact)
    ) * inv_mu[:, None, None]
    expected = expected.view(B, H, N, width)
    expected = expected * mask[:, None, :, None]
    tolerance = 8e-4 if dtype == torch.float32 else 3e-2
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual.masked_select(padding.expand_as(actual))) == 0
    torch.testing.assert_close(actual.float(), expected, rtol=tolerance, atol=tolerance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("masked", [False, True])
def test_fused_backward_matches_reference_and_does_not_leak_mask(masked: bool) -> None:
    if not load_mathdx_backend():
        pytest.skip(str(mathdx_load_error()))

    torch.manual_seed(92 + masked)
    B, H, N, rank, width = 2, 3, 41, 16, 32
    u = torch.randn(B, H, N, rank, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(B, H, N, width, device="cuda", dtype=torch.bfloat16)
    p = torch.randn(B, H, N, width, device="cuda", dtype=torch.bfloat16)
    gamma = (0.1 + torch.rand(B * H, device="cuda")).float()
    mask = torch.tensor(
        [[True] * N, [True] * 9 + [False] * (N - 9)], device="cuda"
    )
    scale = torch.tensor([1.0, 1.7], device="cuda")
    padding = ~mask[:, None, :, None]
    if masked:
        u = torch.where(padding, torch.full_like(u, float("nan")), u)
        y = torch.where(padding, torch.full_like(y, float("nan")), y)
        p = torch.where(padding, torch.full_like(p, float("nan")), p)

    with torch.no_grad():
        actual = try_lsso_backward_fused(
            u, y, p, gamma,
            valid_mask=mask if masked else None,
            length_scale=scale if masked else None,
        )
    assert actual is not None
    grad_u, grad_mu, grad_gamma = actual
    active = mask[:, None, :, None] if masked else torch.ones(
        B, 1, N, 1, device="cuda", dtype=torch.bool
    )
    uf = torch.where(active, u, torch.zeros_like(u)).float().flatten(0, 1)
    yf = torch.where(active, y, torch.zeros_like(y)).float().flatten(0, 1)
    pf = torch.where(active, p, torch.zeros_like(p)).float().flatten(0, 1)
    ytu = yf.transpose(1, 2) @ uf
    ptu = pf.transpose(1, 2) @ uf
    scale2 = (
        scale.square()[:, None].expand(B, H).reshape(B * H)
        if masked else torch.ones(B * H, device="cuda")
    )
    expected_u = -(pf @ ytu + yf @ ptu)
    expected_u *= (gamma * scale2)[:, None, None]
    expected_u = expected_u.view(B, H, N, rank)
    expected_mu = -(pf * yf).sum(dim=(1, 2))
    expected_gamma = -(ptu * ytu).sum(dim=(1, 2)) * scale2
    assert torch.isfinite(grad_u).all()
    assert torch.isfinite(grad_mu).all()
    assert torch.isfinite(grad_gamma).all()
    if masked:
        assert torch.count_nonzero(grad_u.masked_select(padding.expand_as(grad_u))) == 0
    torch.testing.assert_close(grad_u.float(), expected_u, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(grad_mu, expected_mu, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(grad_gamma, expected_gamma, rtol=2e-4, atol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_backward_respects_sequence_dispatch_cap() -> None:
    if not load_mathdx_backend():
        pytest.skip(str(mathdx_load_error()))
    B, H, N, rank, width = 1, 2, 513, 16, 32
    u = torch.randn(B, H, N, rank, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(B, H, N, width, device="cuda", dtype=torch.bfloat16)
    p = torch.randn_like(y)
    gamma = torch.ones(B * H, device="cuda")
    with torch.no_grad():
        assert try_lsso_backward_fused(u, y, p, gamma) is None
        assert try_lsso_backward_fused(
            u, y, p, gamma, max_sequence=None
        ) is not None
