"""Archived broad test suite for the pre-audit CUDA operator surface."""

from __future__ import annotations

import pytest
import torch

from lsso.mathdx_backend import (
    _scalar_backward_shape_eligible,
    load_mathdx_backend,
    mathdx_load_error,
    solve_spd,
    solve_spd_autograd,
    solve_spd_bpb,
    solve_spd_or_torch,
    stats_solve_readout,
    stats_solve_spd,
    try_lsso_backward_fused,
    try_dual_backward_statistics_tensorcore,
    try_dual_grad_u_tensorcore,
    try_masked_rotary_trace_stats_solve_readout,
    try_masked_trace_stats_solve_readout,
    try_masked_stats_solve_readout,
    try_masked_stats_solve_spd,
    try_stats_solve_readout,
    try_trace_stats_solve_readout,
    try_rank_rotary,
)


def test_info_debug_check_is_opt_in_and_reports_failed_systems(monkeypatch) -> None:
    import lsso.mathdx_backend as backend

    bad = torch.tensor([0, 2, 0, -1], dtype=torch.int32)
    backend._check_mathdx_info(bad, "disabled")
    monkeypatch.setenv("LSSO_MATHDX_DEBUG_INFO", "1")
    with pytest.raises(RuntimeError, match=r"systems \[1, 3\].*\[2, -1\]"):
        backend._check_mathdx_info(bad, "trace-test")
    backend._check_mathdx_info(torch.zeros_like(bad), "trace-test")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_path_counters_identify_trace_and_backward_routes(monkeypatch) -> None:
    import lsso.mathdx_backend as backend

    if not load_mathdx_backend():
        pytest.skip(str(mathdx_load_error()))
    monkeypatch.setenv("LSSO_PATH_COUNTERS", "1")
    backend.reset_mathdx_path_counters()
    B, H, N, rank, width = 2, 2, 31, 32, 32
    u = torch.randn(B, H, N, rank, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(B, H, N, width, device="cuda", dtype=torch.bfloat16)
    alpha = torch.full((B * H,), 0.4, device="cuda")
    gain = torch.ones(B * H, device="cuda")
    with torch.no_grad():
        result = try_trace_stats_solve_readout(
            u, c, alpha, gain, normalization_eps=1e-5,
            length_reference=1.0, length_normalize=False,
        )
    assert result is not None
    systems = B * H
    y = torch.randn(systems, N, width, device="cuda", dtype=torch.bfloat16)
    p = torch.randn_like(y)
    with torch.no_grad():
        stats = try_dual_backward_statistics_tensorcore(
            u.flatten(0, 1), y, p
        )
    assert stats is not None
    counters = backend.get_mathdx_path_counters()
    assert counters["forward.trace_unmasked_cta"] == 1
    assert counters["backward.dual_statistics"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rank", [16, 32, 48, 64])
@pytest.mark.parametrize("masked", [False, True])
def test_trace_single_kernel_ranks_match_compact_readout_contract_and_mask(
    dtype: torch.dtype, rank: int, masked: bool
) -> None:
    """Trace native paths round the FP32 compact solve once before readout."""

    if not load_mathdx_backend():
        pytest.skip(str(mathdx_load_error()))
    torch.manual_seed(3100 + rank + int(masked))
    B, H, N, width = 2, 2, 53, 32
    u = (0.05 * torch.randn(B, H, N, rank, device="cuda")).to(dtype)
    c = torch.randn(B, H, N, width, device="cuda", dtype=dtype)
    alpha = 0.1 + 0.8 * torch.rand(B * H, device="cuda")
    gain = 0.5 + torch.rand(B * H, device="cuda")
    mask = torch.arange(N, device="cuda")[None, :] < torch.tensor(
        [N, 17], device="cuda"
    )[:, None]
    active = mask[:, None, :, None]
    if masked:
        input_u = torch.where(active, u, torch.full_like(u, float("nan")))
        input_c = torch.where(active, c, torch.full_like(c, float("nan")))
        with torch.no_grad():
            result = try_masked_trace_stats_solve_readout(
                input_u,
                input_c,
                mask,
                alpha,
                gain,
                normalization_eps=1e-5,
                length_reference=1.0,
                length_normalize=False,
            )
        safe_u = torch.where(active, u, torch.zeros_like(u))
        safe_c = torch.where(active, c, torch.zeros_like(c))
        lengths = mask.sum(-1, dtype=torch.float32)
    else:
        with torch.no_grad():
            result = try_trace_stats_solve_readout(
                u,
                c,
                alpha,
                gain,
                normalization_eps=1e-5,
                length_reference=1.0,
                length_normalize=False,
            )
        safe_u, safe_c = u, c
        lengths = torch.full((B,), float(N), device="cuda")
    assert result is not None
    actual, effective, denominator, scale_squared = result

    uf = safe_u.float().flatten(0, 1)
    cf = safe_c.float().flatten(0, 1)
    gram = uf.transpose(1, 2) @ uf
    cross = uf.transpose(1, 2) @ cf
    element_count = lengths[:, None].expand(B, H).reshape(-1) * rank
    expected_denominator = (
        gram.diagonal(dim1=-2, dim2=-1).sum(-1) + 1e-5 * element_count
    )
    expected_scale = element_count / expected_denominator
    expected_effective = alpha * expected_scale
    system = (
        torch.eye(rank, device="cuda")[None]
        + expected_effective[:, None, None] * gram
    )
    # This dtype conversion is the shared native/fallback precision contract.
    compact = torch.linalg.solve(system, cross).to(dtype)
    expected = cf - uf @ (
        compact.float() * expected_effective[:, None, None]
    )
    expected = expected.view(B, H, N, width) * gain.view(B, H, 1, 1)
    if masked:
        expected = torch.where(active, expected, torch.zeros_like(expected))
    tolerance = 3e-3 if dtype == torch.float16 else 2e-2
    assert torch.isfinite(actual).all()
    if masked:
        assert torch.count_nonzero(
            actual.masked_select((~active).expand_as(actual))
        ) == 0
    torch.testing.assert_close(
        actual.float(), expected, rtol=tolerance, atol=tolerance
    )
    torch.testing.assert_close(
        effective.flatten(), expected_effective, rtol=3e-4, atol=3e-4
    )
    torch.testing.assert_close(
        denominator.flatten(), expected_denominator, rtol=3e-4, atol=3e-4
    )
    torch.testing.assert_close(
        scale_squared.flatten(), expected_scale, rtol=3e-4, atol=3e-4
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rank", [16, 32, 48, 64])
def test_dual_backward_statistics_reuses_u_without_changing_values(
    dtype: torch.dtype, rank: int
) -> None:
    if not load_mathdx_backend():
        pytest.skip(str(mathdx_load_error()))
    torch.manual_seed(71 + rank)
    u = torch.randn(5, 37, rank, device="cuda", dtype=dtype)
    y = torch.randn(5, 37, 64, device="cuda", dtype=dtype)
    p = torch.randn_like(y)
    with torch.no_grad():
        actual = try_dual_backward_statistics_tensorcore(u, y, p)
    assert actual is not None
    ytu, ptu, grad_mu = actual
    torch.testing.assert_close(
        ytu, y.float().transpose(1, 2) @ u.float(), rtol=2e-4, atol=2e-4
    )
    torch.testing.assert_close(
        ptu, p.float().transpose(1, 2) @ u.float(), rtol=2e-4, atol=2e-4
    )
    torch.testing.assert_close(
        grad_mu, -(p.float() * y.float()).sum((1, 2)), rtol=2e-4, atol=2e-4
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rank", [16, 32, 48, 64])
def test_trace_backward_dispatch_predicates_mask_and_fuses_radial_epilogue(
    dtype: torch.dtype, rank: int
) -> None:
    if not load_mathdx_backend():
        pytest.skip(str(mathdx_load_error()))
    torch.manual_seed(3200 + rank)
    B, H, N, width = 2, 3, 41, 64
    systems = B * H
    mask = torch.arange(N, device="cuda")[None, :] < torch.tensor(
        [N, 7], device="cuda"
    )[:, None]
    active = mask.repeat_interleave(H, dim=0).unsqueeze(-1)
    u = torch.randn(systems, N, rank, device="cuda", dtype=dtype)
    y = torch.randn(systems, N, width, device="cuda", dtype=dtype)
    p = torch.randn_like(y)
    u = torch.where(active, u, torch.full_like(u, float("nan")))
    y = torch.where(active, y, torch.full_like(y, float("nan")))
    p = torch.where(active, p, torch.full_like(p, float("nan")))
    safe_u = torch.where(active, u, torch.zeros_like(u))
    safe_y = torch.where(active, y, torch.zeros_like(y))
    safe_p = torch.where(active, p, torch.zeros_like(p))

    with torch.no_grad():
        statistics = try_dual_backward_statistics_tensorcore(
            u, y, p, valid_mask=mask, heads=H
        )
    assert statistics is not None
    ytu, ptu, grad_mu = statistics
    expected_ytu = safe_y.float().transpose(1, 2) @ safe_u.float()
    expected_ptu = safe_p.float().transpose(1, 2) @ safe_u.float()
    expected_mu = -(safe_p.float() * safe_y.float()).sum((1, 2))
    torch.testing.assert_close(ytu, expected_ytu, rtol=3e-4, atol=3e-4)
    torch.testing.assert_close(ptu, expected_ptu, rtol=3e-4, atol=3e-4)
    torch.testing.assert_close(grad_mu, expected_mu, rtol=3e-4, atol=3e-4)

    coefficient = torch.randn(systems, device="cuda")
    radial = torch.randn(systems, device="cuda")
    ytu_readout = ytu.to(dtype)
    ptu_readout = ptu.to(dtype)
    with torch.no_grad():
        grad_u = try_dual_grad_u_tensorcore(
            p,
            y,
            ytu_readout,
            ptu_readout,
            coefficient,
            radial_u=u,
            radial_coefficient=radial,
            valid_mask=mask,
            heads=H,
        )
    assert grad_u is not None
    expected_grad_u = (
        safe_p.float() @ ytu_readout.float()
        + safe_y.float() @ ptu_readout.float()
    ) * coefficient[:, None, None]
    expected_grad_u.add_(safe_u.float() * radial[:, None, None])
    tolerance = 2e-2 if dtype == torch.float16 else 7e-2
    assert torch.isfinite(grad_u).all()
    assert torch.count_nonzero(
        grad_u.masked_select((~active).expand_as(grad_u))
    ) == 0
    torch.testing.assert_close(
        grad_u.float(), expected_grad_u, rtol=tolerance, atol=tolerance
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tensorcore_grad_u_fuses_trace_radial_epilogue() -> None:
    if not load_mathdx_backend():
        pytest.skip(str(mathdx_load_error()))
    torch.manual_seed(81)
    systems, sequence, width, rank = 7, 41, 64, 32
    p = torch.randn(systems, sequence, width, device="cuda", dtype=torch.bfloat16)
    y = torch.randn_like(p)
    ytu = torch.randn(systems, width, rank, device="cuda", dtype=torch.bfloat16)
    ptu = torch.randn_like(ytu)
    u = torch.randn(systems, sequence, rank, device="cuda", dtype=torch.bfloat16)
    coefficient = torch.randn(systems, device="cuda")
    radial = torch.randn(systems, device="cuda")
    with torch.no_grad():
        actual = try_dual_grad_u_tensorcore(
            p,
            y,
            ytu,
            ptu,
            coefficient,
            radial_u=u,
            radial_coefficient=radial,
        )
    assert actual is not None
    expected = (
        coefficient[:, None, None]
        * (p.float() @ ytu.float() + y.float() @ ptu.float())
        + radial[:, None, None] * u.float()
    ).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_masked_trace_kernel_skips_padding_and_returns_trace_state(monkeypatch) -> None:
    if not load_mathdx_backend():
        pytest.skip(str(mathdx_load_error()))
    monkeypatch.setenv("LSSO_MATHDX_MASKED_TRACE", "1")
    torch.manual_seed(91)
    B, H, N, rank, width = 3, 4, 73, 32, 64
    u = (0.1 * torch.randn(B, H, N, rank, device="cuda")).bfloat16()
    c = torch.randn(B, H, N, width, device="cuda", dtype=torch.bfloat16)
    lengths = torch.tensor([73, 41, 9], device="cuda")
    mask = torch.arange(N, device="cuda")[None, :] < lengths[:, None]
    alpha = 0.2 * torch.rand(B * H, device="cuda")
    gain = 0.5 + torch.rand(B * H, device="cuda")
    with torch.no_grad():
        actual = try_masked_trace_stats_solve_readout(
            u,
            c,
            mask,
            alpha,
            gain,
            normalization_eps=1e-5,
            length_reference=64.0,
            length_normalize=True,
        )
    assert actual is not None
    output, effective, denominator, scale_squared = actual
    active = mask[:, None, :, None]
    uf = torch.where(active, u.float(), 0.0)
    cf = torch.where(active, c.float(), 0.0)
    gram = uf.transpose(-1, -2) @ uf
    cross = uf.transpose(-1, -2) @ cf
    energy = gram.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
    expected_denominator = energy + 1e-5 * lengths[:, None, None, None] * rank
    expected_scale = (rank * 64.0) / expected_denominator
    expected_effective = alpha.view(B, H, 1, 1) * expected_scale
    compact = torch.linalg.solve(
        torch.eye(rank, device="cuda").view(1, 1, rank, rank)
        + expected_effective * gram,
        cross,
    )
    expected = (cf - expected_effective * (uf @ compact)) * gain.view(B, H, 1, 1)
    expected = torch.where(active, expected, 0.0)
    torch.testing.assert_close(output.float(), expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(effective, expected_effective, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(denominator, expected_denominator, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(scale_squared, expected_scale, rtol=2e-4, atol=2e-4)
    assert torch.count_nonzero(output.masked_select(~active.expand_as(output))) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rank_rotary_handles_strided_layout_and_per_sample_positions() -> None:
    if not load_mathdx_backend():
        pytest.skip(str(mathdx_load_error()))
    torch.manual_seed(101)
    B, H, N, rank = 3, 4, 29, 32
    token_major = torch.randn(B, N, H, rank, device="cuda", dtype=torch.bfloat16)
    u = token_major.transpose(1, 2)
    positions = torch.stack(
        [torch.arange(N, device="cuda") * (sample + 1) for sample in range(B)]
    )
    inv_freq = 10000.0 ** (
        -torch.arange(rank // 2, device="cuda") / (rank // 2)
    )
    angles = positions[:, :, None] * inv_freq
    cos = angles.cos().to(u.dtype).view(B, 1, N, rank // 2).contiguous()
    sin = angles.sin().to(u.dtype).view(B, 1, N, rank // 2).contiguous()
    actual = try_rank_rotary(u, cos, sin)
    assert actual is not None and actual.is_contiguous()
    expected = torch.empty(u.shape, device="cuda", dtype=u.dtype)
    even, odd = u[..., 0::2].float(), u[..., 1::2].float()
    expected[..., 0::2] = (even * cos.float() - odd * sin.float()).to(u.dtype)
    expected[..., 1::2] = (even * sin.float() + odd * cos.float()).to(u.dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rotary_trace_fusion_matches_materialized_contract(monkeypatch) -> None:
    if not load_mathdx_backend():
        pytest.skip(str(mathdx_load_error()))
    monkeypatch.setenv("LSSO_MATHDX_ROTARY_STATS", "1")
    monkeypatch.setenv("LSSO_MATHDX_MASKED_TRACE", "1")
    torch.manual_seed(111)
    B, H, N, rank, width = 2, 4, 65, 32, 64
    uc = torch.randn(
        B, N, H * (rank + width), device="cuda", dtype=torch.bfloat16
    )
    u_flat, c_flat = uc.split((H * rank, H * width), dim=-1)
    u = u_flat.view(B, N, H, rank).transpose(1, 2)
    c = c_flat.view(B, N, H, width).transpose(1, 2)
    mask = torch.arange(N, device="cuda")[None, :] < torch.tensor(
        [65, 13], device="cuda"
    )[:, None]
    inv_freq = 10000.0 ** (
        -torch.arange(rank // 2, device="cuda") / (rank // 2)
    )
    angles = torch.arange(N, device="cuda")[:, None] * inv_freq
    cos = angles.cos().to(u.dtype).view(1, 1, N, rank // 2).contiguous()
    sin = angles.sin().to(u.dtype).view(1, 1, N, rank // 2).contiguous()
    alpha = torch.full((B * H,), 0.1, device="cuda")
    gain = torch.ones(B * H, device="cuda")
    with torch.no_grad():
        fused = try_masked_rotary_trace_stats_solve_readout(
            u,
            c,
            mask,
            cos,
            sin,
            alpha,
            gain,
            normalization_eps=1e-5,
            length_reference=64.0,
            length_normalize=True,
        )
        rotated = try_rank_rotary(u, cos, sin)
        assert rotated is not None
        explicit = try_masked_trace_stats_solve_readout(
            rotated,
            c.contiguous(),
            mask,
            alpha,
            gain,
            normalization_eps=1e-5,
            length_reference=64.0,
            length_normalize=True,
        )
    assert fused is not None and explicit is not None
    for actual, expected in zip(fused, explicit, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    output = fused[0]
    assert output.transpose(1, 2).is_contiguous()


def test_scalar_backward_dispatches_only_launch_bound_shapes(monkeypatch) -> None:
    monkeypatch.delenv("LSSO_MATHDX_FUSED_BACKWARD", raising=False)
    assert _scalar_backward_shape_eligible(64, 197, 32, 64)
    assert not _scalar_backward_shape_eligible(65, 197, 32, 64)
    monkeypatch.setenv("LSSO_MATHDX_FUSED_BACKWARD", "1")
    assert _scalar_backward_shape_eligible(9216, 197, 32, 64)
    monkeypatch.setenv("LSSO_MATHDX_FUSED_BACKWARD", "0")
    assert not _scalar_backward_shape_eligible(1, 17, 16, 16)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_retired_token_rms_ops_are_not_registered() -> None:
    if not load_mathdx_backend():
        pytest.skip(str(mathdx_load_error()))
    for name in (
        "normalize_basis",
        "normalize_basis_backward",
        "normalize_rank_rotary",
        "normalize_rank_rotary_backward",
    ):
        assert not hasattr(torch.ops.lsso_mathdx, name)
from lsso.modules_v2 import apply_rank_rotary


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("rank", [16, 32, 48, 64])
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
@pytest.mark.parametrize("rank", [48, 64])
def test_mathdx_large_rank_rejects_multi_system_cta_schedule(rank: int) -> None:
    if not load_mathdx_backend():
        pytest.skip(str(mathdx_load_error()))
    gram = torch.eye(rank, device="cuda").expand(4, rank, rank).contiguous()
    rhs = torch.randn(4, rank, 32, device="cuda")
    with pytest.raises(RuntimeError, match="rank-48/64.*batches_per_block=1"):
        solve_spd_bpb(gram, rhs, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("rank", [1, 7, 15, 17, 24, 31, 33, 40, 47, 49, 56, 63])
def test_mathdx_arbitrary_rank_bucket_matches_torch(rank: int) -> None:
    if not load_mathdx_backend():
        pytest.skip(str(mathdx_load_error()))
    torch.manual_seed(1000 + rank)
    a = torch.randn(5, rank, rank, device="cuda")
    gram = a @ a.transpose(1, 2) + 0.5 * torch.eye(rank, device="cuda")
    rhs = torch.randn(5, rank, 19, device="cuda")
    with torch.no_grad():
        actual = solve_spd_or_torch(gram, rhs)
        expected = torch.linalg.solve(gram, rhs)
    torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("rank", [7, 24, 40, 56])
def test_mathdx_arbitrary_rank_bucket_autograd_matches_torch(rank: int) -> None:
    if not load_mathdx_backend():
        pytest.skip(str(mathdx_load_error()))
    torch.manual_seed(2000 + rank)
    a = torch.randn(3, rank, rank, device="cuda")
    gram_value = a @ a.transpose(1, 2) + torch.eye(rank, device="cuda")
    rhs_value = torch.randn(3, rank, 11, device="cuda")
    probe = torch.randn_like(rhs_value)

    gram = gram_value.detach().requires_grad_()
    rhs = rhs_value.detach().requires_grad_()
    actual = solve_spd_autograd(gram, rhs)
    actual_grads = torch.autograd.grad((actual * probe).sum(), (gram, rhs))

    gram_ref = gram_value.detach().requires_grad_()
    rhs_ref = rhs_value.detach().requires_grad_()
    expected = torch.linalg.solve(gram_ref, rhs_ref)
    expected_grads = torch.autograd.grad(
        (expected * probe).sum(), (gram_ref, rhs_ref)
    )
    torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-4)
    for value, reference in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(value, reference, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rank_above_largest_bucket_uses_torch_fallback() -> None:
    rank = 65
    a = torch.randn(2, rank, rank, device="cuda")
    gram = a @ a.transpose(1, 2) + torch.eye(rank, device="cuda")
    rhs = torch.randn(2, rank, 9, device="cuda")
    actual = solve_spd_or_torch(gram, rhs)
    expected = torch.linalg.solve(gram, rhs)
    torch.testing.assert_close(actual, expected)


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
@pytest.mark.parametrize("rhs_width", [48, 64, 96, 128, 192])
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
@pytest.mark.parametrize("rhs_width", [48, 96, 192])
def test_mathdx_masked_stats_skips_padding_and_matches_torch(
    dtype: torch.dtype, rank: int, rhs_width: int
) -> None:
    if not load_mathdx_backend():
        pytest.skip(str(mathdx_load_error()))

    torch.manual_seed(7)
    B, H, N, dh = 3, 4, 129, rhs_width
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
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("width", [48, 96, 192])
def test_masked_fused_readout_is_padding_safe(
    dtype: torch.dtype, width: int
) -> None:
    if not load_mathdx_backend():
        pytest.skip(str(mathdx_load_error()))

    torch.manual_seed(91)
    B, H, N, rank = 3, 2, 97, 16
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
    gain = (0.8 + torch.rand(B * H, device="cuda")).float()
    alpha = (0.1 + torch.rand(B * H, device="cuda")).float()
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
            u, y, p, gain, alpha,
            valid_mask=mask if masked else None,
            length_scale=scale if masked else None,
        )
    assert actual is not None
    grad_u, grad_gain, grad_alpha = actual
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
    expected_u *= ((alpha / gain) * scale2)[:, None, None]
    expected_u = expected_u.view(B, H, N, rank)
    grad_mu_fixed_gamma = -(pf * yf).sum(dim=(1, 2))
    grad_gamma_fixed_mu = -(ptu * ytu).sum(dim=(1, 2)) * scale2
    expected_gain = -(
        grad_mu_fixed_gamma + alpha * grad_gamma_fixed_mu
    ) / gain.square()
    expected_alpha = grad_gamma_fixed_mu / gain
    assert torch.isfinite(grad_u).all()
    assert torch.isfinite(grad_gain).all()
    assert torch.isfinite(grad_alpha).all()
    if masked:
        assert torch.count_nonzero(grad_u.masked_select(padding.expand_as(grad_u))) == 0
    torch.testing.assert_close(grad_u.float(), expected_u, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(grad_gain, expected_gain, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(grad_alpha, expected_alpha, rtol=2e-4, atol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_backward_respects_sequence_dispatch_cap() -> None:
    if not load_mathdx_backend():
        pytest.skip(str(mathdx_load_error()))
    B, H, N, rank, width = 1, 2, 513, 16, 32
    u = torch.randn(B, H, N, rank, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(B, H, N, width, device="cuda", dtype=torch.bfloat16)
    p = torch.randn_like(y)
    gain = torch.ones(B * H, device="cuda")
    alpha = torch.ones(B * H, device="cuda")
    with torch.no_grad():
        assert try_lsso_backward_fused(u, y, p, gain, alpha) is None
        assert try_lsso_backward_fused(
            u, y, p, gain, alpha, max_sequence=None
        ) is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rank", [16, 32, 48, 64])
def test_dual_tensorcore_grad_u_matches_bmm_and_padding_is_zero(
    dtype: torch.dtype, rank: int, monkeypatch
) -> None:
    if not load_mathdx_backend():
        pytest.skip(str(mathdx_load_error()))
    torch.manual_seed(704 + rank)
    systems, sequence, width = 5, 19, 64
    p = torch.randn(systems, sequence, width, device="cuda", dtype=dtype)
    y = torch.randn_like(p)
    p[:, -3:] = 0
    y[:, -3:] = 0
    ytu = torch.randn(systems, width, rank, device="cuda", dtype=dtype)
    ptu = torch.randn_like(ytu)
    coefficient = torch.randn(systems, device="cuda")
    with torch.no_grad():
        actual = try_dual_grad_u_tensorcore(p, y, ytu, ptu, coefficient)
    assert actual is not None
    monkeypatch.setenv("LSSO_MATHDX_DUAL_GRAD_U", "0")
    with torch.no_grad():
        assert try_dual_grad_u_tensorcore(p, y, ytu, ptu, coefficient) is None
    expected = torch.bmm(p.float(), ytu.float())
    expected.add_(torch.bmm(y.float(), ptu.float()))
    expected.mul_(coefficient[:, None, None])
    tolerance = 2e-2 if dtype == torch.bfloat16 else 8e-3
    torch.testing.assert_close(
        actual.float(), expected, rtol=tolerance, atol=tolerance
    )
    assert torch.count_nonzero(actual[:, -3:]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_wide_rhs_forward_dispatch_policy() -> None:
    if not load_mathdx_backend():
        pytest.skip(str(mathdx_load_error()))

    def attempt(systems: int, sequence: int, width: int):
        u = torch.randn(
            systems, sequence, 16, device="cuda", dtype=torch.bfloat16
        )
        c = torch.randn(
            systems, sequence, width, device="cuda", dtype=torch.bfloat16
        )
        alpha = torch.full((systems,), 0.01, device="cuda")
        inv_mu = torch.ones(systems, device="cuda")
        with torch.no_grad():
            return try_stats_solve_readout(u, c, alpha, inv_mu)

    assert attempt(8, 197, 96) is not None
    assert attempt(8, 197, 128) is not None
    assert attempt(8, 197, 192) is not None
    assert attempt(8, 197, 256) is None
    assert attempt(129, 197, 96) is None
    assert attempt(8, 257, 96) is None
    # The established <=64 path keeps its original, less restrictive policy.
    assert attempt(129, 257, 64) is not None
