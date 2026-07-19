from __future__ import annotations

import pytest
import torch

from lsso.mathdx_backend import (
    get_mathdx_path_counters,
    load_mathdx_backend,
    mathdx_load_error,
    reset_mathdx_path_counters,
    solve_spd,
    solve_spd_autograd,
    solve_spd_or_torch,
    try_dual_backward_statistics_tensorcore,
    try_dual_grad_u_tensorcore,
    try_masked_trace_stats_solve_readout,
    try_rank_rotary,
    try_trace_stats_solve_readout,
)


RETIRED_OPS = {
    "solve_spd_bpb",
    "stats_solve_spd",
    "stats_solve_readout",
    "masked_stats_solve_spd",
    "masked_rotary_trace_stats_solve_readout",
    "lsso_backward_gain_alpha_fused",
}


def _require_backend() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if not load_mathdx_backend():
        pytest.skip(str(mathdx_load_error()))


def _trace_reference(
    u: torch.Tensor,
    c: torch.Tensor,
    alpha: torch.Tensor,
    gain: torch.Tensor,
    mask: torch.Tensor | None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, N, rank = u.shape
    active = (
        torch.ones(B, N, device=u.device, dtype=torch.bool)
        if mask is None
        else mask
    )
    active4 = active[:, None, :, None]
    safe_u = torch.where(active4, u, torch.zeros_like(u)).float()
    safe_c = torch.where(active4, c, torch.zeros_like(c)).float()
    ubh, cbh = safe_u.flatten(0, 1), safe_c.flatten(0, 1)
    gram = ubh.transpose(1, 2) @ ubh
    rhs = ubh.transpose(1, 2) @ cbh
    counts = active.sum(1).clamp_min(1).float()
    elements = counts[:, None].expand(B, H).reshape(-1) * rank
    denominator = gram.diagonal(dim1=-2, dim2=-1).sum(-1) + eps * elements
    scale2 = elements / denominator.clamp_min(torch.finfo(torch.float32).tiny)
    effective = alpha * scale2
    system = torch.eye(rank, device=u.device) + effective[:, None, None] * gram
    compact = torch.linalg.solve(system, rhs).to(u.dtype).float()
    output = gain[:, None, None] * (
        cbh - effective[:, None, None] * (ubh @ compact)
    )
    output = output.view(B, H, N, c.shape[-1])
    output = torch.where(active4, output, torch.zeros_like(output))
    shape = (B, H, 1, 1)
    return (
        output,
        effective.view(shape),
        denominator.view(shape),
        scale2.view(shape),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_extension_exposes_only_the_audited_operator_surface() -> None:
    _require_backend()
    for name in RETIRED_OPS:
        assert not hasattr(torch.ops.lsso_mathdx, name)
    for name in {
        "solve_spd",
        "masked_stats_solve_readout",
        "masked_trace_stats_solve_readout",
        "dual_backward_statistics_tensorcore",
        "dual_grad_u_tensorcore",
        "rank_rotary",
    }:
        assert hasattr(torch.ops.lsso_mathdx, name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fp32_trace_uses_portable_fallback_not_a_native_variant() -> None:
    _require_backend()
    u = torch.randn(1, 2, 17, 16, device="cuda")
    c = torch.randn(1, 2, 17, 32, device="cuda")
    alpha = torch.ones(2, device="cuda")
    gain = torch.ones(2, device="cuda")
    with torch.no_grad():
        assert try_trace_stats_solve_readout(
            u, c, alpha, gain, normalization_eps=1e-6,
            length_reference=1.0, length_normalize=False,
        ) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("rank", [1, 16, 24, 32, 48, 56, 64])
def test_bucketed_spd_solve_and_autograd_match_torch(rank: int) -> None:
    _require_backend()
    torch.manual_seed(10 + rank)
    a = torch.randn(5, rank, rank, device="cuda")
    gram_value = a @ a.transpose(1, 2) + torch.eye(rank, device="cuda")
    rhs_value = torch.randn(5, rank, 37, device="cuda")
    actual = solve_spd_or_torch(gram_value, rhs_value)
    expected = torch.linalg.solve(gram_value, rhs_value)
    torch.testing.assert_close(actual, expected, rtol=6e-4, atol=6e-4)

    gram = gram_value.detach().requires_grad_()
    rhs = rhs_value.detach().requires_grad_()
    probe = torch.randn_like(rhs)
    value = solve_spd_autograd(gram, rhs)
    grads = torch.autograd.grad((value * probe).sum(), (gram, rhs))
    gram_ref = gram_value.detach().requires_grad_()
    rhs_ref = rhs_value.detach().requires_grad_()
    reference = torch.linalg.solve(gram_ref, rhs_ref)
    grads_ref = torch.autograd.grad(
        (reference * probe).sum(), (gram_ref, rhs_ref)
    )
    for actual_grad, expected_grad in zip(grads, grads_ref, strict=True):
        torch.testing.assert_close(
            actual_grad, expected_grad, rtol=1.5e-3, atol=1.5e-3
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("rank", [16, 32, 48, 64])
def test_trace_forward_matches_contract_and_never_reads_padding(
    masked: bool, rank: int, monkeypatch
) -> None:
    _require_backend()
    monkeypatch.setenv("LSSO_PATH_COUNTERS", "1")
    reset_mathdx_path_counters()
    torch.manual_seed(100 + rank)
    B, H, N, width = 2, 3, 73, 64
    u = torch.randn(B, H, N, rank, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(B, N, H, width, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    alpha = 0.5 + torch.rand(B * H, device="cuda")
    gain = 0.8 + torch.rand(B * H, device="cuda")
    mask = None
    if masked:
        lengths = torch.tensor([73, 11], device="cuda")
        mask = torch.arange(N, device="cuda")[None] < lengths[:, None]
        padding = ~mask[:, None, :, None]
        u = torch.where(padding, torch.full_like(u, float("nan")), u)
        c = torch.where(padding, torch.full_like(c, float("nan")), c)
        c = c.transpose(1, 2).contiguous().transpose(1, 2)
    with torch.no_grad():
        if masked:
            actual = try_masked_trace_stats_solve_readout(
                u, c, mask, alpha, gain, normalization_eps=1e-6,
                length_reference=1.0, length_normalize=False,
            )
        else:
            actual = try_trace_stats_solve_readout(
                u, c, alpha, gain, normalization_eps=1e-6,
                length_reference=1.0, length_normalize=False,
            )
    assert actual is not None
    assert not c.is_contiguous()
    assert actual[0].stride() == c.stride()
    assert actual[0].transpose(1, 2).is_contiguous()
    expected = _trace_reference(u, c, alpha, gain, mask, 1e-6)
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            value.float(), reference.float(), rtol=3e-2, atol=3e-2
        )
        assert torch.isfinite(value).all()
    expected_counter = (
        "forward.trace_masked_cta" if masked else "forward.trace_unmasked_cta"
    )
    assert get_mathdx_path_counters().get(expected_counter) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dual_backward_tensorcore_paths_match_bmm_and_mask() -> None:
    _require_backend()
    torch.manual_seed(230)
    B, H, N, width, rank = 2, 3, 67, 64, 32
    u = 0.1 * torch.randn(B, H, N, rank, device="cuda", dtype=torch.bfloat16)
    y = (0.1 * torch.randn(
        B, N, H, width, device="cuda", dtype=torch.bfloat16
    )).transpose(1, 2)
    p = (0.1 * torch.randn(
        B, N, H, width, device="cuda", dtype=torch.bfloat16
    )).transpose(1, 2)
    lengths = torch.tensor([67, 9], device="cuda")
    mask = torch.arange(N, device="cuda")[None] < lengths[:, None]
    u_bh = u.flatten(0, 1).contiguous()
    y_bh = y.flatten(0, 1).contiguous()
    p_bh = p.flatten(0, 1).contiguous()
    with torch.no_grad():
        stats = try_dual_backward_statistics_tensorcore(
            u, y, p, valid_mask=mask, heads=H
        )
    assert stats is not None
    ytu, ptu, _ = stats
    active = mask[:, None, :, None]
    uf = torch.where(active, u, 0).float().flatten(0, 1)
    yf = torch.where(active, y, 0).float().flatten(0, 1)
    pf = torch.where(active, p, 0).float().flatten(0, 1)
    torch.testing.assert_close(ytu, yf.transpose(1, 2) @ uf, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(ptu, pf.transpose(1, 2) @ uf, rtol=3e-2, atol=3e-2)

    coefficient = torch.randn(B * H, device="cuda")
    radial = (0.1 * torch.randn_like(uf)).view(B, H, N, rank).to(u.dtype)
    radial_coefficient = torch.randn(B * H, device="cuda")
    with torch.no_grad():
        grad_u = try_dual_grad_u_tensorcore(
            p, y, ytu.to(u.dtype), ptu.to(u.dtype), coefficient,
            radial_u=radial,
            radial_coefficient=radial_coefficient,
            valid_mask=mask, heads=H,
        )
    assert grad_u is not None
    assert grad_u.shape == (B, H, N, rank)
    expected = torch.bmm(p_bh, ytu.to(u.dtype)).float()
    expected += torch.bmm(y_bh, ptu.to(u.dtype)).float()
    expected = expected * coefficient[:, None, None]
    expected += radial.float().flatten(0, 1) * radial_coefficient[:, None, None]
    expected = expected.view(B, H, N, rank)
    expected = torch.where(active, expected, 0)
    torch.testing.assert_close(grad_u.float(), expected, rtol=4e-2, atol=8e-2)
    assert torch.count_nonzero(grad_u.masked_select((~active).expand_as(grad_u))) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rank_rotary_forward_and_inverse_backward_are_consistent() -> None:
    _require_backend()
    torch.manual_seed(301)
    value = torch.randn(
        2, 4, 65, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    angles = torch.randn(65, 16, device="cuda").to(torch.bfloat16)
    cos, sin = angles.cos().contiguous(), angles.sin().contiguous()
    rotated = try_rank_rotary(value, cos, sin)
    assert rotated is not None
    restored = torch.ops.lsso_mathdx.rank_rotary(rotated, cos, sin, True)
    torch.testing.assert_close(restored.float(), value.float(), rtol=2e-2, atol=2e-2)
    probe = torch.randn_like(rotated)
    grad = torch.autograd.grad((rotated * probe).sum(), value)[0]
    expected = torch.ops.lsso_mathdx.rank_rotary(probe, cos, sin, True)
    torch.testing.assert_close(grad.float(), expected.float(), rtol=0, atol=0)
