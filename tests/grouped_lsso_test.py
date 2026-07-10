from __future__ import annotations

import torch

from lsso import GroupedLSSO, GroupedRRLSSO, LSSO, RRLSSO, lsso
from lsso.modules_v2 import apply_rank_rotary


def test_grouped_lsso_matches_per_head_lsso_when_groups_equal_heads() -> None:
    torch.manual_seed(0)
    B, N, D, H, r = 2, 13, 48, 3, 8
    x = torch.randn(B, N, D)
    reference = LSSO(dim=D, num_heads=H, rank=r)
    grouped = GroupedLSSO(
        dim=D,
        num_heads=H,
        num_relation_groups=H,
        rank=r,
    )
    grouped.load_state_dict(reference.state_dict(), strict=True)

    torch.testing.assert_close(grouped(x), reference(x), atol=1e-6, rtol=1e-6)


def test_grouped_rrlsso_matches_per_head_rrlsso_when_groups_equal_heads() -> None:
    torch.manual_seed(1)
    B, N, D, H, r = 2, 11, 64, 4, 8
    x = torch.randn(B, N, D)
    position_ids = torch.arange(N) * 2
    reference = RRLSSO(dim=D, num_heads=H, rank=r)
    grouped = GroupedRRLSSO(
        dim=D,
        num_heads=H,
        num_relation_groups=H,
        rank=r,
    )
    grouped.load_state_dict(reference.state_dict(), strict=True)

    torch.testing.assert_close(
        grouped(x, position_ids=position_ids),
        reference(x, position_ids=position_ids),
        atol=1e-6,
        rtol=1e-6,
    )


def test_grouped_lsso_matches_manual_grouped_solve() -> None:
    torch.manual_seed(2)
    B, N, D, H, G, r = 2, 9, 64, 4, 2, 8
    x = torch.randn(B, N, D)
    layer = GroupedLSSO(
        dim=D,
        num_heads=H,
        num_relation_groups=G,
        rank=r,
    )

    UC = layer.w_uc(x)
    U, C = UC.split((G * r, D), dim=-1)
    U = U.view(B, N, G, r).transpose(1, 2).contiguous()
    C = C.view(B, N, G, D // G).transpose(1, 2).contiguous()
    U = U * torch.rsqrt(torch.mean(U * U, dim=-1, keepdim=True) + layer.eps)
    mu = (torch.nn.functional.softplus(layer.theta_mu) + layer.eps).view(1, G, 1, 1)
    gamma = (layer.gamma_max * torch.sigmoid(layer.theta_gamma)).view(1, G, 1, 1)
    expected = lsso(U, C, mu, gamma, eye=layer._eye)
    expected = expected.transpose(1, 2).contiguous().view(B, N, D)
    expected = layer.w_o(expected)

    torch.testing.assert_close(layer(x), expected, atol=1e-6, rtol=1e-6)


def test_grouped_rrlsso_shift_invariance_and_backward() -> None:
    torch.manual_seed(3)
    B, N, D, H, G, r = 2, 17, 64, 4, 2, 8
    x = torch.randn(B, N, D, requires_grad=True)
    positions = torch.arange(N)
    layer = GroupedRRLSSO(
        dim=D,
        num_heads=H,
        num_relation_groups=G,
        rank=r,
    )
    layer.record_diagnostics = True

    y = layer(x, position_ids=positions)
    shifted = layer(x, position_ids=positions + 23)
    torch.testing.assert_close(shifted, y, atol=2e-6, rtol=2e-6)
    y.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert layer.last_diagnostics is not None
    assert layer.last_diagnostics.effective_rank.shape[:2] == (B, G)


def test_grouped_relation_reduces_parameters_and_masks_outputs() -> None:
    torch.manual_seed(4)
    D, H, G, r = 96, 6, 2, 16
    per_head = LSSO(dim=D, num_heads=H, rank=r)
    grouped = GroupedLSSO(
        dim=D,
        num_heads=H,
        num_relation_groups=G,
        rank=r,
    )
    assert grouped.solve_reduction == H / G
    assert grouped.w_uc.out_features == D + G * r
    assert grouped.w_uc.weight.numel() < per_head.w_uc.weight.numel()

    x = torch.randn(2, 7, D)
    valid_mask = torch.tensor(
        [[True, True, True, True, False, False, False], [True] * 7]
    )
    y = grouped(x, valid_mask=valid_mask)
    torch.testing.assert_close(y[0, 4:], torch.zeros_like(y[0, 4:]))


def test_grouped_rrlsso_rotates_one_basis_per_group() -> None:
    torch.manual_seed(5)
    B, N, D, H, G, r = 1, 8, 32, 4, 2, 8
    x = torch.randn(B, N, D)
    layer = GroupedRRLSSO(
        dim=D,
        num_heads=H,
        num_relation_groups=G,
        rank=r,
    )
    UC = layer.w_uc(x)
    U, _C = UC.split((G * r, D), dim=-1)
    U = U.view(B, N, G, r).transpose(1, 2).contiguous()
    U = U * torch.rsqrt(torch.mean(U * U, dim=-1, keepdim=True) + layer.eps)
    rotated = layer._prepare_relation_basis(U, None)
    torch.testing.assert_close(rotated, apply_rank_rotary(U))
    assert rotated.shape == (B, G, N, r)


def test_group_validation() -> None:
    for groups in (0, 3, 5):
        try:
            GroupedLSSO(dim=64, num_heads=4, num_relation_groups=groups, rank=8)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected invalid relation group count {groups} to fail")


if __name__ == "__main__":
    test_grouped_lsso_matches_per_head_lsso_when_groups_equal_heads()
    test_grouped_rrlsso_matches_per_head_rrlsso_when_groups_equal_heads()
    test_grouped_lsso_matches_manual_grouped_solve()
    test_grouped_rrlsso_shift_invariance_and_backward()
    test_grouped_relation_reduces_parameters_and_masks_outputs()
    test_grouped_rrlsso_rotates_one_basis_per_group()
    test_group_validation()
    print("grouped LSSO tests passed")
