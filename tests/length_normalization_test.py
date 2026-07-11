from __future__ import annotations

import torch
import torch.nn.functional as F

from lsso import GroupedRRLSSO, LSSO, RRLSSO
from lsso.modules import length_normalize_basis, lsso


def _inputs(sequence: int = 7):
    torch.manual_seed(42000 + sequence)
    u = torch.randn(2, 3, sequence, 4, dtype=torch.float64)
    c = torch.randn(2, 3, sequence, 5, dtype=torch.float64)
    mu = torch.rand(1, 3, 1, 1, dtype=torch.float64) + 0.5
    gamma = 0.2 * torch.rand(1, 3, 1, 1, dtype=torch.float64)
    return u, c, mu, gamma


def test_repeating_tokens_does_not_change_global_correction() -> None:
    u, c, mu, gamma = _inputs()
    repeats = 4

    y = lsso(u, c, mu, gamma, length_normalize=True)
    y_repeated = lsso(
        u.repeat_interleave(repeats, dim=2),
        c.repeat_interleave(repeats, dim=2),
        mu,
        gamma,
        length_normalize=True,
    )

    torch.testing.assert_close(
        y_repeated,
        y.repeat_interleave(repeats, dim=2),
        rtol=1e-10,
        atol=1e-10,
    )

    legacy = lsso(u, c, mu, gamma, length_normalize=False)
    legacy_repeated = lsso(
        u.repeat_interleave(repeats, dim=2),
        c.repeat_interleave(repeats, dim=2),
        mu,
        gamma,
        length_normalize=False,
    )
    assert (
        legacy_repeated - legacy.repeat_interleave(repeats, dim=2)
    ).abs().max().item() > 1e-4


def test_padding_uses_each_samples_effective_length() -> None:
    u, c, mu, gamma = _inputs(sequence=9)
    pad = 6
    padded_u = torch.cat((u, torch.randn(*u.shape[:2], pad, u.shape[-1], dtype=u.dtype)), dim=2)
    padded_c = torch.cat((c, torch.randn(*c.shape[:2], pad, c.shape[-1], dtype=c.dtype)), dim=2)
    valid_mask = torch.zeros(u.shape[0], u.shape[2] + pad, dtype=torch.bool)
    valid_mask[:, : u.shape[2]] = True

    expected = lsso(u, c, mu, gamma, length_normalize=True)
    actual = lsso(
        padded_u,
        padded_c,
        mu,
        gamma,
        length_normalize=True,
        valid_mask=valid_mask,
    )

    torch.testing.assert_close(actual[:, :, : u.shape[2]], expected, rtol=1e-10, atol=1e-10)
    assert torch.count_nonzero(actual[:, :, u.shape[2] :]).item() == 0


def test_reference_length_preserves_legacy_anchor() -> None:
    u, c, mu, gamma = _inputs(sequence=11)
    legacy = lsso(u, c, mu, gamma, length_normalize=False)
    anchored = lsso(
        u,
        c,
        mu,
        gamma,
        length_normalize=True,
        length_reference=u.shape[2],
    )
    torch.testing.assert_close(anchored, legacy, rtol=1e-12, atol=1e-12)


def test_basis_helper_scales_from_effective_lengths() -> None:
    u = torch.ones(2, 3, 5, 4)
    mask = torch.tensor(
        [[True, True, True, True, True], [True, True, False, False, False]]
    )
    actual = length_normalize_basis(u, mask, reference_length=8.0)
    torch.testing.assert_close(actual[0], u[0] * (8.0 / 5.0) ** 0.5)
    torch.testing.assert_close(actual[1], u[1] * (8.0 / 2.0) ** 0.5)


def test_bidirectional_defaults_start_in_retuned_strength_interval() -> None:
    layers = (
        LSSO(dim=32, num_heads=4, rank=8),
        RRLSSO(dim=32, num_heads=4, rank=8),
        GroupedRRLSSO(dim=32, num_heads=4, num_relation_groups=2, rank=8),
    )
    for layer in layers:
        mu = F.softplus(layer.theta_mu) + layer.eps
        gamma = layer.gamma_max * torch.sigmoid(layer.theta_gamma)
        strength = gamma / mu
        assert torch.all((strength >= 0.85) & (strength <= 1.15))
