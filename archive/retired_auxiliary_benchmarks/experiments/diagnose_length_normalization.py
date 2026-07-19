"""Archived diagnostic for the superseded RMS/length-normalized solve."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from lsso.modules import length_normalize_basis, lsso


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure global-correction scale while repeating a fixed token sequence."
    )
    parser.add_argument("--base-length", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--gamma-max", type=float, default=1.2)
    parser.add_argument("--theta-gamma", type=float, default=0.5)
    args = parser.parse_args()

    if args.max_length < args.base_length or args.max_length % args.base_length:
        raise ValueError("max-length must be an integer multiple of base-length")

    torch.manual_seed(args.seed)
    u0 = torch.randn(1, args.heads, args.base_length, args.rank, dtype=torch.float64)
    c0 = torch.randn(1, args.heads, args.base_length, args.head_dim, dtype=torch.float64)
    u0 = u0 * torch.rsqrt(torch.mean(u0 * u0, dim=-1, keepdim=True) + 1e-5)

    mu = F.softplus(torch.zeros(args.heads, dtype=torch.float64)) + 1e-5
    gamma = args.gamma_max * torch.sigmoid(
        torch.full((args.heads,), args.theta_gamma, dtype=torch.float64)
    )

    print(
        "length\tlegacy_correction_ratio\tnormalized_correction_ratio\t"
        "legacy_max_eigenvalue\tnormalized_max_eigenvalue"
    )
    length = args.base_length
    while length <= args.max_length:
        repeats = length // args.base_length
        u = u0.repeat(1, 1, repeats, 1)
        c = c0.repeat(1, 1, repeats, 1)
        row: list[float] = []
        for normalize in (False, True):
            _, aux = lsso(
                u,
                c,
                mu,
                gamma,
                length_normalize=normalize,
                return_aux=True,
            )
            correction_ratio = (
                torch.linalg.vector_norm(aux.correction, dim=(-2, -1))
                / torch.linalg.vector_norm(aux.local, dim=(-2, -1)).clamp_min(1e-12)
            ).mean()
            basis = length_normalize_basis(u) if normalize else u
            covariance = basis.transpose(-2, -1) @ basis
            max_eigenvalue = torch.linalg.eigvalsh(covariance).amax(dim=-1).mean()
            row.extend((correction_ratio.item(), max_eigenvalue.item()))
        print(f"{length}\t{row[0]:.8f}\t{row[2]:.8f}\t{row[1]:.8f}\t{row[3]:.8f}")
        length *= 2


if __name__ == "__main__":
    main()
