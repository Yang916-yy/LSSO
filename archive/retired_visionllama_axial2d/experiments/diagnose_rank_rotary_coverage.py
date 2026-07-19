from __future__ import annotations

import argparse

import torch

from lsso.rotary_2d import build_2d_rotary_factors


def _deviation(cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((cos.float() - 1.0).square() + sin.float().square())


def _summarize(name: str, cos: torch.Tensor, sin: torch.Tensor) -> None:
    deviation = _deviation(cos, sin).reshape(-1, cos.shape[-1])
    exact = (deviation < 1e-7).float().mean()
    near_001 = (deviation < 0.01).float().mean()
    near_01 = (deviation < 0.1).float().mean()
    globally_near = (deviation.max(dim=0).values < 0.1).sum()
    print(
        f"{name}: exact={100 * exact:.2f}% near<0.01={100 * near_001:.2f}% "
        f"near<0.1={100 * near_01:.2f}% globally_near_pairs="
        f"{int(globally_near)}/{deviation.shape[-1]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--grids", nargs="+", type=int, default=[8, 14, 32])
    parser.add_argument("--base", type=float, default=10000.0)
    args = parser.parse_args()

    for grid in args.grids:
        cos_2d, sin_2d = build_2d_rotary_factors(
            args.rank,
            spatial_shape=(grid, grid),
            num_prefix_tokens=1,
            base=args.base,
        )
        _summarize(f"2d grid={grid}x{grid} + zero-prefix", cos_2d, sin_2d)

        length = grid * grid + 1
        half = args.rank // 2
        inv_freq = args.base ** (-torch.arange(half, dtype=torch.float32) / half)
        angles = torch.arange(length, dtype=torch.float32)[:, None] * inv_freq[None]
        _summarize(
            f"1d length={length}",
            angles.cos().view(1, 1, length, half),
            angles.sin().view(1, 1, length, half),
        )


if __name__ == "__main__":
    main()
