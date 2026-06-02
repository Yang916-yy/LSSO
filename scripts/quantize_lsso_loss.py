from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lsso import lsso


@dataclass
class QuantResult:
    source: str
    scheme: str
    bits: int
    per: str
    batch_size: int
    seq_len: int
    dim: int
    num_heads: int
    rank: int
    rel_l2: float
    max_abs: float
    cos_error: float
    mse: float
    y_rms: float
    yq_rms: float


def symmetric_fake_quant(x: torch.Tensor, bits: int = 8, per: str = "tensor") -> torch.Tensor:
    if bits >= 16:
        return x
    qmax = (1 << (bits - 1)) - 1
    if per == "tensor":
        scale = x.detach().abs().amax().clamp_min(1e-8) / qmax
    elif per == "lastdim":
        scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    elif per == "token":
        scale = x.detach().abs().amax(dim=(-1,), keepdim=True).clamp_min(1e-8) / qmax
    elif per == "head":
        scale = x.detach().abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8) / qmax
    else:
        raise ValueError(f"unknown quant granularity: {per}")
    q = torch.round(x / scale).clamp(-qmax, qmax)
    return q * scale


def quantized_lsso(
    U: torch.Tensor,
    C: torch.Tensor,
    mu: torch.Tensor,
    gamma: torch.Tensor,
    *,
    bits: int,
    per: str,
    scheme: str,
    eye: torch.Tensor,
) -> torch.Tensor:
    if scheme == "U":
        return lsso(symmetric_fake_quant(U, bits, per), C, mu, gamma, eye=eye)
    if scheme == "C":
        return lsso(U, symmetric_fake_quant(C, bits, per), mu, gamma, eye=eye)
    if scheme == "U+C":
        return lsso(
            symmetric_fake_quant(U, bits, per),
            symmetric_fake_quant(C, bits, per),
            mu,
            gamma,
            eye=eye,
        )
    if scheme == "U+C+Y":
        y = lsso(
            symmetric_fake_quant(U, bits, per),
            symmetric_fake_quant(C, bits, per),
            mu,
            gamma,
            eye=eye,
        )
        return symmetric_fake_quant(y, bits, per)
    if scheme == "G+UtC":
        return lsso_with_quantized_system(U, C, mu, gamma, bits=bits, per=per, eye=eye)
    raise ValueError(f"unknown scheme: {scheme}")


def lsso_with_quantized_system(
    U: torch.Tensor,
    C: torch.Tensor,
    mu: torch.Tensor,
    gamma: torch.Tensor,
    *,
    bits: int,
    per: str,
    eye: torch.Tensor,
) -> torch.Tensor:
    B, H, N, r = U.shape
    dh = C.shape[-1]
    inv_mu = mu.reciprocal()
    gamma_over_mu = gamma * inv_mu
    gamma_over_mu2 = gamma_over_mu * inv_mu
    local = inv_mu * C

    U_bh = U.flatten(0, 1)
    C_bh = C.flatten(0, 1)
    Ut_bh = U_bh.transpose(1, 2)
    UtU = torch.bmm(Ut_bh, U_bh).view(B, H, r, r)
    UtC = torch.bmm(Ut_bh, C_bh).view(B, H, r, dh)
    G = eye.float() + gamma_over_mu.float() * UtU.float()

    G = symmetric_fake_quant(G, bits, per)
    UtC = symmetric_fake_quant(UtC, bits, per)

    K = torch.linalg.solve_ex(
        G.view(B * H, r, r),
        UtC.float().view(B * H, r, dh),
        check_errors=False,
    ).result.to(U.dtype)
    UK = torch.bmm(U_bh, K).view(B, H, N, dh)
    return local - gamma_over_mu2 * UK


def metrics(y: torch.Tensor, yq: torch.Tensor) -> tuple[float, float, float, float, float, float]:
    yf = y.float().flatten(1)
    yqf = yq.float().flatten(1)
    diff = yqf - yf
    rel_l2 = diff.norm() / yf.norm().clamp_min(1e-8)
    max_abs = diff.abs().max()
    cos = F.cosine_similarity(yf, yqf, dim=1).mean()
    mse = diff.square().mean()
    return (
        rel_l2.item(),
        max_abs.item(),
        (1.0 - cos).item(),
        mse.item(),
        yf.square().mean().sqrt().item(),
        yqf.square().mean().sqrt().item(),
    )


def make_random_case(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, N, D, r = args.batch_size, args.num_heads, args.seq_len, args.dim, args.rank
    dh = D // H
    U = torch.randn(B, H, N, r, device=args.device, dtype=torch.float32)
    U = U * torch.rsqrt(torch.mean(U * U, dim=-1, keepdim=True) + 1e-5)
    C = torch.randn(B, H, N, dh, device=args.device, dtype=torch.float32)
    mu = (F.softplus(torch.zeros(H, device=args.device)) + 1e-5).view(1, H, 1, 1)
    gamma = (args.gamma_max * torch.sigmoid(torch.full((H,), args.theta_gamma_init, device=args.device))).view(1, H, 1, 1)
    return U, C, mu, gamma


def run_case(
    args: argparse.Namespace,
    source: str,
    U: torch.Tensor,
    C: torch.Tensor,
    mu: torch.Tensor,
    gamma: torch.Tensor,
) -> list[QuantResult]:
    eye = torch.eye(args.rank, device=args.device, dtype=torch.float32).view(1, 1, args.rank, args.rank)
    with torch.no_grad():
        y = lsso(U, C, mu, gamma, eye=eye)
        results = []
        for bits in args.bits:
            for per in args.per:
                for scheme in args.schemes:
                    yq = quantized_lsso(U, C, mu, gamma, bits=bits, per=per, scheme=scheme, eye=eye)
                    rel_l2, max_abs, cos_error, mse, y_rms, yq_rms = metrics(y, yq)
                    results.append(
                        QuantResult(
                            source=source,
                            scheme=scheme,
                            bits=bits,
                            per=per,
                            batch_size=args.batch_size,
                            seq_len=args.seq_len,
                            dim=args.dim,
                            num_heads=args.num_heads,
                            rank=args.rank,
                            rel_l2=rel_l2,
                            max_abs=max_abs,
                            cos_error=cos_error,
                            mse=mse,
                            y_rms=y_rms,
                            yq_rms=yq_rms,
                        )
                    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["random"], default="random")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--gamma-max", type=float, default=0.3)
    parser.add_argument("--theta-gamma-init", type=float, default=-4.0)
    parser.add_argument("--bits", nargs="+", type=int, default=[8, 6, 4])
    parser.add_argument("--per", nargs="+", default=["tensor", "head", "lastdim"])
    parser.add_argument("--schemes", nargs="+", default=["U", "C", "U+C", "U+C+Y", "G+UtC"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dim % args.num_heads != 0:
        raise ValueError("dim must be divisible by num_heads")
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    U, C, mu, gamma = make_random_case(args)
    rows = run_case(args, args.source, U, C, mu, gamma)
    for row in rows:
        print(json.dumps(asdict(row), sort_keys=True), flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(asdict(row), sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
