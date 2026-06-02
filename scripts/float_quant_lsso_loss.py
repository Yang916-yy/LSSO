from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.models.vit import VisionEncoder
from lsso import lsso
from train_cifar import build_loaders


@dataclass
class FloatQuantResult:
    source: str
    format: str
    scheme: str
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
    finite: bool


def fp8_cast(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return x.to(dtype).to(torch.float32)


def nvfp4_fake_quant(x: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    # E2M1 representable magnitudes used by NVIDIA docs: 0, .5, 1, 1.5, 2, 3, 4, 6.
    code = torch.tensor(
        [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=x.device,
        dtype=torch.float32,
    )
    xf = x.float()
    orig_shape = xf.shape
    last = orig_shape[-1]
    pad = (-last) % block_size
    if pad:
        xf = F.pad(xf, (0, pad))
    blocks = xf.reshape(-1, block_size)

    scale = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 6.0
    scale = fp8_cast(scale, torch.float8_e4m3fn).clamp_min(1e-8)
    normalized = blocks / scale
    idx = (normalized.unsqueeze(-1) - code.view(1, 1, -1)).abs().argmin(dim=-1)
    q = code[idx] * scale
    q = q.reshape(*orig_shape[:-1], last + pad)
    if pad:
        q = q[..., :last]
    return q


def quantize_tensor(x: torch.Tensor, fmt: str) -> torch.Tensor:
    if fmt == "fp8_e4m3":
        return fp8_cast(x, torch.float8_e4m3fn)
    if fmt == "fp8_e5m2":
        return fp8_cast(x, torch.float8_e5m2)
    if fmt == "nvfp4":
        return nvfp4_fake_quant(x)
    raise ValueError(f"unknown format: {fmt}")


def lsso_with_quantized_system(
    U: torch.Tensor,
    C: torch.Tensor,
    mu: torch.Tensor,
    gamma: torch.Tensor,
    *,
    fmt: str,
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

    G = quantize_tensor(G, fmt)
    UtC = quantize_tensor(UtC, fmt)
    K = torch.linalg.solve_ex(
        G.view(B * H, r, r),
        UtC.float().view(B * H, r, dh),
        check_errors=False,
    ).result.to(U.dtype)
    UK = torch.bmm(U_bh, K).view(B, H, N, dh)
    return local - gamma_over_mu2 * UK


def quantized_lsso(
    U: torch.Tensor,
    C: torch.Tensor,
    mu: torch.Tensor,
    gamma: torch.Tensor,
    *,
    fmt: str,
    scheme: str,
    eye: torch.Tensor,
) -> torch.Tensor:
    if scheme == "U":
        return lsso(quantize_tensor(U, fmt), C, mu, gamma, eye=eye)
    if scheme == "C":
        return lsso(U, quantize_tensor(C, fmt), mu, gamma, eye=eye)
    if scheme == "U+C":
        return lsso(quantize_tensor(U, fmt), quantize_tensor(C, fmt), mu, gamma, eye=eye)
    if scheme == "U+C+Y":
        y = lsso(quantize_tensor(U, fmt), quantize_tensor(C, fmt), mu, gamma, eye=eye)
        return quantize_tensor(y, fmt)
    if scheme == "G+UtC":
        return lsso_with_quantized_system(U, C, mu, gamma, fmt=fmt, eye=eye)
    raise ValueError(f"unknown scheme: {scheme}")


def metrics(y: torch.Tensor, yq: torch.Tensor) -> tuple[float, float, float, float, float, float, bool]:
    finite = torch.isfinite(yq).all().item()
    yf = y.float().flatten(1)
    yqf = yq.float().flatten(1)
    diff = yqf - yf
    rel_l2 = diff.norm() / yf.norm().clamp_min(1e-8)
    cos = F.cosine_similarity(yf, yqf, dim=1).mean()
    return (
        rel_l2.item(),
        diff.abs().max().item(),
        (1.0 - cos).item(),
        diff.square().mean().item(),
        yf.square().mean().sqrt().item(),
        yqf.square().mean().sqrt().item(),
        bool(finite),
    )


def make_random_case(args: argparse.Namespace) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, N, D, r = args.batch_size, args.num_heads, args.seq_len, args.dim, args.rank
    dh = D // H
    U = torch.randn(B, H, N, r, device=args.device)
    U = U * torch.rsqrt(torch.mean(U * U, dim=-1, keepdim=True) + 1e-5)
    C = torch.randn(B, H, N, dh, device=args.device)
    mu = (F.softplus(torch.zeros(H, device=args.device)) + 1e-5).view(1, H, 1, 1)
    gamma = (args.gamma_max * torch.sigmoid(torch.full((H,), args.theta_gamma_init, device=args.device))).view(1, H, 1, 1)
    return "random", U, C, mu, gamma


def make_cifar_case(args: argparse.Namespace) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_args = checkpoint["args"]
    loader_args = argparse.Namespace(
        dataset="cifar10",
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True,
        image_size=32,
    )
    train_loader, _, num_classes = build_loaders(loader_args)
    model = VisionEncoder(
        image_size=32,
        patch_size=4,
        num_classes=num_classes,
        dim=model_args["dim"],
        depth=model_args["depth"],
        num_heads=model_args["num_heads"],
        mixer=model_args["mixer"],
        rank=model_args["rank"],
        mlp_ratio=model_args["mlp_ratio"],
        dropout=model_args["dropout"],
        gamma_max=model_args["gamma_max"],
        theta_gamma_init=model_args["theta_gamma_init"],
    ).to(args.device).eval()
    model.load_state_dict(checkpoint["model"])
    images, _ = next(iter(train_loader))
    images = images.to(args.device, non_blocking=True)
    with torch.no_grad(), torch.amp.autocast(device_type=args.device, enabled=args.device == "cuda"):
        x = model.patch_embed(images).flatten(2).transpose(1, 2)
        cls = model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = model.pos_drop(x + model.pos_embed)
        block = model.blocks[args.layer]
        z = block.norm1(x)
        layer = block.mixer
        B, N, D = z.shape
        H, r, dh = layer.num_heads, layer.rank, layer.head_dim
        UC = layer.w_uc(z)
        U, C = UC.split((H * r, D), dim=-1)
        U = U.view(B, N, H, r).transpose(1, 2).contiguous()
        C = C.view(B, N, H, dh).transpose(1, 2).contiguous()
        U = U * torch.rsqrt(torch.mean(U * U, dim=-1, keepdim=True) + layer.eps)
        mu = (F.softplus(layer.theta_mu) + layer.eps).view(1, H, 1, 1)
        gamma = (layer.gamma_max * torch.sigmoid(layer.theta_gamma)).view(1, H, 1, 1)
    return f"cifar10_layer{args.layer}", U.float(), C.float(), mu.float(), gamma.float()


def run_case(
    source: str,
    U: torch.Tensor,
    C: torch.Tensor,
    mu: torch.Tensor,
    gamma: torch.Tensor,
    args: argparse.Namespace,
) -> list[FloatQuantResult]:
    B, H, N, r = U.shape
    D = C.shape[-1] * H
    eye = torch.eye(r, device=args.device).view(1, 1, r, r)
    rows = []
    with torch.no_grad():
        y = lsso(U, C, mu, gamma, eye=eye)
        for fmt in args.formats:
            for scheme in args.schemes:
                yq = quantized_lsso(U, C, mu, gamma, fmt=fmt, scheme=scheme, eye=eye)
                rel_l2, max_abs, cos_error, mse, y_rms, yq_rms, finite = metrics(y, yq)
                rows.append(
                    FloatQuantResult(
                        source=source,
                        format=fmt,
                        scheme=scheme,
                        batch_size=B,
                        seq_len=N,
                        dim=D,
                        num_heads=H,
                        rank=r,
                        rel_l2=rel_l2,
                        max_abs=max_abs,
                        cos_error=cos_error,
                        mse=mse,
                        y_rms=y_rms,
                        yq_rms=yq_rms,
                        finite=finite,
                    )
                )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["random", "cifar"], default="random")
    parser.add_argument("--checkpoint", default="runs/20260526-234735_cifar10_lsso_r32_g0.3_tgi-4.0_d96_L3_h3_s1.pt")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--gamma-max", type=float, default=0.3)
    parser.add_argument("--theta-gamma-init", type=float, default=-4.0)
    parser.add_argument("--formats", nargs="+", default=["fp8_e4m3", "fp8_e5m2", "nvfp4"])
    parser.add_argument("--schemes", nargs="+", default=["U", "C", "U+C", "U+C+Y", "G+UtC"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    if args.source == "random":
        source, U, C, mu, gamma = make_random_case(args)
    else:
        source, U, C, mu, gamma = make_cifar_case(args)

    rows = run_case(source, U, C, mu, gamma, args)
    for row in rows:
        print(json.dumps(asdict(row), sort_keys=True), flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(asdict(row), sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
