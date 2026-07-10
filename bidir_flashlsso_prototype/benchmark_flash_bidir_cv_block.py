from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bidir_lsso_flash_layer import BidirLSSOFlashPrototype, BidirRRLSSOFlashPrototype
from lsso import LSSO, RRLSSO


class CVMixerBlock(nn.Module):
    def __init__(self, mixer: nn.Module, dim: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = mixer
        self.norm2 = nn.LayerNorm(dim)
        hidden = dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        try:
            h = self.mixer(h, position_ids=position_ids)
        except TypeError:
            h = self.mixer(h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


def _time_ms(fn, *, warmup: int = 5, iters: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def _make_pair(kind: str, *, dim: int, heads: int, rank: int) -> tuple[nn.Module, nn.Module]:
    if kind == "lsso":
        ref_mixer = LSSO(dim=dim, num_heads=heads, rank=rank, causal=False, bias=True)
        flash_mixer = BidirLSSOFlashPrototype(dim=dim, num_heads=heads, rank=rank, bias=True)
    elif kind == "rrlsso":
        ref_mixer = RRLSSO(dim=dim, num_heads=heads, rank=rank, causal=False, bias=True)
        flash_mixer = BidirRRLSSOFlashPrototype(dim=dim, num_heads=heads, rank=rank, bias=True)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    ref = CVMixerBlock(ref_mixer, dim=dim)
    flash = CVMixerBlock(flash_mixer, dim=dim)
    flash.load_state_dict(ref.state_dict(), strict=False)
    flash.norm1.load_state_dict(copy.deepcopy(ref.norm1.state_dict()))
    flash.norm2.load_state_dict(copy.deepcopy(ref.norm2.state_dict()))
    flash.mlp.load_state_dict(copy.deepcopy(ref.mlp.state_dict()))
    return ref.cuda().train(), flash.cuda().train()


def bench_case(kind: str, *, B: int, N: int, D: int, H: int, R: int, dtype: torch.dtype) -> None:
    torch.manual_seed(24000 + B + N + D + H + R + (0 if kind == "lsso" else 1000))
    ref, flash = _make_pair(kind, dim=D, heads=H, rank=R)
    x0 = torch.randn(B, N, D, device="cuda", dtype=torch.float32)
    pos = torch.arange(N, device="cuda")

    with torch.autocast(device_type="cuda", dtype=dtype, enabled=(dtype != torch.float32)):
        y_ref = ref(x0, position_ids=pos)
        y = flash(x0, position_ids=pos)
    max_err = (y.float() - y_ref.float()).abs().max().item()

    def ref_step():
        ref.zero_grad(set_to_none=True)
        x = x0.detach().clone().requires_grad_(True)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=(dtype != torch.float32)):
            loss = ref(x, position_ids=pos).float().square().mean()
        loss.backward()

    def flash_step():
        flash.zero_grad(set_to_none=True)
        x = x0.detach().clone().requires_grad_(True)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=(dtype != torch.float32)):
            loss = flash(x, position_ids=pos).float().square().mean()
        loss.backward()

    ref_ms = _time_ms(ref_step)
    flash_ms = _time_ms(flash_step)
    speedup = ref_ms / flash_ms if flash_ms > 0 else float("inf")
    print(
        f"{kind:6s} B={B} N={N:5d} D={D:4d} H={H:2d} R={R:2d} dtype={str(dtype).split('.')[-1]:>8} "
        f"torch_block={ref_ms:9.3f}ms flash_block={flash_ms:9.3f}ms speedup={speedup:5.2f}x err={max_err:.3e}"
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")
    torch.set_float32_matmul_precision("high")
    print(torch.cuda.get_device_name())
    for dtype in (torch.bfloat16,):
        for kind in ("lsso", "rrlsso"):
            for N in (196, 784, 3136):
                bench_case(kind, B=2, N=N, D=512, H=8, R=32, dtype=dtype)


if __name__ == "__main__":
    main()
