from __future__ import annotations

import sys
import math
import copy
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lsso import RRLSSO
from lsso.modules import lsso
from lsso.modules_v2 import apply_rank_rotary


def fp32_stats_lsso(U: torch.Tensor, C: torch.Tensor, mu: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast(device_type=U.device.type, enabled=False):
        B, H, _N, r = U.shape
        dh = C.shape[-1]
        U32 = U.float()
        C32 = C.float()
        mu32 = mu.view(1, H, 1, 1).float()
        gamma32 = gamma.view(1, H, 1, 1).float()
        inv_mu = mu32.reciprocal()
        alpha = gamma32 * inv_mu
        beta = alpha * inv_mu
        S = torch.einsum("bhnr,bhns->bhrs", U32, U32)
        P = torch.einsum("bhnr,bhnd->bhrd", U32, C32)
        eye = torch.eye(r, device=U.device, dtype=torch.float32).view(1, 1, r, r)
        G = eye + alpha * S
        K = torch.linalg.solve_ex(G.reshape(B * H, r, r), P.reshape(B * H, r, dh), check_errors=False)[0]
        K = K.view(B, H, r, dh)
        Y = inv_mu * C32 - beta * torch.einsum("bhnr,bhrd->bhnd", U32, K)
        return Y


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach()
    b = b.detach()
    return float((a - b).float().norm().cpu() / b.float().norm().clamp_min(1e-12).cpu())


def _max(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach() - b.detach()).float().abs().max().cpu())


def check_case(*, B: int, H: int, N: int, R: int, DH: int, gamma_value: float) -> None:
    torch.manual_seed(26000 + B + H + N + R + DH + int(gamma_value * 10000))
    U0 = torch.randn(B, H, N, R, device="cuda", dtype=torch.float32)
    U0 = U0 * torch.rsqrt(torch.mean(U0 * U0, dim=-1, keepdim=True) + 1e-5)
    C0 = torch.randn(B, H, N, DH, device="cuda", dtype=torch.float32)
    U = U0.to(torch.bfloat16).detach().requires_grad_(True)
    C = C0.to(torch.bfloat16).detach().requires_grad_(True)
    mu = torch.full((H,), 0.6931572, device="cuda", dtype=torch.float32, requires_grad=True)
    gamma = torch.full((H,), gamma_value, device="cuda", dtype=torch.float32, requires_grad=True)
    grad = torch.randn(B, H, N, DH, device="cuda", dtype=torch.bfloat16)

    y_current = lsso(
        U,
        C,
        mu.view(1, H, 1, 1),
        gamma.view(1, H, 1, 1),
        causal=False,
        length_normalize=False,
    )
    y_fp32 = fp32_stats_lsso(U, C, mu, gamma)
    y_fp32_cast = y_fp32.to(y_current.dtype)

    grads_current = torch.autograd.grad(
        y_current,
        (U, C, mu, gamma),
        grad,
        retain_graph=True,
        allow_unused=False,
    )
    grads_fp32 = torch.autograd.grad(
        y_fp32,
        (U, C, mu, gamma),
        grad.float(),
        allow_unused=False,
    )

    print(
        f"N={N:5d} gamma={gamma_value:.6f} y_dtype={str(y_current.dtype):>12} "
        f"y_rel_vs_fp32={_rel(y_current, y_fp32):.3e} "
        f"y_rel_vs_fp32_cast={_rel(y_current, y_fp32_cast):.3e} "
        f"y_max={_max(y_current, y_fp32):.3e} "
        f"dU_rel={_rel(grads_current[0], grads_fp32[0]):.3e} "
        f"dC_rel={_rel(grads_current[1], grads_fp32[1]):.3e} "
        f"dmu_rel={_rel(grads_current[2], grads_fp32[2]):.3e} "
        f"dgamma_rel={_rel(grads_current[3], grads_fp32[3]):.3e}"
    )


def _set_gamma(layer: RRLSSO, gamma_value: float) -> None:
    ratio = min(max(gamma_value / float(layer.gamma_max), 1e-6), 1.0 - 1e-6)
    theta = math.log(ratio / (1.0 - ratio))
    with torch.no_grad():
        layer.theta_gamma.fill_(theta)


def rrlsso_forward_fp32_stats(layer: RRLSSO, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    B, N, D = x.shape
    H = layer.num_heads
    r = layer.rank
    dh = layer.head_dim
    UC = layer.w_uc(x)
    U, C = UC.split((H * r, D), dim=-1)
    U = U.view(B, N, H, r).transpose(1, 2).contiguous()
    C = C.view(B, N, H, dh).transpose(1, 2).contiguous()
    if layer.normalize_u:
        U = U * torch.rsqrt(torch.mean(U * U, dim=-1, keepdim=True) + layer.eps)
    U = apply_rank_rotary(U, position_ids, base=layer.rope_base, scale=layer.rope_scale)
    mu = torch.nn.functional.softplus(layer.theta_mu) + layer.eps
    gamma = layer.gamma_max * torch.sigmoid(layer.theta_gamma)
    Y = fp32_stats_lsso(U, C, mu, gamma).to(C.dtype)
    Y = Y.transpose(1, 2).contiguous().view(B, N, D)
    return layer.w_o(Y)


def check_layer_case(*, B: int, N: int, D: int, H: int, R: int, gamma_value: float) -> None:
    torch.manual_seed(27000 + B + N + D + H + R + int(gamma_value * 10000))
    x0 = torch.randn(B, N, D, device="cuda", dtype=torch.float32)
    grad = torch.randn(B, N, D, device="cuda", dtype=torch.float32)
    pos = torch.arange(N, device="cuda")
    current = RRLSSO(dim=D, num_heads=H, rank=R, causal=False, bias=True).cuda()
    _set_gamma(current, gamma_value)
    forced = copy.deepcopy(current).cuda()
    x_current = x0.detach().clone().requires_grad_(True)
    x_forced = x0.detach().clone().requires_grad_(True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y_current = current(x_current, position_ids=pos)
        y_forced = rrlsso_forward_fp32_stats(forced, x_forced, pos)
    (y_current.float() * grad).mean().backward()
    (y_forced.float() * grad).mean().backward()

    grad_items: list[tuple[str, torch.Tensor, torch.Tensor]] = [("x", x_current.grad, x_forced.grad)]
    for (name, p), (name_f, p_f) in zip(current.named_parameters(), forced.named_parameters()):
        if name != name_f:
            raise AssertionError(f"parameter mismatch {name} vs {name_f}")
        if p.grad is not None and p_f.grad is not None:
            grad_items.append((name, p.grad, p_f.grad))

    selected = {
        "x": None,
        "theta_gamma": None,
        "w_uc.weight": None,
        "w_o.weight": None,
    }
    for name, a, b in grad_items:
        if name in selected:
            selected[name] = _rel(a, b)
    print(
        f"LAYER N={N:5d} gamma={gamma_value:.6f} "
        f"y_rel={_rel(y_current, y_forced):.3e} y_max={_max(y_current, y_forced):.3e} "
        f"x_grad={selected['x']:.3e} "
        f"theta_gamma_grad={selected['theta_gamma']:.3e} "
        f"w_uc_grad={selected['w_uc.weight']:.3e} "
        f"w_o_grad={selected['w_o.weight']:.3e}"
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")
    print(torch.cuda.get_device_name())
    for gamma_value in (0.005395863, 0.0357609, 0.0806824):
        for N in (196, 784, 3136, 8192):
            check_case(B=2, H=8, N=N, R=32, DH=64, gamma_value=gamma_value)
    print("\nLayer-level RRLSSO autocast comparison")
    for gamma_value in (0.005395863, 0.0357609):
        for N in (196, 784, 3136):
            check_layer_case(B=2, N=N, D=512, H=8, R=32, gamma_value=gamma_value)


if __name__ == "__main__":
    main()
