from __future__ import annotations

from typing import Any

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - depends on optional runtime
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


def triton_available() -> bool:
    return bool(_TRITON_AVAILABLE)


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _check_inputs(U: torch.Tensor, C: torch.Tensor, mu: torch.Tensor, gamma: torch.Tensor) -> tuple[int, int, int, int, int]:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed. Install the optional Triton runtime to use causal_backend='triton'.")
    if not U.is_cuda or not C.is_cuda:
        raise RuntimeError("Triton causal LSSO requires CUDA tensors.")
    if U.dim() != 4 or C.dim() != 4:
        raise ValueError("U and C must have shapes [B, H, N, r] and [B, H, N, dh].")
    if U.shape[:3] != C.shape[:3]:
        raise ValueError(f"U and C leading dimensions must match, got {tuple(U.shape)} and {tuple(C.shape)}.")
    B, H, N, r = U.shape
    dh = C.shape[-1]
    if r not in (4, 8, 16, 32):
        raise ValueError(f"Triton causal LSSO currently supports rank in {{4, 8, 16, 32}}, got {r}.")
    if dh > 128:
        raise ValueError(f"Triton causal LSSO currently supports head_dim <= 128, got {dh}.")
    if mu.numel() not in (H, B * H):
        raise ValueError(f"mu must have H or B*H elements, got {mu.numel()} for B={B}, H={H}.")
    if gamma.numel() not in (H, B * H):
        raise ValueError(f"gamma must have H or B*H elements, got {gamma.numel()} for B={B}, H={H}.")
    return B, H, N, r, dh


if _TRITON_AVAILABLE:

    @triton.jit
    def _causal_prefix_lsso_forward_kernel(
        U,
        C,
        MU,
        GAMMA,
        Y,
        M_STATE,
        P_STATE,
        START: tl.constexpr,
        B: tl.constexpr,
        H: tl.constexpr,
        N: tl.constexpr,
        R: tl.constexpr,
        DH: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_DH: tl.constexpr,
        MU_HAS_BATCH: tl.constexpr,
        GAMMA_HAS_BATCH: tl.constexpr,
        EXCLUSIVE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        h = pid % H
        rows = tl.arange(0, R)
        cols = tl.arange(0, R)
        dcols = tl.arange(0, BLOCK_DH)

        mu_idx = tl.where(MU_HAS_BATCH, pid, h)
        gamma_idx = tl.where(GAMMA_HAS_BATCH, pid, h)
        mu = tl.load(MU + mu_idx).to(tl.float32)
        gamma = tl.load(GAMMA + gamma_idx).to(tl.float32)
        inv_mu = 1.0 / mu
        alpha = gamma * inv_mu
        gamma_over_mu2 = alpha * inv_mu

        M = tl.load(M_STATE + pid * R * R + rows[:, None] * R + cols[None, :]).to(tl.float32)
        P = tl.load(
            P_STATE + pid * R * DH + rows[:, None] * DH + dcols[None, :],
            mask=dcols[None, :] < DH,
            other=0.0,
        ).to(tl.float32)

        for t in range(BLOCK_N):
            n = START + t
            token_mask = n < N
            u = tl.load(
                U + pid * N * R + n * R + rows,
                mask=token_mask,
                other=0.0,
            ).to(tl.float32)
            c = tl.load(
                C + pid * N * DH + n * DH + dcols,
                mask=token_mask & (dcols < DH),
                other=0.0,
            ).to(tl.float32)

            v = tl.sum(M * u[None, :], axis=1)
            den = 1.0 + alpha * tl.sum(u * v, axis=0)
            den = tl.maximum(den, 1.0e-6)

            if EXCLUSIVE:
                corr = tl.sum(v[:, None] * P, axis=0)
                P += u[:, None] * c[None, :]
                M -= (alpha / den) * v[:, None] * v[None, :]
            else:
                P += u[:, None] * c[None, :]
                M -= (alpha / den) * v[:, None] * v[None, :]
                corr = tl.sum((v / den)[:, None] * P, axis=0)

            y = inv_mu * c - gamma_over_mu2 * corr
            tl.store(
                Y + pid * N * DH + n * DH + dcols,
                y,
                mask=token_mask & (dcols < DH),
            )

        tl.store(M_STATE + pid * R * R + rows[:, None] * R + cols[None, :], M)
        tl.store(
            P_STATE + pid * R * DH + rows[:, None] * DH + dcols[None, :],
            P,
            mask=dcols[None, :] < DH,
        )


def causal_prefix_lsso_triton_forward(
    U: torch.Tensor,
    C: torch.Tensor,
    mu: torch.Tensor,
    gamma: torch.Tensor,
    *,
    exclusive: bool = False,
    chunk_size: int = 256,
) -> torch.Tensor:
    B, H, N, r, dh = _check_inputs(U, C, mu, gamma)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    if not U.is_contiguous():
        U = U.contiguous()
    if not C.is_contiguous():
        C = C.contiguous()

    y = torch.empty_like(C)
    mu_flat = mu.reshape(-1).contiguous()
    gamma_flat = gamma.reshape(-1).contiguous()
    mu_has_batch = mu_flat.numel() == B * H
    gamma_has_batch = gamma_flat.numel() == B * H
    block_dh = _next_power_of_2(dh)
    if block_dh > 128:
        raise ValueError(f"head_dim={dh} is too large for the current Triton causal prototype.")

    M_state = torch.eye(r, device=U.device, dtype=torch.float32).view(1, 1, r, r).repeat(B, H, 1, 1)
    P_state = torch.zeros(B, H, r, dh, device=U.device, dtype=torch.float32)

    for start in range(0, N, chunk_size):
        _causal_prefix_lsso_forward_kernel[(B * H,)](
            U,
            C,
            mu_flat,
            gamma_flat,
            y,
            M_state,
            P_state,
            START=start,
            B=B,
            H=H,
            N=N,
            R=r,
            DH=dh,
            BLOCK_N=min(chunk_size, N - start),
            BLOCK_DH=block_dh,
            MU_HAS_BATCH=mu_has_batch,
            GAMMA_HAS_BATCH=gamma_has_batch,
            EXCLUSIVE=exclusive,
            num_warps=4 if r >= 16 else 2,
        )
    return y


class _CausalPrefixLSSOTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        U: torch.Tensor,
        C: torch.Tensor,
        mu: torch.Tensor,
        gamma: torch.Tensor,
        exclusive: bool,
        chunk_size: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(U, C, mu, gamma)
        ctx.exclusive = bool(exclusive)
        ctx.chunk_size = int(chunk_size)
        return causal_prefix_lsso_triton_forward(
            U,
            C,
            mu,
            gamma,
            exclusive=ctx.exclusive,
            chunk_size=ctx.chunk_size,
        )

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        U, C, mu, gamma = ctx.saved_tensors
        with torch.enable_grad():
            U_ = U.detach().requires_grad_(ctx.needs_input_grad[0])
            C_ = C.detach().requires_grad_(ctx.needs_input_grad[1])
            mu_ = mu.detach().requires_grad_(ctx.needs_input_grad[2])
            gamma_ = gamma.detach().requires_grad_(ctx.needs_input_grad[3])
            from .modules import _lsso_prefix_chunked_forward

            y = _lsso_prefix_chunked_forward(
                U_,
                C_,
                mu_,
                gamma_,
                exclusive=ctx.exclusive,
                chunk_size=ctx.chunk_size,
            )
            grads = torch.autograd.grad(
                y,
                (U_, C_, mu_, gamma_),
                grad_output,
                allow_unused=True,
            )
        return (*grads, None, None)


def causal_prefix_lsso_triton(
    U: torch.Tensor,
    C: torch.Tensor,
    mu: torch.Tensor,
    gamma: torch.Tensor,
    *,
    exclusive: bool = False,
    chunk_size: int = 256,
) -> torch.Tensor:
    if torch.is_grad_enabled() and (U.requires_grad or C.requires_grad or mu.requires_grad or gamma.requires_grad):
        return _CausalPrefixLSSOTriton.apply(U, C, mu, gamma, exclusive, chunk_size)
    return causal_prefix_lsso_triton_forward(U, C, mu, gamma, exclusive=exclusive, chunk_size=chunk_size)
