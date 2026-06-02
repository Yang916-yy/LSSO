from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised on systems without Triton
    triton = None
    tl = None


def _next_power_of_2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


def triton_available() -> bool:
    return triton is not None and tl is not None


if triton_available():

    @triton.jit
    def _fused_gram_kernel(
        U,
        C,
        UtU,
        UtC,
        N: tl.constexpr,
        R: tl.constexpr,
        DH: tl.constexpr,
        stride_u_bh: tl.constexpr,
        stride_u_n: tl.constexpr,
        stride_u_r: tl.constexpr,
        stride_c_bh: tl.constexpr,
        stride_c_n: tl.constexpr,
        stride_c_dh: tl.constexpr,
        stride_utu_bh: tl.constexpr,
        stride_utu_i: tl.constexpr,
        stride_utu_j: tl.constexpr,
        stride_utc_bh: tl.constexpr,
        stride_utc_i: tl.constexpr,
        stride_utc_j: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_DH: tl.constexpr,
    ):
        bh = tl.program_id(0)
        offs_n = tl.arange(0, BLOCK_N)
        offs_r = tl.arange(0, BLOCK_R)
        offs_dh = tl.arange(0, BLOCK_DH)

        acc_utu = tl.zeros((BLOCK_R, BLOCK_R), tl.float32)
        acc_utc = tl.zeros((BLOCK_R, BLOCK_DH), tl.float32)

        for n0 in range(0, N, BLOCK_N):
            n = n0 + offs_n
            u = tl.load(
                U + bh * stride_u_bh + n[:, None] * stride_u_n + offs_r[None, :] * stride_u_r,
                mask=(n[:, None] < N) & (offs_r[None, :] < R),
                other=0.0,
            )
            c = tl.load(
                C + bh * stride_c_bh + n[:, None] * stride_c_n + offs_dh[None, :] * stride_c_dh,
                mask=(n[:, None] < N) & (offs_dh[None, :] < DH),
                other=0.0,
            )
            u_t = tl.trans(u)
            acc_utu += tl.dot(u_t, u, input_precision="tf32")
            acc_utc += tl.dot(u_t, c, input_precision="tf32")

        tl.store(
            UtU + bh * stride_utu_bh + offs_r[:, None] * stride_utu_i + offs_r[None, :] * stride_utu_j,
            acc_utu,
            mask=(offs_r[:, None] < R) & (offs_r[None, :] < R),
        )
        tl.store(
            UtC + bh * stride_utc_bh + offs_r[:, None] * stride_utc_i + offs_dh[None, :] * stride_utc_j,
            acc_utc,
            mask=(offs_r[:, None] < R) & (offs_dh[None, :] < DH),
        )

    @triton.jit
    def _fused_system_kernel(
        U,
        C,
        gamma_over_mu,
        G,
        UtC,
        N: tl.constexpr,
        R: tl.constexpr,
        DH: tl.constexpr,
        stride_u_bh: tl.constexpr,
        stride_u_n: tl.constexpr,
        stride_u_r: tl.constexpr,
        stride_c_bh: tl.constexpr,
        stride_c_n: tl.constexpr,
        stride_c_dh: tl.constexpr,
        stride_g_bh: tl.constexpr,
        stride_g_i: tl.constexpr,
        stride_g_j: tl.constexpr,
        stride_utc_bh: tl.constexpr,
        stride_utc_i: tl.constexpr,
        stride_utc_j: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_DH: tl.constexpr,
    ):
        bh = tl.program_id(0)
        offs_n = tl.arange(0, BLOCK_N)
        offs_r = tl.arange(0, BLOCK_R)
        offs_dh = tl.arange(0, BLOCK_DH)

        acc_utu = tl.zeros((BLOCK_R, BLOCK_R), tl.float32)
        acc_utc = tl.zeros((BLOCK_R, BLOCK_DH), tl.float32)

        for n0 in range(0, N, BLOCK_N):
            n = n0 + offs_n
            u = tl.load(
                U + bh * stride_u_bh + n[:, None] * stride_u_n + offs_r[None, :] * stride_u_r,
                mask=(n[:, None] < N) & (offs_r[None, :] < R),
                other=0.0,
            )
            c = tl.load(
                C + bh * stride_c_bh + n[:, None] * stride_c_n + offs_dh[None, :] * stride_c_dh,
                mask=(n[:, None] < N) & (offs_dh[None, :] < DH),
                other=0.0,
            )
            u_t = tl.trans(u)
            acc_utu += tl.dot(u_t, u, input_precision="tf32")
            acc_utc += tl.dot(u_t, c, input_precision="tf32")

        scale = tl.load(gamma_over_mu + bh)
        ident = tl.where(offs_r[:, None] == offs_r[None, :], 1.0, 0.0)
        system = scale * acc_utu + ident
        tl.store(
            G + bh * stride_g_bh + offs_r[:, None] * stride_g_i + offs_r[None, :] * stride_g_j,
            system,
            mask=(offs_r[:, None] < R) & (offs_r[None, :] < R),
        )
        tl.store(
            UtC + bh * stride_utc_bh + offs_r[:, None] * stride_utc_i + offs_dh[None, :] * stride_utc_j,
            acc_utc,
            mask=(offs_r[:, None] < R) & (offs_dh[None, :] < DH),
        )

    @triton.jit
    def _correction_apply_kernel(
        U,
        K,
        C,
        inv_mu,
        gamma_over_mu2,
        Y,
        N: tl.constexpr,
        R: tl.constexpr,
        DH: tl.constexpr,
        stride_u_bh: tl.constexpr,
        stride_u_n: tl.constexpr,
        stride_u_r: tl.constexpr,
        stride_k_bh: tl.constexpr,
        stride_k_r: tl.constexpr,
        stride_k_dh: tl.constexpr,
        stride_c_bh: tl.constexpr,
        stride_c_n: tl.constexpr,
        stride_c_dh: tl.constexpr,
        stride_y_bh: tl.constexpr,
        stride_y_n: tl.constexpr,
        stride_y_dh: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_DH: tl.constexpr,
    ):
        bh = tl.program_id(0)
        tile_n = tl.program_id(1)
        tile_dh = tl.program_id(2)

        offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_r = tl.arange(0, BLOCK_R)
        offs_dh = tile_dh * BLOCK_DH + tl.arange(0, BLOCK_DH)

        acc = tl.zeros((BLOCK_N, BLOCK_DH), tl.float32)
        for r0 in range(0, R, BLOCK_R):
            r = r0 + offs_r
            u = tl.load(
                U + bh * stride_u_bh + offs_n[:, None] * stride_u_n + r[None, :] * stride_u_r,
                mask=(offs_n[:, None] < N) & (r[None, :] < R),
                other=0.0,
            )
            k = tl.load(
                K + bh * stride_k_bh + r[:, None] * stride_k_r + offs_dh[None, :] * stride_k_dh,
                mask=(r[:, None] < R) & (offs_dh[None, :] < DH),
                other=0.0,
            )
            acc += tl.dot(u, k, input_precision="tf32")

        c = tl.load(
            C + bh * stride_c_bh + offs_n[:, None] * stride_c_n + offs_dh[None, :] * stride_c_dh,
            mask=(offs_n[:, None] < N) & (offs_dh[None, :] < DH),
            other=0.0,
        ).to(tl.float32)
        local_scale = tl.load(inv_mu + bh)
        correction_scale = tl.load(gamma_over_mu2 + bh)
        y = local_scale * c - correction_scale * acc
        tl.store(
            Y + bh * stride_y_bh + offs_n[:, None] * stride_y_n + offs_dh[None, :] * stride_y_dh,
            y,
            mask=(offs_n[:, None] < N) & (offs_dh[None, :] < DH),
        )

    @triton.jit
    def _backward_fused_gram_kernel(
        U,
        Y,
        P,
        YtU,
        PtU,
        N: tl.constexpr,
        R: tl.constexpr,
        DH: tl.constexpr,
        stride_u_bh: tl.constexpr,
        stride_u_n: tl.constexpr,
        stride_u_r: tl.constexpr,
        stride_y_bh: tl.constexpr,
        stride_y_n: tl.constexpr,
        stride_y_dh: tl.constexpr,
        stride_p_bh: tl.constexpr,
        stride_p_n: tl.constexpr,
        stride_p_dh: tl.constexpr,
        stride_ytu_bh: tl.constexpr,
        stride_ytu_i: tl.constexpr,
        stride_ytu_j: tl.constexpr,
        stride_ptu_bh: tl.constexpr,
        stride_ptu_i: tl.constexpr,
        stride_ptu_j: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_DH: tl.constexpr,
    ):
        bh = tl.program_id(0)
        offs_n = tl.arange(0, BLOCK_N)
        offs_dh = tl.arange(0, BLOCK_DH)
        offs_r = tl.arange(0, BLOCK_R)

        acc_ytu = tl.zeros((BLOCK_DH, BLOCK_R), tl.float32)
        acc_ptu = tl.zeros((BLOCK_DH, BLOCK_R), tl.float32)

        for n0 in range(0, N, BLOCK_N):
            n = n0 + offs_n
            u = tl.load(
                U + bh * stride_u_bh + n[:, None] * stride_u_n + offs_r[None, :] * stride_u_r,
                mask=(n[:, None] < N) & (offs_r[None, :] < R),
                other=0.0,
            )
            y = tl.load(
                Y + bh * stride_y_bh + n[:, None] * stride_y_n + offs_dh[None, :] * stride_y_dh,
                mask=(n[:, None] < N) & (offs_dh[None, :] < DH),
                other=0.0,
            )
            p = tl.load(
                P + bh * stride_p_bh + n[:, None] * stride_p_n + offs_dh[None, :] * stride_p_dh,
                mask=(n[:, None] < N) & (offs_dh[None, :] < DH),
                other=0.0,
            )
            u_t = u
            acc_ytu += tl.dot(tl.trans(y), u_t, input_precision="tf32")
            acc_ptu += tl.dot(tl.trans(p), u_t, input_precision="tf32")

        tl.store(
            YtU + bh * stride_ytu_bh + offs_dh[:, None] * stride_ytu_i + offs_r[None, :] * stride_ytu_j,
            acc_ytu,
            mask=(offs_dh[:, None] < DH) & (offs_r[None, :] < R),
        )
        tl.store(
            PtU + bh * stride_ptu_bh + offs_dh[:, None] * stride_ptu_i + offs_r[None, :] * stride_ptu_j,
            acc_ptu,
            mask=(offs_dh[:, None] < DH) & (offs_r[None, :] < R),
        )

    @triton.jit
    def _backward_grad_u_kernel(
        Y,
        P,
        YtU,
        PtU,
        gamma,
        grad_U,
        N: tl.constexpr,
        R: tl.constexpr,
        DH: tl.constexpr,
        stride_y_bh: tl.constexpr,
        stride_y_n: tl.constexpr,
        stride_y_dh: tl.constexpr,
        stride_p_bh: tl.constexpr,
        stride_p_n: tl.constexpr,
        stride_p_dh: tl.constexpr,
        stride_ytu_bh: tl.constexpr,
        stride_ytu_i: tl.constexpr,
        stride_ytu_j: tl.constexpr,
        stride_ptu_bh: tl.constexpr,
        stride_ptu_i: tl.constexpr,
        stride_ptu_j: tl.constexpr,
        stride_gu_bh: tl.constexpr,
        stride_gu_n: tl.constexpr,
        stride_gu_r: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_DH: tl.constexpr,
    ):
        bh = tl.program_id(0)
        tile_n = tl.program_id(1)
        offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_r = tl.arange(0, BLOCK_R)
        offs_dh = tl.arange(0, BLOCK_DH)

        y = tl.load(
            Y + bh * stride_y_bh + offs_n[:, None] * stride_y_n + offs_dh[None, :] * stride_y_dh,
            mask=(offs_n[:, None] < N) & (offs_dh[None, :] < DH),
            other=0.0,
        )
        p = tl.load(
            P + bh * stride_p_bh + offs_n[:, None] * stride_p_n + offs_dh[None, :] * stride_p_dh,
            mask=(offs_n[:, None] < N) & (offs_dh[None, :] < DH),
            other=0.0,
        )
        ytu = tl.load(
            YtU + bh * stride_ytu_bh + offs_dh[:, None] * stride_ytu_i + offs_r[None, :] * stride_ytu_j,
            mask=(offs_dh[:, None] < DH) & (offs_r[None, :] < R),
            other=0.0,
        )
        ptu = tl.load(
            PtU + bh * stride_ptu_bh + offs_dh[:, None] * stride_ptu_i + offs_r[None, :] * stride_ptu_j,
            mask=(offs_dh[:, None] < DH) & (offs_r[None, :] < R),
            other=0.0,
        )
        acc = tl.dot(p, ytu, input_precision="tf32") + tl.dot(y, ptu, input_precision="tf32")
        scale = -tl.load(gamma + bh)
        tl.store(
            grad_U + bh * stride_gu_bh + offs_n[:, None] * stride_gu_n + offs_r[None, :] * stride_gu_r,
            scale * acc,
            mask=(offs_n[:, None] < N) & (offs_r[None, :] < R),
        )


def _can_use_triton(U: torch.Tensor, C: torch.Tensor, max_block: int = 64) -> bool:
    if not triton_available() or not U.is_cuda or not C.is_cuda:
        return False
    if U.dim() != 4 or C.dim() != 4:
        return False
    if U.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if C.dtype != U.dtype:
        return False
    r = U.shape[-1]
    dh = C.shape[-1]
    return r <= max_block and dh <= max_block


def fused_gram_utc_triton(U: torch.Tensor, C: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    Compute U^T U and U^T C in one Triton launch.

    U: [B, H, N, r]
    C: [B, H, N, dh]
    """
    if not _can_use_triton(U, C):
        return None

    B, H, N, r = U.shape
    dh = C.shape[-1]
    block_r = _next_power_of_2(r)
    block_dh = _next_power_of_2(dh)
    if block_r > 64 or block_dh > 64:
        return None

    U_bh = U.contiguous().flatten(0, 1)
    C_bh = C.contiguous().flatten(0, 1)
    UtU = torch.empty((B * H, r, r), device=U.device, dtype=torch.float32)
    UtC = torch.empty((B * H, r, dh), device=U.device, dtype=torch.float32)
    _fused_gram_kernel[(B * H,)](
        U_bh,
        C_bh,
        UtU,
        UtC,
        N,
        r,
        dh,
        U_bh.stride(0),
        U_bh.stride(1),
        U_bh.stride(2),
        C_bh.stride(0),
        C_bh.stride(1),
        C_bh.stride(2),
        UtU.stride(0),
        UtU.stride(1),
        UtU.stride(2),
        UtC.stride(0),
        UtC.stride(1),
        UtC.stride(2),
        BLOCK_N=64,
        BLOCK_R=block_r,
        BLOCK_DH=block_dh,
        num_warps=4,
    )
    return UtU.view(B, H, r, r), UtC.view(B, H, r, dh)


def _expand_bh_scale(scale: torch.Tensor, B: int, H: int) -> torch.Tensor:
    if scale.dim() == 1:
        return scale.view(1, H).expand(B, H).float().contiguous().flatten()
    return scale.expand(B, H, 1, 1).reshape(B, H).float().contiguous().flatten()


def fused_gram_system_utc_triton(
    U: torch.Tensor,
    C: torch.Tensor,
    gamma_over_mu: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    Compute G = I + gamma/mu * U^T U and U^T C in one Triton launch.

    This is the lower-launch inference path; it intentionally returns G rather
    than UtU, so diagnostics that need raw UtU should use fused_gram_utc_triton.
    """
    if not _can_use_triton(U, C):
        return None

    B, H, N, r = U.shape
    dh = C.shape[-1]
    block_r = _next_power_of_2(r)
    block_dh = _next_power_of_2(dh)
    if block_r > 64 or block_dh > 64:
        return None

    U_bh = U.contiguous().flatten(0, 1)
    C_bh = C.contiguous().flatten(0, 1)
    scale_bh = _expand_bh_scale(gamma_over_mu, B, H).to(device=U.device)
    G = torch.empty((B * H, r, r), device=U.device, dtype=torch.float32)
    UtC = torch.empty((B * H, r, dh), device=U.device, dtype=torch.float32)
    _fused_system_kernel[(B * H,)](
        U_bh,
        C_bh,
        scale_bh,
        G,
        UtC,
        N,
        r,
        dh,
        U_bh.stride(0),
        U_bh.stride(1),
        U_bh.stride(2),
        C_bh.stride(0),
        C_bh.stride(1),
        C_bh.stride(2),
        G.stride(0),
        G.stride(1),
        G.stride(2),
        UtC.stride(0),
        UtC.stride(1),
        UtC.stride(2),
        BLOCK_N=64,
        BLOCK_R=block_r,
        BLOCK_DH=block_dh,
        num_warps=4,
    )
    return G.view(B, H, r, r), UtC.view(B, H, r, dh)


def correction_apply_triton(
    U: torch.Tensor,
    C: torch.Tensor,
    K: torch.Tensor,
    mu: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor | None:
    """
    Compute Y = mu^-1 C - gamma / mu^2 * (U @ K) in one Triton launch.

    U: [B, H, N, r]
    C: [B, H, N, dh]
    K: [B, H, r, dh]
    mu/gamma: broadcastable to [B, H, 1, 1] or [H]
    """
    if not _can_use_triton(U, C) or K.device != U.device:
        return None

    B, H, N, r = U.shape
    dh = C.shape[-1]
    block_r = min(32, _next_power_of_2(r))
    block_dh = _next_power_of_2(dh)
    if block_dh > 64:
        return None

    mu_bh = _expand_bh_scale(mu, B, H).to(device=U.device)
    gamma_bh = _expand_bh_scale(gamma, B, H).to(device=U.device)
    inv_mu = mu_bh.reciprocal().contiguous()
    gamma_over_mu2 = (gamma_bh * inv_mu.square()).contiguous()

    U_bh = U.contiguous().flatten(0, 1)
    C_bh = C.contiguous().flatten(0, 1)
    K_bh = K.contiguous().flatten(0, 1)
    Y_bh = torch.empty((B * H, N, dh), device=U.device, dtype=C.dtype)
    grid = (B * H, triton.cdiv(N, 32), triton.cdiv(dh, block_dh))
    _correction_apply_kernel[grid](
        U_bh,
        K_bh,
        C_bh,
        inv_mu,
        gamma_over_mu2,
        Y_bh,
        N,
        r,
        dh,
        U_bh.stride(0),
        U_bh.stride(1),
        U_bh.stride(2),
        K_bh.stride(0),
        K_bh.stride(1),
        K_bh.stride(2),
        C_bh.stride(0),
        C_bh.stride(1),
        C_bh.stride(2),
        Y_bh.stride(0),
        Y_bh.stride(1),
        Y_bh.stride(2),
        BLOCK_N=32,
        BLOCK_R=block_r,
        BLOCK_DH=block_dh,
        num_warps=4,
    )
    return Y_bh.view(B, H, N, dh)


def backward_lowrank_grads_triton(
    U: torch.Tensor,
    Y: torch.Tensor,
    P: torch.Tensor,
    gamma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """
    Compute the low-rank-heavy custom backward terms:

        YtU = Y^T U
        PtU = P^T U
        grad_U = -gamma * (P @ YtU + Y @ PtU)

    Shapes:
        U: [B, H, N, r]
        Y/P: [B, H, N, dh]
        gamma: broadcastable to [B, H, 1, 1] or [H]
    """
    if not _can_use_triton(U, Y) or P.device != U.device or P.dtype != Y.dtype:
        return None

    B, H, N, r = U.shape
    dh = Y.shape[-1]
    block_r = _next_power_of_2(r)
    block_dh = _next_power_of_2(dh)
    if block_r > 64 or block_dh > 64:
        return None

    U_bh = U.contiguous().flatten(0, 1)
    Y_bh = Y.contiguous().flatten(0, 1)
    P_bh = P.contiguous().flatten(0, 1)
    YtU = torch.empty((B * H, dh, r), device=U.device, dtype=U.dtype)
    PtU = torch.empty((B * H, dh, r), device=U.device, dtype=U.dtype)
    _backward_fused_gram_kernel[(B * H,)](
        U_bh,
        Y_bh,
        P_bh,
        YtU,
        PtU,
        N,
        r,
        dh,
        U_bh.stride(0),
        U_bh.stride(1),
        U_bh.stride(2),
        Y_bh.stride(0),
        Y_bh.stride(1),
        Y_bh.stride(2),
        P_bh.stride(0),
        P_bh.stride(1),
        P_bh.stride(2),
        YtU.stride(0),
        YtU.stride(1),
        YtU.stride(2),
        PtU.stride(0),
        PtU.stride(1),
        PtU.stride(2),
        BLOCK_N=64,
        BLOCK_R=block_r,
        BLOCK_DH=block_dh,
        num_warps=4,
    )

    gamma_bh = _expand_bh_scale(gamma, B, H).to(device=U.device)
    grad_U = torch.empty((B * H, N, r), device=U.device, dtype=U.dtype)
    _backward_grad_u_kernel[(B * H, triton.cdiv(N, 32))](
        Y_bh,
        P_bh,
        YtU,
        PtU,
        gamma_bh,
        grad_U,
        N,
        r,
        dh,
        Y_bh.stride(0),
        Y_bh.stride(1),
        Y_bh.stride(2),
        P_bh.stride(0),
        P_bh.stride(1),
        P_bh.stride(2),
        YtU.stride(0),
        YtU.stride(1),
        YtU.stride(2),
        PtU.stride(0),
        PtU.stride(1),
        PtU.stride(2),
        grad_U.stride(0),
        grad_U.stride(1),
        grad_U.stride(2),
        BLOCK_N=32,
        BLOCK_R=block_r,
        BLOCK_DH=block_dh,
        num_warps=4,
    )
    return grad_U.view(B, H, N, r), YtU.view(B, H, dh, r), PtU.view(B, H, dh, r)
