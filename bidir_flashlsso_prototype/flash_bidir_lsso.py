from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from lsso.mathdx_backend import solve_spd_or_torch


@triton.autotune(
    configs=[
        triton.Config({"BL": BL, "BR": BR}, num_warps=num_warps, num_stages=2)
        for BL in (32, 64, 128)
        for BR in (16, 32)
        for num_warps in (2, 4, 8)
    ],
    key=["N", "R"],
)
@triton.jit
def _utu_kernel(
    U,
    S,
    stride_u_bh,
    stride_u_n,
    stride_u_r,
    stride_s_bh,
    stride_s_i,
    stride_s_j,
    N: tl.constexpr,
    R: tl.constexpr,
    BL: tl.constexpr,
    BR: tl.constexpr,
):
    start_j, start_i, off_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = start_i * BR + tl.arange(0, BR)
    cols = start_j * BR + tl.arange(0, BR)
    offs_n = tl.arange(0, BL)
    acc = tl.zeros([BR, BR], dtype=tl.float32)

    for start_n in range(0, N, BL):
        n = start_n + offs_n
        u_i = tl.load(
            U + off_bh * stride_u_bh + n[None, :] * stride_u_n + rows[:, None] * stride_u_r,
            mask=(rows[:, None] < R) & (n[None, :] < N),
            other=0.0,
        ).to(tl.float32)
        u_j = tl.load(
            U + off_bh * stride_u_bh + n[:, None] * stride_u_n + cols[None, :] * stride_u_r,
            mask=(n[:, None] < N) & (cols[None, :] < R),
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(u_i, u_j, allow_tf32=False)

    tl.store(
        S + off_bh * stride_s_bh + rows[:, None] * stride_s_i + cols[None, :] * stride_s_j,
        acc,
        mask=(rows[:, None] < R) & (cols[None, :] < R),
    )


@triton.autotune(
    configs=[
        triton.Config({"BL": BL, "BR": BR, "BD": BD}, num_warps=num_warps, num_stages=2)
        for BL in (32, 64, 128)
        for BR in (16, 32)
        for BD in (32, 64)
        for num_warps in (2, 4, 8)
    ],
    key=["N", "R", "DH"],
)
@triton.jit
def _utc_kernel(
    U,
    C,
    P,
    stride_u_bh,
    stride_u_n,
    stride_u_r,
    stride_c_bh,
    stride_c_n,
    stride_c_d,
    stride_p_bh,
    stride_p_r,
    stride_p_d,
    N: tl.constexpr,
    R: tl.constexpr,
    DH: tl.constexpr,
    BL: tl.constexpr,
    BR: tl.constexpr,
    BD: tl.constexpr,
):
    start_d, start_r, off_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = start_r * BR + tl.arange(0, BR)
    dcols = start_d * BD + tl.arange(0, BD)
    offs_n = tl.arange(0, BL)
    acc = tl.zeros([BR, BD], dtype=tl.float32)

    for start_n in range(0, N, BL):
        n = start_n + offs_n
        u = tl.load(
            U + off_bh * stride_u_bh + n[None, :] * stride_u_n + rows[:, None] * stride_u_r,
            mask=(rows[:, None] < R) & (n[None, :] < N),
            other=0.0,
        ).to(tl.float32)
        c = tl.load(
            C + off_bh * stride_c_bh + n[:, None] * stride_c_n + dcols[None, :] * stride_c_d,
            mask=(n[:, None] < N) & (dcols[None, :] < DH),
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(u, c, allow_tf32=False)

    tl.store(
        P + off_bh * stride_p_bh + rows[:, None] * stride_p_r + dcols[None, :] * stride_p_d,
        acc,
        mask=(rows[:, None] < R) & (dcols[None, :] < DH),
    )


@triton.autotune(
    configs=[
        triton.Config({"BL": BL, "BR": BR}, num_warps=num_warps, num_stages=2)
        for BL in (32, 64, 128)
        for BR in (16, 32)
        for num_warps in (2, 4, 8)
    ],
    key=["N", "R", "DH"],
)
@triton.jit
def _stats_kernel(
    U,
    C,
    S,
    P,
    stride_u_bh,
    stride_u_n,
    stride_u_r,
    stride_c_bh,
    stride_c_n,
    stride_c_d,
    stride_s_bh,
    stride_s_i,
    stride_s_j,
    stride_p_bh,
    stride_p_r,
    stride_p_d,
    N: tl.constexpr,
    R: tl.constexpr,
    DH: tl.constexpr,
    BL: tl.constexpr,
    BR: tl.constexpr,
):
    start_aux, start_r, off_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ndb: tl.constexpr = tl.cdiv(DH, BR)
    rows = start_r * BR + tl.arange(0, BR)
    cols = start_aux * BR + tl.arange(0, BR)
    offs_n = tl.arange(0, BL)
    acc = tl.zeros([BR, BR], dtype=tl.float32)

    for start_n in range(0, N, BL):
        n = start_n + offs_n
        u = tl.load(
            U + off_bh * stride_u_bh + n[None, :] * stride_u_n + rows[:, None] * stride_u_r,
            mask=(rows[:, None] < R) & (n[None, :] < N),
            other=0.0,
        ).to(tl.float32)
        if start_aux < ndb:
            c = tl.load(
                C + off_bh * stride_c_bh + n[:, None] * stride_c_n + cols[None, :] * stride_c_d,
                mask=(n[:, None] < N) & (cols[None, :] < DH),
                other=0.0,
            ).to(tl.float32)
        else:
            s_cols = (start_aux - ndb) * BR + tl.arange(0, BR)
            c = tl.load(
                U + off_bh * stride_u_bh + n[:, None] * stride_u_n + s_cols[None, :] * stride_u_r,
                mask=(n[:, None] < N) & (s_cols[None, :] < R),
                other=0.0,
            ).to(tl.float32)
        acc += tl.dot(u, c, allow_tf32=False)

    if start_aux < ndb:
        tl.store(
            P + off_bh * stride_p_bh + rows[:, None] * stride_p_r + cols[None, :] * stride_p_d,
            acc,
            mask=(rows[:, None] < R) & (cols[None, :] < DH),
        )
    else:
        s_cols = (start_aux - ndb) * BR + tl.arange(0, BR)
        tl.store(
            S + off_bh * stride_s_bh + rows[:, None] * stride_s_i + s_cols[None, :] * stride_s_j,
            acc,
            mask=(rows[:, None] < R) & (s_cols[None, :] < R),
        )


@triton.jit
def _utu32_kernel(
    U,
    S,
    stride_u_bh,
    stride_u_n,
    stride_u_r,
    stride_s_bh,
    stride_s_i,
    stride_s_j,
    N: tl.constexpr,
    BL: tl.constexpr,
):
    off_bh = tl.program_id(0)
    r = tl.arange(0, 32)
    n_offs = tl.arange(0, BL)
    acc = tl.zeros([32, 32], dtype=tl.float32)
    for start_n in range(0, N, BL):
        n = start_n + n_offs
        u_t = tl.load(
            U + off_bh * stride_u_bh + r[:, None] * stride_u_r + n[None, :] * stride_u_n,
            mask=n[None, :] < N,
            other=0.0,
        ).to(tl.float32)
        u = tl.load(
            U + off_bh * stride_u_bh + n[:, None] * stride_u_n + r[None, :] * stride_u_r,
            mask=n[:, None] < N,
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(u_t, u, allow_tf32=False)
    tl.store(S + off_bh * stride_s_bh + r[:, None] * stride_s_i + r[None, :] * stride_s_j, acc)


@triton.jit
def _utc32_d64_kernel(
    U,
    C,
    P,
    stride_u_bh,
    stride_u_n,
    stride_u_r,
    stride_c_bh,
    stride_c_n,
    stride_c_d,
    stride_p_bh,
    stride_p_r,
    stride_p_d,
    N: tl.constexpr,
    BL: tl.constexpr,
):
    off_bh = tl.program_id(0)
    r = tl.arange(0, 32)
    d = tl.arange(0, 64)
    n_offs = tl.arange(0, BL)
    acc = tl.zeros([32, 64], dtype=tl.float32)
    for start_n in range(0, N, BL):
        n = start_n + n_offs
        u_t = tl.load(
            U + off_bh * stride_u_bh + r[:, None] * stride_u_r + n[None, :] * stride_u_n,
            mask=n[None, :] < N,
            other=0.0,
        ).to(tl.float32)
        c = tl.load(
            C + off_bh * stride_c_bh + n[:, None] * stride_c_n + d[None, :] * stride_c_d,
            mask=n[:, None] < N,
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(u_t, c, allow_tf32=False)
    tl.store(P + off_bh * stride_p_bh + r[:, None] * stride_p_r + d[None, :] * stride_p_d, acc)


@triton.autotune(
    configs=[
        triton.Config({"BL": BL, "BR": BR, "BD": BD}, num_warps=num_warps, num_stages=2)
        for BL in (32, 64, 128)
        for BR in (16, 32)
        for BD in (32, 64)
        for num_warps in (2, 4, 8)
    ],
    key=["N", "R", "DH"],
)
@triton.jit
def _out_kernel(
    U,
    C,
    K,
    Y,
    MU,
    GAMMA,
    stride_u_bh,
    stride_u_n,
    stride_u_r,
    stride_c_bh,
    stride_c_n,
    stride_c_d,
    stride_k_bh,
    stride_k_r,
    stride_k_d,
    stride_y_bh,
    stride_y_n,
    stride_y_d,
    N: tl.constexpr,
    R: tl.constexpr,
    DH: tl.constexpr,
    BL: tl.constexpr,
    BR: tl.constexpr,
    BD: tl.constexpr,
):
    start_d, start_n, off_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows_n = start_n * BL + tl.arange(0, BL)
    dcols = start_d * BD + tl.arange(0, BD)
    rcols = tl.arange(0, BR)

    mu = tl.load(MU + off_bh).to(tl.float32)
    gamma = tl.load(GAMMA + off_bh).to(tl.float32)
    inv_mu = 1.0 / mu
    beta = gamma * inv_mu * inv_mu
    corr = tl.zeros([BL, BD], dtype=tl.float32)

    for start_r in range(0, R, BR):
        rr = start_r + rcols
        u = tl.load(
            U + off_bh * stride_u_bh + rows_n[:, None] * stride_u_n + rr[None, :] * stride_u_r,
            mask=(rows_n[:, None] < N) & (rr[None, :] < R),
            other=0.0,
        )
        k = tl.load(
            K + off_bh * stride_k_bh + rr[:, None] * stride_k_r + dcols[None, :] * stride_k_d,
            mask=(rr[:, None] < R) & (dcols[None, :] < DH),
            other=0.0,
        )
        corr += tl.dot(u.to(tl.float32), k.to(tl.float32), allow_tf32=False)

    c = tl.load(
        C + off_bh * stride_c_bh + rows_n[:, None] * stride_c_n + dcols[None, :] * stride_c_d,
        mask=(rows_n[:, None] < N) & (dcols[None, :] < DH),
        other=0.0,
    ).to(tl.float32)
    y = inv_mu * c - beta * corr
    tl.store(
        Y + off_bh * stride_y_bh + rows_n[:, None] * stride_y_n + dcols[None, :] * stride_y_d,
        y,
        mask=(rows_n[:, None] < N) & (dcols[None, :] < DH),
    )


@triton.jit
def _out32_d64_kernel(
    U,
    C,
    K,
    Y,
    MU,
    GAMMA,
    stride_u_bh,
    stride_u_n,
    stride_u_r,
    stride_c_bh,
    stride_c_n,
    stride_c_d,
    stride_k_bh,
    stride_k_r,
    stride_k_d,
    stride_y_bh,
    stride_y_n,
    stride_y_d,
    N: tl.constexpr,
    BL: tl.constexpr,
):
    start_n, off_bh = tl.program_id(0), tl.program_id(1)
    n = start_n * BL + tl.arange(0, BL)
    r = tl.arange(0, 32)
    d = tl.arange(0, 64)
    mu = tl.load(MU + off_bh).to(tl.float32)
    gamma = tl.load(GAMMA + off_bh).to(tl.float32)
    inv_mu = 1.0 / mu
    beta = gamma * inv_mu * inv_mu
    u = tl.load(
        U + off_bh * stride_u_bh + n[:, None] * stride_u_n + r[None, :] * stride_u_r,
        mask=n[:, None] < N,
        other=0.0,
    ).to(tl.float32)
    k = tl.load(
        K + off_bh * stride_k_bh + r[:, None] * stride_k_r + d[None, :] * stride_k_d,
    ).to(tl.float32)
    c = tl.load(
        C + off_bh * stride_c_bh + n[:, None] * stride_c_n + d[None, :] * stride_c_d,
        mask=n[:, None] < N,
        other=0.0,
    ).to(tl.float32)
    y = inv_mu * c - beta * tl.dot(u, k, allow_tf32=False)
    tl.store(
        Y + off_bh * stride_y_bh + n[:, None] * stride_y_n + d[None, :] * stride_y_d,
        y,
        mask=n[:, None] < N,
    )


@triton.autotune(
    configs=[
        triton.Config({"BL": BL, "BR": BR, "BD": BD}, num_warps=num_warps, num_stages=2)
        for BL in (32, 64, 128)
        for BR in (16, 32)
        for BD in (32, 64)
        for num_warps in (2, 4, 8)
    ],
    key=["N", "R", "DH"],
)
@triton.jit
def _bwd_token_kernel(
    U,
    C,
    DO,
    K,
    DP,
    DS,
    DU,
    DC,
    MU,
    GAMMA,
    stride_u_bh,
    stride_u_n,
    stride_u_r,
    stride_c_bh,
    stride_c_n,
    stride_c_d,
    stride_k_bh,
    stride_k_r,
    stride_k_d,
    stride_dp_bh,
    stride_dp_r,
    stride_dp_d,
    stride_ds_bh,
    stride_ds_i,
    stride_ds_j,
    stride_du_bh,
    stride_du_n,
    stride_du_r,
    stride_dc_bh,
    stride_dc_n,
    stride_dc_d,
    N: tl.constexpr,
    R: tl.constexpr,
    DH: tl.constexpr,
    BL: tl.constexpr,
    BR: tl.constexpr,
    BD: tl.constexpr,
):
    start_aux, start_n, off_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows_n = start_n * BL + tl.arange(0, BL)
    rcols = start_aux * BR + tl.arange(0, BR)
    dcols = start_aux * BD + tl.arange(0, BD)
    mu = tl.load(MU + off_bh).to(tl.float32)
    gamma = tl.load(GAMMA + off_bh).to(tl.float32)
    inv_mu = 1.0 / mu
    beta = gamma * inv_mu * inv_mu

    if start_aux * BR < R:
        du = tl.zeros([BL, BR], dtype=tl.float32)
        for start_d in range(0, DH, BD):
            dd = start_d + tl.arange(0, BD)
            do = tl.load(
                DO + off_bh * stride_c_bh + rows_n[:, None] * stride_c_n + dd[None, :] * stride_c_d,
                mask=(rows_n[:, None] < N) & (dd[None, :] < DH),
                other=0.0,
            )
            c = tl.load(
                C + off_bh * stride_c_bh + rows_n[:, None] * stride_c_n + dd[None, :] * stride_c_d,
                mask=(rows_n[:, None] < N) & (dd[None, :] < DH),
                other=0.0,
            )
            k = tl.load(
                K + off_bh * stride_k_bh + rcols[:, None] * stride_k_r + dd[None, :] * stride_k_d,
                mask=(rcols[:, None] < R) & (dd[None, :] < DH),
                other=0.0,
            )
            dp = tl.load(
                DP + off_bh * stride_dp_bh + rcols[:, None] * stride_dp_r + dd[None, :] * stride_dp_d,
                mask=(rcols[:, None] < R) & (dd[None, :] < DH),
                other=0.0,
            )
            du += -beta * tl.dot(do.to(tl.float32), tl.trans(k.to(tl.float32)), allow_tf32=False)
            du += tl.dot(c.to(tl.float32), tl.trans(dp.to(tl.float32)), allow_tf32=False)

        for start_r in range(0, R, BR):
            rr = start_r + tl.arange(0, BR)
            u = tl.load(
                U + off_bh * stride_u_bh + rows_n[:, None] * stride_u_n + rr[None, :] * stride_u_r,
                mask=(rows_n[:, None] < N) & (rr[None, :] < R),
                other=0.0,
            )
            ds_a = tl.load(
                DS + off_bh * stride_ds_bh + rr[:, None] * stride_ds_i + rcols[None, :] * stride_ds_j,
                mask=(rr[:, None] < R) & (rcols[None, :] < R),
                other=0.0,
            )
            ds_b = tl.load(
                DS + off_bh * stride_ds_bh + rcols[:, None] * stride_ds_i + rr[None, :] * stride_ds_j,
                mask=(rcols[:, None] < R) & (rr[None, :] < R),
                other=0.0,
            )
            du += tl.dot(u.to(tl.float32), (ds_a + tl.trans(ds_b)).to(tl.float32), allow_tf32=False)

        tl.store(
            DU + off_bh * stride_du_bh + rows_n[:, None] * stride_du_n + rcols[None, :] * stride_du_r,
            du,
            mask=(rows_n[:, None] < N) & (rcols[None, :] < R),
        )

    if start_aux * BD < DH:
        dc = tl.load(
            DO + off_bh * stride_c_bh + rows_n[:, None] * stride_c_n + dcols[None, :] * stride_c_d,
            mask=(rows_n[:, None] < N) & (dcols[None, :] < DH),
            other=0.0,
        ).to(tl.float32) * inv_mu
        for start_r in range(0, R, BR):
            rr = start_r + tl.arange(0, BR)
            u = tl.load(
                U + off_bh * stride_u_bh + rows_n[:, None] * stride_u_n + rr[None, :] * stride_u_r,
                mask=(rows_n[:, None] < N) & (rr[None, :] < R),
                other=0.0,
            )
            dp = tl.load(
                DP + off_bh * stride_dp_bh + rr[:, None] * stride_dp_r + dcols[None, :] * stride_dp_d,
                mask=(rr[:, None] < R) & (dcols[None, :] < DH),
                other=0.0,
            )
            dc += tl.dot(u.to(tl.float32), dp.to(tl.float32), allow_tf32=False)

        tl.store(
            DC + off_bh * stride_dc_bh + rows_n[:, None] * stride_dc_n + dcols[None, :] * stride_dc_d,
            dc,
            mask=(rows_n[:, None] < N) & (dcols[None, :] < DH),
        )


def _torch_bidir_lsso(U: torch.Tensor, C: torch.Tensor, mu: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    B, H, _N, r = U.shape
    dh = C.shape[-1]
    U32 = U.float()
    C32 = C.float()
    mu32 = mu.reshape(1, H, 1, 1).float()
    gamma32 = gamma.reshape(1, H, 1, 1).float()
    inv_mu = mu32.reciprocal()
    alpha = gamma32 * inv_mu
    beta = alpha * inv_mu
    S = torch.einsum("bhnr,bhns->bhrs", U32, U32)
    P = torch.einsum("bhnr,bhnd->bhrd", U32, C32)
    eye = torch.eye(r, device=U.device, dtype=torch.float32).view(1, 1, r, r)
    G = eye + alpha * S
    K = torch.linalg.solve(G.reshape(B * H, r, r), P.reshape(B * H, r, dh)).view(B, H, r, dh)
    Y = inv_mu * C32 - beta * torch.einsum("bhnr,bhrd->bhnd", U32, K)
    return Y.to(C.dtype)


class FlashBidirLSSOFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, U: torch.Tensor, C: torch.Tensor, mu: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        if not U.is_cuda or not C.is_cuda:
            y = _torch_bidir_lsso(U, C, mu, gamma)
            ctx.save_for_backward(U, C, mu, gamma)
            return y
        if U.dim() != 4 or C.dim() != 4:
            raise ValueError("U and C must have shapes [B, H, N, r] and [B, H, N, dh]")
        if U.shape[:3] != C.shape[:3]:
            raise ValueError(f"U and C leading dimensions differ: {tuple(U.shape)} vs {tuple(C.shape)}")
        U = U.contiguous()
        C = C.contiguous()
        B, H, N, r = U.shape
        dh = C.shape[-1]
        if r not in (16, 32):
            raise ValueError(f"flash bidir LSSO v0 supports rank 16 or 32, got {r}")
        if dh > 128:
            raise ValueError(f"flash bidir LSSO v0 supports head_dim <= 128, got {dh}")

        bh = B * H
        U_bh = U.reshape(bh, N, r)
        C_bh = C.reshape(bh, N, dh)
        S = torch.empty(bh, r, r, device=U.device, dtype=torch.float32)
        P = torch.empty(bh, r, dh, device=U.device, dtype=torch.float32)
        if r == 32 and dh == 64:
            _utu32_kernel[(bh,)](
                U_bh,
                S,
                U_bh.stride(0),
                U_bh.stride(1),
                U_bh.stride(2),
                S.stride(0),
                S.stride(1),
                S.stride(2),
                N=N,
                BL=128,
                num_warps=8,
            )
            _utc32_d64_kernel[(bh,)](
                U_bh,
                C_bh,
                P,
                U_bh.stride(0),
                U_bh.stride(1),
                U_bh.stride(2),
                C_bh.stride(0),
                C_bh.stride(1),
                C_bh.stride(2),
                P.stride(0),
                P.stride(1),
                P.stride(2),
                N=N,
                BL=128,
                num_warps=8,
            )
        else:
            grid_stats = lambda meta: (triton.cdiv(dh, meta["BR"]) + triton.cdiv(r, meta["BR"]), triton.cdiv(r, meta["BR"]), bh)
            _stats_kernel[grid_stats](
                U_bh,
                C_bh,
                S,
                P,
                U_bh.stride(0),
                U_bh.stride(1),
                U_bh.stride(2),
                C_bh.stride(0),
                C_bh.stride(1),
                C_bh.stride(2),
                S.stride(0),
                S.stride(1),
                S.stride(2),
                P.stride(0),
                P.stride(1),
                P.stride(2),
                N=N,
                R=r,
                DH=dh,
            )

        mu_bh = mu.reshape(1, H).expand(B, H).reshape(bh).float().contiguous()
        gamma_bh = gamma.reshape(1, H).expand(B, H).reshape(bh).float().contiguous()
        alpha = (gamma_bh / mu_bh).view(bh, 1, 1)
        eye = torch.eye(r, device=U.device, dtype=torch.float32).view(1, r, r)
        G = eye + alpha * S
        K = solve_spd_or_torch(G, P).contiguous()

        Y_bh = torch.empty(bh, N, dh, device=U.device, dtype=C.dtype)
        if r == 32 and dh == 64:
            _out32_d64_kernel[(triton.cdiv(N, 128), bh)](
                U_bh,
                C_bh,
                K,
                Y_bh,
                mu_bh,
                gamma_bh,
                U_bh.stride(0),
                U_bh.stride(1),
                U_bh.stride(2),
                C_bh.stride(0),
                C_bh.stride(1),
                C_bh.stride(2),
                K.stride(0),
                K.stride(1),
                K.stride(2),
                Y_bh.stride(0),
                Y_bh.stride(1),
                Y_bh.stride(2),
                N=N,
                BL=128,
                num_warps=4,
            )
        else:
            grid_y = lambda meta: (triton.cdiv(dh, meta["BD"]), triton.cdiv(N, meta["BL"]), bh)
            _out_kernel[grid_y](
                U_bh,
                C_bh,
                K,
                Y_bh,
                mu_bh,
                gamma_bh,
                U_bh.stride(0),
                U_bh.stride(1),
                U_bh.stride(2),
                C_bh.stride(0),
                C_bh.stride(1),
                C_bh.stride(2),
                K.stride(0),
                K.stride(1),
                K.stride(2),
                Y_bh.stride(0),
                Y_bh.stride(1),
                Y_bh.stride(2),
                N=N,
                R=r,
                DH=dh,
            )
        Y = Y_bh.view(B, H, N, dh)
        ctx.save_for_backward(U, C, mu, gamma, S, K)
        return Y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        U, C, mu, gamma, S, K = ctx.saved_tensors
        needs = ctx.needs_input_grad
        if not U.is_cuda or not C.is_cuda:
            with torch.enable_grad():
                U_ref = U.detach().requires_grad_(needs[0])
                C_ref = C.detach().requires_grad_(needs[1])
                mu_ref = mu.detach().requires_grad_(needs[2])
                gamma_ref = gamma.detach().requires_grad_(needs[3])
                Y_ref = _torch_bidir_lsso(U_ref, C_ref, mu_ref, gamma_ref)
            return torch.autograd.grad(
                Y_ref,
                (U_ref, C_ref, mu_ref, gamma_ref),
                grad_output,
                allow_unused=True,
            )

        U = U.contiguous()
        C = C.contiguous()
        DO = grad_output.contiguous()
        B, H, N, r = U.shape
        dh = C.shape[-1]
        bh = B * H
        U_bh = U.reshape(bh, N, r)
        C_bh = C.reshape(bh, N, dh)
        DO_bh = DO.reshape(bh, N, dh)
        K = K.contiguous()
        S = S.contiguous()

        mu_bh = mu.reshape(1, H).expand(B, H).reshape(bh).float().contiguous()
        gamma_bh = gamma.reshape(1, H).expand(B, H).reshape(bh).float().contiguous()
        inv_mu = mu_bh.reciprocal()
        alpha = gamma_bh * inv_mu
        beta = alpha * inv_mu

        dK_raw = torch.empty(bh, r, dh, device=U.device, dtype=torch.float32)
        grid_dk = lambda meta: (triton.cdiv(dh, meta["BD"]), triton.cdiv(r, meta["BR"]), bh)
        _utc_kernel[grid_dk](
            U_bh,
            DO_bh,
            dK_raw,
            U_bh.stride(0),
            U_bh.stride(1),
            U_bh.stride(2),
            DO_bh.stride(0),
            DO_bh.stride(1),
            DO_bh.stride(2),
            dK_raw.stride(0),
            dK_raw.stride(1),
            dK_raw.stride(2),
            N=N,
            R=r,
            DH=dh,
        )
        dK = -beta.view(bh, 1, 1) * dK_raw

        eye = torch.eye(r, device=U.device, dtype=torch.float32).view(1, r, r)
        G = eye + alpha.view(bh, 1, 1) * S
        dP = solve_spd_or_torch(G, dK)
        dG = -torch.bmm(dP, K.transpose(1, 2))
        dS = alpha.view(bh, 1, 1) * dG

        corr = torch.bmm(U_bh.float(), K)
        d_a = (DO_bh.float() * C_bh.float()).sum(dim=(1, 2))
        d_beta = -(DO_bh.float() * corr).sum(dim=(1, 2))
        d_alpha = (dG * S).sum(dim=(1, 2))
        dgamma_bh = d_alpha * inv_mu + d_beta * inv_mu * inv_mu
        dmu_bh = (
            -d_a * inv_mu * inv_mu
            - d_alpha * gamma_bh * inv_mu * inv_mu
            - 2.0 * d_beta * gamma_bh * inv_mu * inv_mu * inv_mu
        )
        dgamma = dgamma_bh.view(B, H).sum(dim=0).to(gamma.dtype) if needs[3] else None
        dmu = dmu_bh.view(B, H).sum(dim=0).to(mu.dtype) if needs[2] else None

        dU = torch.empty_like(U_bh)
        dC = torch.empty_like(C_bh)
        grid_tok = lambda meta: (
            max(triton.cdiv(r, meta["BR"]), triton.cdiv(dh, meta["BD"])),
            triton.cdiv(N, meta["BL"]),
            bh,
        )
        _bwd_token_kernel[grid_tok](
            U_bh,
            C_bh,
            DO_bh,
            K,
            dP,
            dS,
            dU,
            dC,
            mu_bh,
            gamma_bh,
            U_bh.stride(0),
            U_bh.stride(1),
            U_bh.stride(2),
            C_bh.stride(0),
            C_bh.stride(1),
            C_bh.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            dP.stride(0),
            dP.stride(1),
            dP.stride(2),
            dS.stride(0),
            dS.stride(1),
            dS.stride(2),
            dU.stride(0),
            dU.stride(1),
            dU.stride(2),
            dC.stride(0),
            dC.stride(1),
            dC.stride(2),
            N=N,
            R=r,
            DH=dh,
        )
        dU_out = dU.view(B, H, N, r) if needs[0] else None
        dC_out = dC.view(B, H, N, dh) if needs[1] else None
        return dU_out, dC_out, dmu, dgamma


def flash_bidir_lsso(U: torch.Tensor, C: torch.Tensor, mu: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """Bidirectional LSSO core with Triton forward and reference backward.

    Args:
        U: solve basis, [B, H, N, r].
        C: value/state tensor, [B, H, N, dh].
        mu: positive per-head shift, [H] or broadcastable.
        gamma: per-head global strength, [H] or broadcastable.
    """
    return FlashBidirLSSOFunction.apply(U, C, mu, gamma)


@torch.no_grad()
def flash_bidir_lsso_forward_parts(
    U: torch.Tensor,
    C: torch.Tensor,
    mu: torch.Tensor,
    gamma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expose forward pieces for profiling. Not used by training."""
    if not U.is_cuda or not C.is_cuda:
        raise RuntimeError("CUDA tensors are required for profiling pieces.")
    U = U.contiguous()
    C = C.contiguous()
    B, H, N, r = U.shape
    dh = C.shape[-1]
    bh = B * H
    U_bh = U.reshape(bh, N, r)
    C_bh = C.reshape(bh, N, dh)
    S = torch.empty(bh, r, r, device=U.device, dtype=torch.float32)
    P = torch.empty(bh, r, dh, device=U.device, dtype=torch.float32)
    grid_stats = lambda meta: (triton.cdiv(dh, meta["BR"]) + triton.cdiv(r, meta["BR"]), triton.cdiv(r, meta["BR"]), bh)
    _stats_kernel[grid_stats](
        U_bh,
        C_bh,
        S,
        P,
        U_bh.stride(0),
        U_bh.stride(1),
        U_bh.stride(2),
        C_bh.stride(0),
        C_bh.stride(1),
        C_bh.stride(2),
        S.stride(0),
        S.stride(1),
        S.stride(2),
        P.stride(0),
        P.stride(1),
        P.stride(2),
        N=N,
        R=r,
        DH=dh,
    )
    mu_bh = mu.reshape(1, H).expand(B, H).reshape(bh).float().contiguous()
    gamma_bh = gamma.reshape(1, H).expand(B, H).reshape(bh).float().contiguous()
    alpha = (gamma_bh / mu_bh).view(bh, 1, 1)
    eye = torch.eye(r, device=U.device, dtype=torch.float32).view(1, r, r)
    G = eye + alpha * S
    K = torch.linalg.solve_ex(G, P, check_errors=False)[0].contiguous()
    Y_bh = torch.empty(bh, N, dh, device=U.device, dtype=C.dtype)
    grid_y = lambda meta: (triton.cdiv(dh, meta["BD"]), triton.cdiv(N, meta["BL"]), bh)
    _out_kernel[grid_y](
        U_bh,
        C_bh,
        K,
        Y_bh,
        mu_bh,
        gamma_bh,
        U_bh.stride(0),
        U_bh.stride(1),
        U_bh.stride(2),
        C_bh.stride(0),
        C_bh.stride(1),
        C_bh.stride(2),
        K.stride(0),
        K.stride(1),
        K.stride(2),
        Y_bh.stride(0),
        Y_bh.stride(1),
        Y_bh.stride(2),
        N=N,
        R=r,
        DH=dh,
    )
    return S, P, G, K, Y_bh.view(B, H, N, dh), torch.stack((mu_bh, gamma_bh), dim=0)
