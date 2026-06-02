from __future__ import annotations

import torch
import torch.nn.functional as F

from lsso.modules import lsso
from lsso.triton_kernels import (
    backward_lowrank_grads_triton,
    correction_apply_triton,
    fused_gram_system_utc_triton,
    fused_gram_utc_triton,
    triton_available,
)


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> None:
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol, msg=name)


def test_triton_lsso_kernels() -> None:
    if not torch.cuda.is_available() or not triton_available():
        print("Triton/CUDA unavailable; skipping Triton LSSO smoke test")
        return

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float16
    B, H, N, r, dh = 2, 4, 257, 32, 32
    U = torch.randn(B, H, N, r, device=device, dtype=dtype)
    C = torch.randn(B, H, N, dh, device=device, dtype=dtype)
    mu = F.softplus(torch.zeros(H, device=device)) + 1e-5
    gamma = 0.1 * torch.sigmoid(torch.full((H,), -4.0, device=device))

    fused = fused_gram_utc_triton(U, C)
    assert fused is not None
    UtU_tri, UtC_tri = fused
    U_bh = U.flatten(0, 1)
    C_bh = C.flatten(0, 1)
    UtU_ref = torch.bmm(U_bh.transpose(1, 2).float(), U_bh.float()).view(B, H, r, r)
    UtC_ref = torch.bmm(U_bh.transpose(1, 2).float(), C_bh.float()).view(B, H, r, dh)
    _assert_close("UtU", UtU_tri, UtU_ref, atol=5e-2, rtol=5e-3)
    _assert_close("UtC", UtC_tri, UtC_ref, atol=5e-2, rtol=5e-3)

    eye = torch.eye(r, device=device).view(1, 1, r, r)
    G = eye + (gamma.view(1, H, 1, 1) / mu.view(1, H, 1, 1)) * UtU_ref
    system = fused_gram_system_utc_triton(U, C, gamma.view(1, H, 1, 1) / mu.view(1, H, 1, 1))
    assert system is not None
    G_tri, UtC_system_tri = system
    _assert_close("G", G_tri, G, atol=5e-2, rtol=5e-3)
    _assert_close("system UtC", UtC_system_tri, UtC_ref, atol=5e-2, rtol=5e-3)

    L = torch.linalg.cholesky_ex(G.view(B * H, r, r), check_errors=False).L
    K = torch.cholesky_solve(
        UtC_ref.view(B * H, r, dh),
        L,
    ).to(dtype).view(B, H, r, dh)
    K_solve_ex = torch.linalg.solve_ex(
        G.view(B * H, r, r),
        UtC_ref.view(B * H, r, dh),
        check_errors=False,
    ).result.to(dtype).view(B, H, r, dh)
    _assert_close("cholesky solve", K, K_solve_ex, atol=5e-3, rtol=5e-3)
    Y_tri = correction_apply_triton(U, C, K, mu, gamma)
    assert Y_tri is not None
    local = C / mu.view(1, H, 1, 1)
    correction = (gamma / (mu * mu)).view(1, H, 1, 1) * torch.bmm(
        U_bh,
        K.flatten(0, 1),
    ).view(B, H, N, dh)
    _assert_close("correction apply", Y_tri, local - correction, atol=5e-3, rtol=5e-3)

    with torch.no_grad():
        Y_ref = lsso(U, C, mu, gamma, use_triton=False)
        Y_opt = lsso(U, C, mu, gamma, use_triton=True)
    _assert_close("lsso functional", Y_opt, Y_ref, atol=5e-2, rtol=5e-3)

    Y_half = Y_ref.to(dtype)
    P = torch.randn_like(Y_half)
    backward_terms = backward_lowrank_grads_triton(U, Y_half, P, gamma.view(1, H, 1, 1))
    assert backward_terms is not None
    grad_U_tri, YtU_tri, PtU_tri = backward_terms
    Y_bh = Y_half.flatten(0, 1)
    P_bh = P.flatten(0, 1)
    YtU_ref = torch.bmm(Y_bh.transpose(1, 2).float(), U_bh.float()).view(B, H, dh, r)
    PtU_ref = torch.bmm(P_bh.transpose(1, 2).float(), U_bh.float()).view(B, H, dh, r)
    grad_U_ref = -gamma.view(1, H, 1, 1).to(dtype) * (
        torch.bmm(P_bh, YtU_ref.flatten(0, 1).to(dtype))
        + torch.bmm(Y_bh, PtU_ref.flatten(0, 1).to(dtype))
    ).view(B, H, N, r)
    _assert_close("backward YtU", YtU_tri, YtU_ref, atol=5e-2, rtol=5e-3)
    _assert_close("backward PtU", PtU_tri, PtU_ref, atol=5e-2, rtol=5e-3)
    _assert_close("backward grad_U", grad_U_tri, grad_U_ref, atol=5e-3, rtol=5e-3)


if __name__ == "__main__":
    test_triton_lsso_kernels()
    print("triton lsso smoke passed")
