from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LSSODiagnostics:
    gamma_over_mu: torch.Tensor
    effective_rank: torch.Tensor
    correction_ratio: torch.Tensor


@dataclass
class LSSOAux:
    UtU: torch.Tensor | None
    local: torch.Tensor
    correction: torch.Tensor
    mu: torch.Tensor
    gamma: torch.Tensor


def _cholesky_spd_solve(G: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Solve a batch of SPD systems G @ x = rhs in float32."""
    chol = torch.linalg.cholesky_ex(G, check_errors=False).L
    return torch.cholesky_solve(rhs, chol)


def _lsso_woodbury_forward(
    U: torch.Tensor,
    C: torch.Tensor,
    mu: torch.Tensor,
    gamma: torch.Tensor,
    eye: torch.Tensor | None = None,
) -> torch.Tensor:
    B, H, N, r = U.shape
    dh = C.shape[-1]
    calc_dtype = torch.promote_types(U.dtype, C.dtype)
    solve_dtype = torch.float64 if calc_dtype == torch.float64 else torch.float32
    output_dtype = C.dtype if U.dtype == C.dtype else calc_dtype
    U_calc = U.to(calc_dtype)
    C_calc = C.to(calc_dtype)
    mu_calc = mu.to(calc_dtype)
    gamma_calc = gamma.to(calc_dtype)
    inv_mu = mu_calc.reciprocal()
    gamma_over_mu = gamma_calc * inv_mu
    gamma_over_mu2 = gamma_over_mu * inv_mu

    U_bh = U_calc.flatten(0, 1)
    C_bh = C_calc.flatten(0, 1)
    Ut_bh = U_bh.transpose(1, 2)
    UtU = torch.bmm(Ut_bh, U_bh).view(B, H, r, r)
    UtC = torch.bmm(Ut_bh, C_bh).view(B, H, r, dh)
    if eye is None:
        eye = torch.eye(r, device=U.device, dtype=calc_dtype).view(1, 1, r, r)
    G = eye.to(solve_dtype) + gamma_over_mu.to(solve_dtype) * UtU.to(solve_dtype)
    K = _cholesky_spd_solve(
        G.view(B * H, r, r),
        UtC.to(solve_dtype).view(B * H, r, dh),
    ).to(calc_dtype)
    UK = torch.bmm(U_bh, K).view(B, H, N, dh)
    return (inv_mu * C_calc - gamma_over_mu2 * UK).to(output_dtype)


class _LSSOAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        U: torch.Tensor,
        C: torch.Tensor,
        mu: torch.Tensor,
        gamma: torch.Tensor,
        eye: torch.Tensor | None,
        use_triton_backward: bool,
    ) -> torch.Tensor:
        Y = _lsso_woodbury_forward(U, C, mu, gamma, eye)
        ctx.save_for_backward(U, Y, mu, gamma)
        ctx.use_triton_backward = use_triton_backward
        return Y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        U, Y, mu, gamma = ctx.saved_tensors
        B, H, N, r = U.shape
        dh = Y.shape[-1]

        grad = grad_output.contiguous()
        P = _lsso_woodbury_forward(U, grad, mu, gamma)

        calc_dtype = torch.float64 if U.dtype == torch.float64 or Y.dtype == torch.float64 else torch.float32
        matmul_dtype = U.dtype if U.dtype in (torch.float16, torch.bfloat16) else calc_dtype
        U_m = U.to(matmul_dtype).flatten(0, 1)
        Y_m = Y.to(matmul_dtype).flatten(0, 1)
        P_m = P.to(matmul_dtype).flatten(0, 1)

        triton_grads = None
        if ctx.use_triton_backward and U.is_cuda and U.dtype in (torch.float16, torch.bfloat16, torch.float32):
            try:
                from .triton_kernels import backward_lowrank_grads_triton

                triton_grads = backward_lowrank_grads_triton(
                    U.to(matmul_dtype),
                    Y.to(matmul_dtype),
                    P.to(matmul_dtype),
                    gamma,
                )
            except Exception:
                triton_grads = None

        if triton_grads is None:
            YtU = torch.bmm(Y_m.transpose(1, 2), U_m)
            PtU = torch.bmm(P_m.transpose(1, 2), U_m)
            grad_U = -gamma.expand(B, H, 1, 1).to(matmul_dtype).reshape(B * H, 1, 1) * (
                torch.bmm(P_m, YtU) + torch.bmm(Y_m, PtU)
            )
            grad_U = grad_U.view(B, H, N, r).to(U.dtype)
        else:
            grad_U, YtU, PtU = triton_grads

        grad_C = P.to(grad_output.dtype)

        if calc_dtype == torch.float64:
            grad_mu_bh = -(P.to(calc_dtype) * Y.to(calc_dtype)).sum(dim=(2, 3)).view(B, H, 1, 1)
        else:
            grad_mu_bh = -(P * Y).sum(dim=(2, 3), dtype=torch.float32).view(B, H, 1, 1)
        if YtU.dim() == 4:
            grad_gamma_bh = -(PtU.to(calc_dtype) * YtU.to(calc_dtype)).sum(dim=(2, 3)).view(B, H, 1, 1)
        else:
            grad_gamma_bh = -(PtU.to(calc_dtype) * YtU.to(calc_dtype)).sum(dim=(1, 2)).view(B, H, 1, 1)

        if mu.shape[0] == 1:
            grad_mu = grad_mu_bh.sum(dim=0, keepdim=True)
        else:
            grad_mu = grad_mu_bh
        if gamma.shape[0] == 1:
            grad_gamma = grad_gamma_bh.sum(dim=0, keepdim=True)
        else:
            grad_gamma = grad_gamma_bh

        return grad_U, grad_C, grad_mu.to(mu.dtype), grad_gamma.to(gamma.dtype), None, None


def lsso(
    U: torch.Tensor,
    C: torch.Tensor,
    mu: torch.Tensor,
    gamma: torch.Tensor,
    *,
    eye: torch.Tensor | None = None,
    no_global: bool = False,
    return_aux: bool = False,
    use_triton: bool = False,
    use_custom_backward: bool = True,
    use_triton_backward: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, LSSOAux]:
    """
    Functional LSSO core, analogous to an attention kernel.

    Args:
        U: low-rank relation features, [B, H, N, r].
        C: target state, [B, H, N, dh].
        mu: positive shift, broadcastable to [B, H, 1, 1] or [H].
        gamma: global strength, broadcastable to [B, H, 1, 1] or [H].
        eye: optional identity buffer shaped [1, 1, r, r].
        no_global: if true, returns only mu^-1 C.
        return_aux: if true, also returns tensors used for diagnostics.
        use_triton: if true, uses experimental inference-only Triton kernels
            when CUDA/no-grad/shape constraints allow it; otherwise falls back
            to the PyTorch implementation.
        use_custom_backward: if true, training uses an implicit custom backward
            for the solve instead of autograd through every Woodbury operation.
        use_triton_backward: if true, custom backward uses experimental Triton
            kernels for low-rank gradient products when supported.

    Returns:
        Y: solved token states, [B, H, N, dh].
    """
    B, H, N, r = U.shape
    dh = C.shape[-1]

    if mu.dim() == 1:
        mu = mu.view(1, H, 1, 1)
    if gamma.dim() == 1:
        gamma = gamma.view(1, H, 1, 1)

    inv_mu = mu.reciprocal()
    gamma_over_mu = gamma * inv_mu
    gamma_over_mu2 = gamma_over_mu * inv_mu

    if (
        use_custom_backward
        and torch.is_grad_enabled()
        and not no_global
        and not return_aux
        and not use_triton
        and (U.requires_grad or C.requires_grad or mu.requires_grad or gamma.requires_grad)
    ):
        return _LSSOAutograd.apply(U, C, mu, gamma, eye, use_triton_backward)

    local = None
    triton_ok = (
        use_triton
        and not torch.is_grad_enabled()
        and not U.requires_grad
        and not C.requires_grad
        and not mu.requires_grad
        and not gamma.requires_grad
    )

    U_bh = U.flatten(0, 1)
    C_bh = C.flatten(0, 1)
    Ut_bh = U_bh.transpose(1, 2)

    if no_global:
        local = inv_mu * C
        correction = torch.zeros_like(local)
        UtU = torch.bmm(Ut_bh, U_bh).view(B, H, r, r) if return_aux else None
        Y = local
    else:
        UtU = None
        UtC = None
        G = None
        if triton_ok:
            try:
                from .triton_kernels import fused_gram_system_utc_triton, fused_gram_utc_triton

                if return_aux:
                    fused = fused_gram_utc_triton(U, C)
                    if fused is not None:
                        UtU, UtC = fused
                else:
                    fused_system = fused_gram_system_utc_triton(U, C, gamma_over_mu)
                    if fused_system is not None:
                        G, UtC = fused_system
            except Exception:
                UtU = None
                UtC = None
                G = None

        if (G is None and UtU is None) or UtC is None:
            UtU = torch.bmm(Ut_bh, U_bh).view(B, H, r, r)
            UtC = torch.bmm(Ut_bh, C_bh).view(B, H, r, dh)

        if G is None:
            if eye is None:
                eye = torch.eye(r, device=U.device, dtype=U.dtype).view(1, 1, r, r)
            solve_dtype = torch.float64 if U.dtype == torch.float64 or C.dtype == torch.float64 else torch.float32
            G = eye.to(solve_dtype) + gamma_over_mu.to(solve_dtype) * UtU.to(solve_dtype)
        K = _cholesky_spd_solve(
            G.view(B * H, r, r),
            UtC.to(G.dtype).view(B * H, r, dh),
        ).to(U.dtype)

        Y = None
        if triton_ok:
            try:
                from .triton_kernels import correction_apply_triton

                Y = correction_apply_triton(U, C, K.view(B, H, r, dh), mu, gamma)
            except Exception:
                Y = None

        if Y is None:
            local = inv_mu * C
            UK = torch.bmm(U_bh, K).view(B, H, N, dh)
            correction = gamma_over_mu2 * UK
            Y = local - correction
        else:
            if return_aux:
                local = inv_mu * C
                correction = local - Y

    if return_aux:
        assert local is not None
        return Y, LSSOAux(UtU=UtU, local=local, correction=correction, mu=mu, gamma=gamma)
    return Y


class LSSO(nn.Module):
    """
    LSSO v1: Learnable Sylvester Solve Operator.

    Per head:
        (mu I + gamma U U^T) Y = C

    Woodbury:
        Y = mu^-1 C
            - gamma / mu^2 * U @ solve(I + gamma/mu * U^T U, U^T C)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        rank: int = 16,
        dropout: float = 0.0,
        eps: float = 1e-5,
        gamma_max: float = 0.1,
        theta_gamma_init: float = -6.0,
        no_global: bool = False,
        normalize_u: bool = True,
        use_triton: bool = False,
        use_custom_backward: bool = True,
        use_triton_backward: bool = False,
    ) -> None:
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rank = rank
        self.eps = eps
        self.gamma_max = gamma_max
        self.no_global = no_global
        self.normalize_u = normalize_u
        self.use_triton = use_triton
        self.use_custom_backward = use_custom_backward
        self.use_triton_backward = use_triton_backward

        self.uc_dim = num_heads * rank + dim
        self.w_uc = nn.Linear(dim, self.uc_dim, bias=False)
        self.w_o = nn.Linear(dim, dim, bias=False)
        self.register_buffer(
            "_eye",
            torch.eye(rank).view(1, 1, rank, rank),
            persistent=False,
        )

        self.theta_mu = nn.Parameter(torch.zeros(num_heads))
        self.theta_gamma = nn.Parameter(
            torch.full((num_heads,), float(theta_gamma_init), dtype=torch.float32)
        )

        self.dropout_p = dropout
        self.record_diagnostics = False
        self.prune_rank_keep: int | None = None
        self.last_diagnostics: LSSODiagnostics | None = None

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, D = x.shape
        H = self.num_heads
        dh = self.head_dim
        r = self.rank
        head_mask = None
        if valid_mask is not None:
            head_mask = valid_mask[:, None, :, None].to(dtype=x.dtype)

        UC = self.w_uc(x)
        U, C = UC.split((H * r, D), dim=-1)

        U = U.view(B, N, H, r).transpose(1, 2).contiguous()

        C = C.view(B, N, H, dh).transpose(1, 2).contiguous()

        if self.normalize_u:
            U = U * torch.rsqrt(torch.mean(U * U, dim=-1, keepdim=True) + self.eps)
        if head_mask is not None:
            U = U * head_mask
            C = C * head_mask

        solve_eye = self._eye
        if self.prune_rank_keep is not None and 0 < self.prune_rank_keep < r:
            keep = int(self.prune_rank_keep)
            scores = U.float().square().mean(dim=-2)
            indices = scores.topk(k=keep, dim=-1, largest=True, sorted=False).indices
            U = U.gather(-1, indices[:, :, None, :].expand(B, H, N, keep))
            solve_eye = None

        mu = F.softplus(self.theta_mu) + self.eps
        gamma = self.gamma_max * torch.sigmoid(self.theta_gamma)
        if self.no_global:
            gamma = torch.zeros_like(gamma)

        mu = mu.view(1, H, 1, 1)
        gamma = gamma.view(1, H, 1, 1)
        if self.record_diagnostics:
            Y, aux = lsso(
                U,
                C,
                mu,
                gamma,
                eye=solve_eye,
                no_global=self.no_global or self.gamma_max == 0.0,
                return_aux=True,
                use_triton=self.use_triton,
                use_custom_backward=self.use_custom_backward,
                use_triton_backward=self.use_triton_backward,
            )
        else:
            Y = lsso(
                U,
                C,
                mu,
                gamma,
                eye=solve_eye,
                no_global=self.no_global or self.gamma_max == 0.0,
                use_triton=self.use_triton,
                use_custom_backward=self.use_custom_backward,
                use_triton_backward=self.use_triton_backward,
            )
            aux = None

        if aux is not None and aux.UtU is not None:
            self.last_diagnostics = self._diagnostics(
                aux.UtU,
                aux.local,
                aux.correction,
                aux.mu,
                aux.gamma,
            )
        else:
            self.last_diagnostics = None

        Y = Y.transpose(1, 2).contiguous().view(B, N, D)
        Y = self.w_o(Y)
        if valid_mask is not None:
            Y = Y * valid_mask[:, :, None].to(dtype=Y.dtype)
        return Y

    def _diagnostics(
        self,
        UtU: torch.Tensor,
        local: torch.Tensor,
        correction: torch.Tensor,
        mu: torch.Tensor,
        gamma: torch.Tensor,
    ) -> LSSODiagnostics:
        with torch.no_grad():
            eigvals = torch.linalg.eigvalsh(UtU.float()).clamp_min(0.0)
            eig_sum = eigvals.sum(dim=-1)
            eig_sq_sum = (eigvals * eigvals).sum(dim=-1).clamp_min(self.eps)
            effective_rank = (eig_sum * eig_sum) / eig_sq_sum

            correction_norm = correction.float().norm(dim=(-2, -1))
            local_norm = local.float().norm(dim=(-2, -1)).clamp_min(self.eps)
            correction_ratio = correction_norm / local_norm

            gamma_over_mu = (gamma / mu).view(-1).detach().float().cpu()

        return LSSODiagnostics(
            gamma_over_mu=gamma_over_mu,
            effective_rank=effective_rank.detach().float().cpu(),
            correction_ratio=correction_ratio.detach().float().cpu(),
        )
