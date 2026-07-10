from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from lsso.modules import LSSODiagnostics, length_normalize_basis


class BertLSSOSelfAttention(nn.Module):
    """
    Drop-in replacement for HuggingFace BertSelfAttention.

    It keeps BERT's BertSelfOutput projection outside this module, so this class
    intentionally does not apply a Wo projection. The replaced block is:

        BertSelfAttention(QKV attention) -> BertSelfOutput(dense + dropout + LN)

    and becomes:

        BertLSSOSelfAttention(low-rank solve) -> BertSelfOutput(dense + dropout + LN)
    """

    def __init__(
        self,
        config,
        rank: int = 16,
        gamma_max: float = 1.2,
        theta_gamma_init: float = 0.5,
        eps: float = 1e-5,
        no_global: bool = False,
        length_normalize: bool = True,
        length_reference: float = 1.0,
    ) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = config.hidden_size
        self.rank = rank
        self.gamma_max = gamma_max
        self.theta_gamma_init = theta_gamma_init
        self.eps = eps
        self.no_global = no_global
        self.length_normalize = length_normalize
        if length_reference <= 0:
            raise ValueError(f"length_reference must be positive, got {length_reference}")
        self.length_reference = float(length_reference)
        self.is_decoder = getattr(config, "is_decoder", False)

        self.w_u = nn.Linear(config.hidden_size, config.num_attention_heads * rank, bias=False)
        self.w_c = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.theta_mu = nn.Parameter(torch.zeros(config.num_attention_heads))
        self.theta_gamma = nn.Parameter(torch.full((config.num_attention_heads,), theta_gamma_init))
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.last_diagnostics: LSSODiagnostics | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_value=None,
        past_key_values=None,
        output_attentions: bool = False,
        **kwargs,
    ):
        if encoder_hidden_states is not None:
            raise NotImplementedError("BertLSSOSelfAttention only supports encoder self-attention.")
        if past_key_value is not None or past_key_values is not None:
            raise NotImplementedError("BertLSSOSelfAttention does not support decoder KV cache.")

        valid_mask = self._valid_mask(attention_mask, hidden_states)
        context_layer = self._solve(hidden_states, valid_mask=valid_mask)
        if head_mask is not None:
            context_layer = self._apply_head_mask(context_layer, head_mask)

        return context_layer, None

    def _solve(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        B, N, D = x.shape
        H = self.num_attention_heads
        dh = self.attention_head_size
        r = self.rank

        U = self.w_u(x).view(B, N, H, r).transpose(1, 2).contiguous()
        C = self.w_c(x).view(B, N, H, dh).transpose(1, 2).contiguous()

        U = U / torch.sqrt(torch.mean(U * U, dim=-1, keepdim=True) + self.eps)
        head_valid = None
        if valid_mask is not None:
            head_valid = valid_mask[:, None, :, None].to(dtype=x.dtype)
            U = U * head_valid
            C = C * head_valid
        if self.length_normalize:
            U = length_normalize_basis(
                U,
                valid_mask,
                reference_length=self.length_reference,
            )

        mu = F.softplus(self.theta_mu) + self.eps
        gamma = self.gamma_max * torch.sigmoid(self.theta_gamma)
        if self.no_global:
            gamma = torch.zeros_like(gamma)

        mu = mu.view(1, H, 1, 1)
        gamma = gamma.view(1, H, 1, 1)
        local = (1.0 / mu) * C

        UtU = torch.matmul(U.transpose(-2, -1), U)
        if self.no_global or self.gamma_max == 0.0:
            Y = local
            correction = torch.zeros_like(local)
        else:
            UtC = torch.matmul(U.transpose(-2, -1), C)
            eye = torch.eye(r, device=x.device, dtype=x.dtype).view(1, 1, r, r)
            G = eye + (gamma / mu) * UtU
            L = torch.linalg.cholesky_ex(G.float(), check_errors=False).L
            K = torch.cholesky_solve(UtC.float(), L).to(x.dtype)
            correction = (gamma / (mu * mu)) * torch.matmul(U, K)
            Y = local - correction

        self.last_diagnostics = self._diagnostics(UtU, local, correction, mu, gamma)
        Y = Y.transpose(1, 2).contiguous().view(B, N, D)
        if valid_mask is not None:
            Y = Y * valid_mask[:, :, None].to(dtype=Y.dtype)
        return self.dropout(Y)

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

    def _valid_mask(
        self,
        attention_mask: torch.Tensor | None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None

        mask = attention_mask
        while mask.dim() > 2:
            mask = mask.squeeze(1)
        if mask.dim() != 2:
            return None

        if mask.dtype == torch.bool:
            # Extended BERT masks usually use True for valid before expansion,
            # but bool additive masks may use True for masked. Prefer the common
            # [B, N] attention_mask convention when all rows contain a CLS token.
            return mask.to(device=hidden_states.device)

        return (mask >= 0).to(device=hidden_states.device)

    def _apply_head_mask(self, context_layer: torch.Tensor, head_mask: torch.Tensor) -> torch.Tensor:
        B, N, D = context_layer.shape
        H = self.num_attention_heads
        dh = self.attention_head_size
        mask = head_mask
        while mask.dim() > 1:
            mask = mask.squeeze(0)
        mask = mask.view(1, 1, H, 1).to(device=context_layer.device, dtype=context_layer.dtype)
        context = context_layer.view(B, N, H, dh) * mask
        return context.view(B, N, D)


@dataclass
class BertLSSOConfig:
    rank: int = 16
    gamma_max: float = 1.2
    theta_gamma_init: float = 0.5
    no_global: bool = False


def replace_bert_self_attention_with_lsso(
    model: nn.Module,
    rank: int = 16,
    gamma_max: float = 1.2,
    theta_gamma_init: float = 0.5,
    no_global: bool = False,
) -> nn.Module:
    """
    Replace every encoder self-attention module in a HuggingFace BERT model.

    Works with BertModel/BertForSequenceClassification-style objects that expose
    `model.bert.encoder.layer` or `model.encoder.layer`.
    """

    encoder = _get_bert_encoder(model)
    config = getattr(model, "config")
    for layer in encoder.layer:
        layer.attention.self = BertLSSOSelfAttention(
            config=config,
            rank=rank,
            gamma_max=gamma_max,
            theta_gamma_init=theta_gamma_init,
            no_global=no_global,
        )
    return model


def iter_bert_lsso_layers(model: nn.Module) -> list[BertLSSOSelfAttention]:
    layers = []
    for module in model.modules():
        if isinstance(module, BertLSSOSelfAttention):
            layers.append(module)
    return layers


def _get_bert_encoder(model: nn.Module):
    if hasattr(model, "bert") and hasattr(model.bert, "encoder"):
        return model.bert.encoder
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return model.encoder
    raise ValueError("Could not find a BERT encoder on the provided model.")
