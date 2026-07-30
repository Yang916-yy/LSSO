from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import CoreMode, LSSOConfig
from .reference import (
    accretive_equilibrium_mix,
    accretive_generator,
    bounded_complement,
    qr_soft_frame,
    rank_rotary,
    tensor_core_linear,
    tensor_core_matmul,
    tf32_fp32_linear,
)


_CONTRACT_VERSION = 11
_ETA_INIT = 0.9
_ETA_INIT_RAW = math.atanh(_ETA_INIT)
_SUPPORTED_ACTIVATION_DTYPES = frozenset(
    (torch.float16, torch.float32, torch.float64)
)


class LSSO(nn.Module):
    r"""The accretive-equilibrium low-rank token mixer.

    Each head builds a QR soft frame P, compresses content into Z = P^T C,
    forms an accretive generator from static or cross-moment coordinates, and
    evaluates its reflected-resolvent equilibrium directly.

    Dynamic, static, and zero compact-core ownership share this one forward
    path. Rank-Rotary is the sole positional ablation.
    """

    def __init__(self, config: LSSOConfig) -> None:
        super().__init__()
        self.config = config
        self.w_bc = nn.Linear(
            config.dim,
            config.num_heads * config.rank + config.dim,
            bias=config.bias,
        )
        self.w_o = nn.Linear(config.dim, config.dim, bias=config.bias)

        if config.core_mode is CoreMode.DYNAMIC:
            self.core_base_raw = nn.Parameter(
                torch.zeros(config.num_heads, config.rank, config.rank)
            )
            self.core_drive_weight = nn.Parameter(
                torch.zeros(config.num_heads, config.head_dim, config.rank)
            )
        elif config.core_mode is CoreMode.STATIC:
            self.core_base_raw = nn.Parameter(
                torch.zeros(config.num_heads, config.rank, config.rank)
            )
            self.register_parameter("core_drive_weight", None)
        else:
            self.register_parameter("core_base_raw", None)
            self.register_parameter("core_drive_weight", None)

        self.eta_raw = nn.Parameter(
            torch.full(
                (config.num_heads,),
                _ETA_INIT_RAW,
                dtype=torch.float32,
            )
        )

    def complement(self) -> torch.Tensor:
        """Return the learned per-head complement."""

        return bounded_complement(self.eta_raw)

    def _contract_state(self) -> dict[str, object]:
        config = self.config
        return {
            "version": _CONTRACT_VERSION,
            "operator": "accretive_equilibrium",
            "dim": config.dim,
            "num_heads": config.num_heads,
            "rank": config.rank,
            "core_mode": config.core_mode.value,
            "rank_rotary": config.rank_rotary,
            "eta_parameterization": "per_head_interior_tanh",
            "numerics": "tf32-wbc-ieee-fgram-tc16-v6",
            "bias": config.bias,
        }

    def get_extra_state(self) -> dict[str, object]:
        """Persist the exact operator contract with tensor state."""

        return self._contract_state()

    def set_extra_state(self, state: object) -> None:
        """Reject checkpoints created for a different operator contract."""

        expected = self._contract_state()
        if not isinstance(state, dict):
            raise RuntimeError("LSSO checkpoint is missing its configuration contract")
        if state != expected:
            keys = sorted(set(state) | set(expected))
            mismatches = ", ".join(
                f"{key}: checkpoint={state.get(key)!r}, model={expected.get(key)!r}"
                for key in keys
                if state.get(key) != expected.get(key)
            )
            raise RuntimeError(f"incompatible LSSO checkpoint contract ({mismatches})")

    def _load_from_state_dict(
        self,
        state_dict: dict[str, object],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Require the semantic contract even for a non-strict tensor load."""

        contract_key = f"{prefix}_extra_state"
        if contract_key not in state_dict:
            error_msgs.append(
                "LSSO checkpoint is missing its configuration contract "
                f"({contract_key!r})"
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @staticmethod
    def _validate_mask(
        valid_mask: torch.Tensor | None,
        *,
        batch: int,
        length: int,
        device: torch.device,
    ) -> torch.Tensor:
        if valid_mask is None:
            return torch.ones(batch, length, dtype=torch.bool, device=device)
        if valid_mask.shape != (batch, length):
            raise ValueError(
                f"valid_mask must have shape {(batch, length)}, "
                f"got {tuple(valid_mask.shape)}"
            )
        if valid_mask.dtype != torch.bool:
            raise TypeError(
                f"valid_mask must have dtype torch.bool, got {valid_mask.dtype}"
            )
        return valid_mask.to(device=device)

    @staticmethod
    def _center_positions(
        position_ids: torch.Tensor | None,
        valid_mask: torch.Tensor | None,
        *,
        dtype: torch.dtype,
        all_valid: bool,
        batch: int | None = None,
        length: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if valid_mask is None:
            if not all_valid:
                raise ValueError("valid_mask is required when tokens are masked")
            if batch is None or length is None or device is None:
                raise ValueError(
                    "batch, length, and device are required without valid_mask"
                )
        else:
            batch, length = valid_mask.shape
            device = valid_mask.device

        if position_ids is None:
            positions = torch.arange(length, device=device, dtype=dtype)
            positions = positions.view(1, length).expand(batch, length)
        else:
            if position_ids.is_floating_point():
                if position_ids.dtype not in (torch.float32, torch.float64):
                    raise TypeError(
                        "floating position_ids must use torch.float32 or "
                        f"torch.float64, got {position_ids.dtype}"
                    )
                # Float64 coordinates retain their relative precision until
                # centering; converting a large absolute offset first loses it.
                position_dtype = (
                    torch.float64
                    if position_ids.dtype == torch.float64
                    else dtype
                )
                positions = position_ids.to(device=device, dtype=position_dtype)
            else:
                if position_ids.dtype == torch.bool or position_ids.is_complex():
                    raise TypeError(
                        "position_ids must use an integer dtype, torch.float32, "
                        f"or torch.float64, got {position_ids.dtype}"
                    )
                positions = position_ids.to(device=device, dtype=torch.int64)
            if positions.ndim == 1:
                if positions.numel() != length:
                    raise ValueError(
                        f"position_ids length {positions.numel()} must match N={length}"
                    )
                positions = positions.view(1, length).expand(batch, length)
            elif positions.shape != (batch, length):
                raise ValueError(
                    "position_ids must be None, [N], or "
                    f"[B, N]={batch, length}; got {tuple(positions.shape)}"
                )

        if all_valid:
            centered = positions - positions[:, :1]
            if centered.dtype == torch.int64:
                centered = centered.to(dtype=dtype)
            centered = centered - centered.mean(dim=-1, keepdim=True)
            return centered.to(dtype=dtype)

        assert valid_mask is not None
        first_valid = valid_mask.to(dtype=torch.int64).argmax(dim=-1, keepdim=True)
        anchor = positions.gather(dim=1, index=first_valid)
        centered = positions - anchor
        if centered.dtype == torch.int64:
            centered = centered.to(dtype=dtype)
        weights = valid_mask.to(dtype=centered.dtype)
        count = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        safe_positions = torch.where(
            valid_mask, centered, torch.zeros_like(centered)
        )
        mean = (safe_positions * weights).sum(dim=-1, keepdim=True) / count
        return torch.where(
            valid_mask, centered - mean, torch.zeros_like(centered)
        ).to(dtype=dtype)

    def _compact_coordinates(
        self,
        compact_state: torch.Tensor,
        valid_count: torch.Tensor,
    ) -> torch.Tensor | None:
        config = self.config
        if config.core_mode is CoreMode.DYNAMIC:
            if self.core_base_raw is None or self.core_drive_weight is None:
                raise RuntimeError("dynamic compact parameters are missing")
            dynamic_coordinates = tensor_core_matmul(
                compact_state,
                self.core_drive_weight.to(dtype=compact_state.dtype),
            )
            base = self.core_base_raw.to(dtype=compact_state.dtype).unsqueeze(0)
            return base + dynamic_coordinates / valid_count.sqrt().view(
                -1, 1, 1, 1
            )

        if config.core_mode is CoreMode.STATIC:
            if self.core_base_raw is None:
                raise RuntimeError("static compact parameters are missing")
            return self.core_base_raw.to(dtype=compact_state.dtype)

        return None

    def _forward_cuda(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run the explicit strict native implementation of the default operator."""

        config = self.config
        if (
            config.core_mode is not CoreMode.DYNAMIC
            or not config.rank_rotary
            or config.rank not in (16, 32, 48, 64)
        ):
            raise ValueError(
                "implementation='cuda' requires core_mode='dynamic', "
                "rank_rotary=True, and rank in {16, 32, 48, 64}"
            )
        if position_ids is not None and position_ids.requires_grad:
            raise ValueError(
                "implementation='cuda' does not support gradients for position_ids"
            )
        if x.device.type != "cuda":
            raise ValueError("implementation='cuda' requires x to be a CUDA tensor")
        if x.dtype not in (torch.float16, torch.float32):
            raise TypeError(
                "implementation='cuda' supports x with dtype torch.float16 or "
                "torch.float32; "
                f"got {x.dtype}"
            )

        if self.core_base_raw is None or self.core_drive_weight is None:
            raise RuntimeError("dynamic compact parameters are missing")

        for name, parameter in self.named_parameters():
            if parameter.device != x.device:
                raise RuntimeError(
                    f"implementation='cuda' requires {name} on {x.device}, "
                    f"got {parameter.device}"
                )
        for name, parameter in (
            ("w_bc.weight", self.w_bc.weight),
            ("w_bc.bias", self.w_bc.bias),
            ("w_o.weight", self.w_o.weight),
            ("w_o.bias", self.w_o.bias),
            ("core_base_raw", self.core_base_raw),
            ("core_drive_weight", self.core_drive_weight),
            ("eta_raw", self.eta_raw),
        ):
            if parameter is None:
                continue
            if parameter.dtype != torch.float32:
                raise TypeError(
                    f"implementation='cuda' requires {name} to use torch.float32, "
                    f"got {parameter.dtype}"
                )
            if not parameter.is_contiguous():
                raise RuntimeError(
                    f"implementation='cuda' requires contiguous {name}"
                )

        batch, length, _dim = x.shape
        all_valid = valid_mask is None
        mask = (
            None
            if all_valid
            else self._validate_mask(
                valid_mask,
                batch=batch,
                length=length,
                device=x.device,
            )
        )
        centered_positions: torch.Tensor | None = None
        if config.rank_rotary and (valid_mask is not None or position_ids is not None):
            centered = self._center_positions(
                position_ids,
                mask,
                dtype=torch.float32,
                all_valid=all_valid,
                batch=batch,
                length=length,
                device=x.device,
            )
            if position_ids is None or position_ids.ndim == 1:
                centered_positions = (
                    centered[0].contiguous()
                    if all_valid
                    else centered.contiguous()
                )
            else:
                centered_positions = centered.contiguous()
        if all_valid:
            valid_counts = None
        else:
            assert mask is not None
            valid_counts = (
                mask.sum(dim=-1).to(dtype=torch.float32).clamp_min(1.0).contiguous()
            )

        # The native mixer consumes the FP32 packed coordinates directly.
        if all_valid:
            safe_x = x
        else:
            assert mask is not None
            safe_x = torch.where(mask[:, :, None], x, torch.zeros_like(x))
        projected = tf32_fp32_linear(safe_x, self.w_bc.weight, self.w_bc.bias)
        if not all_valid:
            assert mask is not None
            projected = torch.where(
                mask[:, :, None], projected, torch.zeros_like(projected)
            )
        if not projected.is_contiguous():
            raise RuntimeError(
                "implementation='cuda' requires w_bc to produce contiguous "
                "projected coordinates"
            )
        from . import cuda as cuda_backend

        mixed = cuda_backend.fast_mix(
            projected,
            self.core_base_raw,
            self.core_drive_weight,
            self.eta_raw,
            centered_positions,
            valid_counts,
        )
        output = tensor_core_linear(mixed, self.w_o.weight, self.w_o.bias)
        if not all_valid:
            assert mask is not None
            output = torch.where(mask[:, :, None], output, torch.zeros_like(output))
        return output.to(dtype=x.dtype)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        *,
        implementation: str = "reference",
    ) -> torch.Tensor:
        config = self.config
        if x.ndim != 3 or x.shape[-1] != config.dim:
            raise ValueError(
                f"x must have shape [B, N, {config.dim}], got {tuple(x.shape)}"
            )
        if not x.is_floating_point():
            raise TypeError("x must be a floating-point tensor")
        if x.dtype not in _SUPPORTED_ACTIVATION_DTYPES:
            raise TypeError(
                f"LSSO does not support x with dtype {x.dtype}; use "
                "torch.float16, torch.float32, or torch.float64"
            )

        batch, length, _dim = x.shape
        if batch == 0:
            raise ValueError("batch size must be positive")
        if length == 0:
            raise ValueError("sequence length must be positive")

        if implementation == "cuda":
            return self._forward_cuda(x, valid_mask, position_ids)
        if implementation != "reference":
            raise ValueError(
                "implementation must be 'reference' or 'cuda', "
                f"got {implementation!r}"
            )

        all_valid = valid_mask is None
        mask = (
            None
            if all_valid
            else self._validate_mask(
                valid_mask,
                batch=batch,
                length=length,
                device=x.device,
            )
        )

        if all_valid:
            safe_x = x
        else:
            assert mask is not None
            safe_x = torch.where(mask[:, :, None], x, torch.zeros_like(x))
        projected = tf32_fp32_linear(
            safe_x,
            self.w_bc.weight,
            self.w_bc.bias,
        )
        relation, content = projected.split(
            (config.num_heads * config.rank, config.dim), dim=-1
        )
        relation = relation.view(
            batch, length, config.num_heads, config.rank
        ).transpose(1, 2)
        content = content.view(
            batch, length, config.num_heads, config.head_dim
        ).transpose(1, 2)
        if not all_valid:
            assert mask is not None
            active = mask[:, None, :, None]
            relation = torch.where(active, relation, torch.zeros_like(relation))
            content = torch.where(active, content, torch.zeros_like(content))

        calc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            relation = relation.to(dtype=calc_dtype)
            content = content.to(dtype=calc_dtype)
            if config.rank_rotary:
                centered = self._center_positions(
                    position_ids,
                    mask,
                    dtype=calc_dtype,
                    all_valid=all_valid,
                    batch=batch,
                    length=length,
                    device=x.device,
                )
                relation = rank_rotary(relation, centered)

            if all_valid:
                valid_count = torch.full(
                    (batch,),
                    float(length),
                    dtype=calc_dtype,
                    device=x.device,
                )
            else:
                assert mask is not None
                valid_count = mask.sum(dim=-1).to(dtype=calc_dtype).clamp_min(1.0)
            relation = relation / valid_count.sqrt().view(batch, 1, 1, 1)
            frame = qr_soft_frame(relation)
            compact_state = tensor_core_matmul(frame.mT, content)
            coordinates = self._compact_coordinates(compact_state, valid_count)
            generator = (
                None
                if coordinates is None
                else accretive_generator(coordinates)
            )
            eta = self.complement().to(device=x.device, dtype=calc_dtype)
            output = accretive_equilibrium_mix(
                frame,
                generator,
                compact_state,
                content,
                eta,
            )

        output = output.transpose(1, 2).contiguous().view(
            batch, length, config.dim
        )
        output = tensor_core_linear(output, self.w_o.weight, self.w_o.bias)
        output = output.to(dtype=x.dtype)
        if all_valid:
            return output
        assert mask is not None
        return torch.where(mask[:, :, None], output, torch.zeros_like(output))

    def extra_repr(self) -> str:
        config = self.config
        return (
            f"dim={config.dim}, num_heads={config.num_heads}, rank={config.rank}, "
            f"core_mode={config.core_mode.value}, "
            f"rank_rotary={config.rank_rotary}, eta=per-head-interior-tanh"
        )


__all__ = ["LSSO"]
