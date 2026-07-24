"""Historical fixed-gain/output-folding ablation; not imported by LSSO."""

import torch
import torch.nn as nn


@torch.no_grad()
def fold_fixed_gain_into_output(
    output: nn.Linear,
    gain: torch.Tensor,
    *,
    group_width: int,
) -> None:
    scale = gain.to(
        device=output.weight.device,
        dtype=output.weight.dtype,
    ).repeat_interleave(group_width)
    if scale.numel() != output.weight.shape[1]:
        raise ValueError("fixed-gain output-fold shape mismatch")
    output.weight.mul_(scale.unsqueeze(0))
