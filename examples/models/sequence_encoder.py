from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lsso import MixerAdapter


class GatedDilatedMotifBlock(nn.Module):
    """Padding-safe local DNA block with independent feature and gate paths."""

    def __init__(
        self,
        dim: int,
        *,
        kernel_size: int = 9,
        dilation: int = 1,
        dropout: float = 0.0,
        layer_scale_init: float = 1e-3,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if dilation <= 0:
            raise ValueError("dilation must be positive")
        padding = dilation * (kernel_size - 1) // 2
        self.norm = nn.LayerNorm(dim)
        self.feature_depthwise = nn.Conv1d(
            dim, dim, kernel_size, padding=padding, dilation=dilation,
            groups=dim, bias=False,
        )
        self.feature_pointwise = nn.Conv1d(dim, dim, 1)
        self.gate_depthwise = nn.Conv1d(
            dim, dim, kernel_size, padding=padding, dilation=dilation,
            groups=dim, bias=False,
        )
        self.gate_pointwise = nn.Conv1d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_scale = nn.Parameter(torch.full((dim,), float(layer_scale_init)))

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        mask = valid_mask.unsqueeze(-1).to(device=x.device, dtype=x.dtype)
        local = (self.norm(x) * mask).transpose(1, 2)
        feature = self.feature_pointwise(self.feature_depthwise(local))
        gate = self.gate_pointwise(self.gate_depthwise(local))
        update = (F.gelu(feature) * torch.sigmoid(gate)).transpose(1, 2)
        x = x + self.dropout(update) * self.layer_scale
        return x * mask


class SequenceMixerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        mixer: str,
        rank: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        rotary_1d: bool = False,
        gain_init: float = 1.0,
        alpha_init: float = 1.2,
        alpha_max: float = 3.0,
    ) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = MixerAdapter(
            dim,
            num_heads,
            mixer,
            rank=rank,
            dropout=dropout,
            rotary_1d=rotary_1d,
            gain_init=gain_init,
            alpha_init=alpha_init,
            alpha_max=alpha_max,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        padding_ratio_hint: float | None = None,
    ) -> torch.Tensor:
        x = x + self.mixer(
            self.norm1(x),
            valid_mask=valid_mask,
            padding_ratio_hint=padding_ratio_hint,
        )
        x = x + self.mlp(self.norm2(x))
        return x * valid_mask.unsqueeze(-1).to(x.dtype)


class SequenceMixerEncoder(nn.Module):
    """Small non-BERT bidirectional encoder shared by auxiliary tasks.

    Learned absolute embeddings supply task-level position capacity. RRLSSO
    independently applies its fixed 1-D rank rotary transform inside the solve.
    """

    def __init__(
        self,
        vocab_size: int,
        *,
        max_length: int,
        pad_token_id: int,
        dim: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        mixer: str = "rrlsso",
        rank: int = 32,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        embedding_dropout: float | None = None,
        projection_dim: int | None = None,
        position_rank: int = 0,
        pooling: str = "mean",
        local_motif_kernel: int = 0,
        local_motif_dilations: tuple[int, ...] = (),
        local_motif_layer_scale: float = 1e-3,
        gain_init: float = 1.0,
        alpha_init: float = 1.2,
        alpha_max: float = 3.0,
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "max", "meanmax"}:
            raise ValueError(f"unsupported sequence pooling: {pooling}")
        if local_motif_kernel < 0 or (
            local_motif_kernel and local_motif_kernel % 2 == 0
        ):
            raise ValueError("local_motif_kernel must be zero or a positive odd integer")
        if local_motif_kernel and local_motif_dilations:
            raise ValueError(
                "local_motif_kernel and local_motif_dilations select different local stems"
            )
        if any(dilation <= 0 for dilation in local_motif_dilations):
            raise ValueError("local_motif_dilations must contain only positive integers")
        self.max_length = int(max_length)
        self.pad_token_id = int(pad_token_id)
        self.pooling = pooling
        self.token_embedding = nn.Embedding(vocab_size, dim, padding_idx=pad_token_id)
        self.local_motif_kernel = int(local_motif_kernel)
        self.local_motif_stem = (
            nn.Sequential(
                nn.Conv1d(
                    dim,
                    dim,
                    kernel_size=self.local_motif_kernel,
                    padding=self.local_motif_kernel // 2,
                    groups=dim,
                    bias=False,
                ),
                nn.GELU(),
                nn.Conv1d(dim, dim, kernel_size=1, bias=False),
                nn.Dropout(dropout),
            )
            if self.local_motif_kernel
            else None
        )
        self.local_motif_dilations = tuple(int(value) for value in local_motif_dilations)
        self.local_motif_blocks = nn.ModuleList(
            GatedDilatedMotifBlock(
                dim,
                kernel_size=9,
                dilation=dilation,
                dropout=dropout,
                layer_scale_init=local_motif_layer_scale,
            )
            for dilation in self.local_motif_dilations
        )
        self.position_rank = (
            int(position_rank) if 0 < position_rank < min(max_length, dim) else 0
        )
        position_dim = self.position_rank or dim
        self.position_embedding = nn.Parameter(torch.zeros(1, max_length, position_dim))
        self.position_projection = (
            nn.Linear(position_dim, dim, bias=False) if self.position_rank else None
        )
        self.embedding_dropout = nn.Dropout(
            dropout if embedding_dropout is None else embedding_dropout
        )
        self.blocks = nn.ModuleList(
            SequenceMixerBlock(
                dim,
                num_heads,
                mixer=mixer,
                rank=rank,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                rotary_1d=True,
                gain_init=gain_init,
                alpha_init=alpha_init,
                alpha_max=alpha_max,
            )
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(dim)
        self.pool_projection = (
            nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU())
            if pooling == "meanmax"
            else None
        )
        output_dim = projection_dim or dim
        self.projection = nn.Linear(dim, output_dim, bias=False)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        if self.position_projection is not None:
            nn.init.trunc_normal_(self.position_projection.weight, std=0.02)

    def _positions(self, length: int) -> torch.Tensor:
        positions = self.position_embedding[:, :length]
        return self.position_projection(positions) if self.position_projection is not None else positions

    def _pool(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        weights = valid.unsqueeze(-1).to(x.dtype)
        mean = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        if self.pooling == "mean":
            return mean
        maximum = x.masked_fill(~valid.unsqueeze(-1), torch.finfo(x.dtype).min).amax(dim=1)
        maximum = torch.where(valid.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
        if self.pooling == "max":
            return maximum
        return self.pool_projection(torch.cat((mean, maximum), dim=-1))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        padding_ratio_hint: float | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, length]")
        if input_ids.shape[1] > self.max_length:
            raise ValueError(
                f"sequence length {input_ids.shape[1]} exceeds max_length={self.max_length}"
            )
        valid = input_ids.ne(self.pad_token_id) if attention_mask is None else attention_mask.bool()
        x = self.token_embedding(input_ids)
        x = x * valid.unsqueeze(-1).to(x.dtype)
        if self.local_motif_stem is not None:
            local = self.local_motif_stem(x.transpose(1, 2)).transpose(1, 2)
            x = x + local * valid.unsqueeze(-1).to(local.dtype)
        x = self.embedding_dropout(x + self._positions(input_ids.shape[1]))
        x = x * valid.unsqueeze(-1).to(x.dtype)
        local_cursor = 0
        local_base, local_extra = divmod(len(self.local_motif_blocks), len(self.blocks))
        for block_index, block in enumerate(self.blocks):
            local_count = local_base + (1 if block_index < local_extra else 0)
            for local_block in self.local_motif_blocks[
                local_cursor : local_cursor + local_count
            ]:
                x = local_block(x, valid)
            local_cursor += local_count
            x = block(x, valid, padding_ratio_hint=padding_ratio_hint)
        x = self.norm(x)
        return self.projection(self._pool(x, valid))

    def encode_normalized(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return F.normalize(self(input_ids, attention_mask), dim=-1)


class ProteinFitnessModel(nn.Module):
    def __init__(self, encoder: SequenceMixerEncoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.projection.out_features, 1)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.head(self.encoder(input_ids, attention_mask)).squeeze(-1)


class MultiScaleTemporalStem(nn.Module):
    """Padding-safe local temporal context before the global mixer."""

    def __init__(
        self,
        dim: int,
        kernels: tuple[int, ...],
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not kernels or any(kernel < 1 or kernel % 2 == 0 for kernel in kernels):
            raise ValueError("local stem kernels must be non-empty positive odd integers")
        self.kernels = tuple(int(kernel) for kernel in kernels)
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.ModuleList(
            nn.Conv1d(
                dim,
                dim,
                kernel,
                padding=kernel // 2,
                groups=dim,
                bias=False,
            )
            for kernel in self.kernels
        )
        self.pointwise = nn.Conv1d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        weights = valid.unsqueeze(-1).to(x.dtype)
        local_input = (self.norm(x) * weights).transpose(1, 2)
        local = sum(branch(local_input) for branch in self.depthwise) / len(self.depthwise)
        local = self.pointwise(F.gelu(local)).transpose(1, 2)
        return (x + self.dropout(local)) * weights


class SpatialLocalBlock(nn.Module):
    """A lightweight 2-D local branch for flattened image sequences.

    The depthwise convolution supplies the sparse, high-rank grid adjacency
    that a low-rank global token mixer should not have to approximate.  The
    pointwise projection only mixes channels; global communication remains in
    the following mixer block.
    """

    def __init__(
        self,
        dim: int,
        *,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.0,
        layer_scale_init: float = 1e-3,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if dilation <= 0:
            raise ValueError("dilation must be positive")
        padding = dilation * (kernel_size - 1) // 2
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv2d(
            dim,
            dim,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=dim,
            bias=False,
        )
        self.pointwise = nn.Conv2d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_scale = nn.Parameter(torch.full((dim,), float(layer_scale_init)))

    def forward(
        self,
        x: torch.Tensor,
        valid: torch.Tensor,
        spatial_shape: tuple[int, int],
    ) -> torch.Tensor:
        batch, length, dim = x.shape
        height, width = spatial_shape
        if length != height * width:
            raise ValueError("flattened sequence length does not match spatial_shape")
        mask = valid.unsqueeze(-1).to(dtype=x.dtype, device=x.device)
        local = (self.norm(x) * mask).reshape(batch, height, width, dim)
        local = local.permute(0, 3, 1, 2).contiguous()
        local = self.pointwise(F.gelu(self.depthwise(local)))
        local = local.permute(0, 2, 3, 1).reshape(batch, length, dim)
        return (x + self.dropout(local) * self.layer_scale) * mask


class SequenceValueEncoder(nn.Module):
    """Bidirectional sequence encoder for continuous per-timestep features."""

    def __init__(
        self,
        input_dim: int,
        *,
        max_length: int,
        dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mixer: str = "rrlsso",
        rank: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        projection_dim: int | None = None,
        position_rank: int = 0,
        pooling: str = "mean",
        local_stem_kernels: tuple[int, ...] = (),
        spatial_shape: tuple[int, int] | None = None,
        local_spatial_kernel: int = 0,
        local_spatial_dilations: tuple[int, ...] = (),
        local_spatial_layer_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "max", "meanmax"}:
            raise ValueError(f"unsupported sequence pooling: {pooling}")
        self.max_length = int(max_length)
        if spatial_shape is not None:
            spatial_shape = tuple(int(value) for value in spatial_shape)
            if len(spatial_shape) != 2 or min(spatial_shape) <= 0:
                raise ValueError("spatial_shape must contain two positive dimensions")
            if spatial_shape[0] * spatial_shape[1] != self.max_length:
                raise ValueError(
                    f"spatial_shape={spatial_shape} does not match max_length={max_length}"
                )
        self.spatial_shape = spatial_shape
        if local_spatial_kernel < 0 or (
            local_spatial_kernel and local_spatial_kernel % 2 == 0
        ):
            raise ValueError(
                "local_spatial_kernel must be zero or a positive odd integer"
            )
        if local_spatial_dilations and spatial_shape is None:
            raise ValueError("local spatial blocks require spatial_shape")
        if local_spatial_dilations and not local_spatial_kernel:
            raise ValueError(
                "local_spatial_kernel must be positive when dilations are configured"
            )
        if len(local_spatial_dilations) not in {0, depth}:
            raise ValueError("local_spatial_dilations must be empty or match depth")
        if any(dilation <= 0 for dilation in local_spatial_dilations):
            raise ValueError("local_spatial_dilations must contain positive integers")
        self.pooling = pooling
        self.input_projection = nn.Linear(input_dim, dim)
        self.local_temporal_stem = (
            MultiScaleTemporalStem(dim, tuple(local_stem_kernels), dropout=dropout)
            if local_stem_kernels
            else None
        )
        self.position_rank = (
            int(position_rank) if 0 < position_rank < min(max_length, dim) else 0
        )
        position_dim = self.position_rank or dim
        self.position_embedding = nn.Parameter(torch.zeros(1, max_length, position_dim))
        self.position_projection = (
            nn.Linear(position_dim, dim, bias=False) if self.position_rank else None
        )
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            SequenceMixerBlock(
                dim,
                num_heads,
                mixer=mixer,
                rank=rank,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                rotary_1d=True,
            )
            for _ in range(depth)
        )
        self.local_spatial_blocks = nn.ModuleList(
            SpatialLocalBlock(
                dim,
                kernel_size=local_spatial_kernel,
                dilation=int(dilation),
                dropout=dropout,
                layer_scale_init=local_spatial_layer_scale,
            )
            for dilation in local_spatial_dilations
        )
        self.norm = nn.LayerNorm(dim)
        self.pool_projection = (
            nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU())
            if pooling == "meanmax"
            else None
        )
        output_dim = projection_dim or dim
        self.projection = nn.Linear(dim, output_dim, bias=False)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        if self.position_projection is not None:
            nn.init.trunc_normal_(self.position_projection.weight, std=0.02)

    def _positions(self, length: int) -> torch.Tensor:
        positions = self.position_embedding[:, :length]
        return self.position_projection(positions) if self.position_projection is not None else positions

    def _pool(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        weights = valid.unsqueeze(-1).to(x.dtype)
        mean = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        if self.pooling == "mean":
            return mean
        maximum = x.masked_fill(~valid.unsqueeze(-1), torch.finfo(x.dtype).min).amax(dim=1)
        maximum = torch.where(valid.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
        if self.pooling == "max":
            return maximum
        return self.pool_projection(torch.cat((mean, maximum), dim=-1))

    def forward(
        self,
        values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        padding_ratio_hint: float | None = None,
    ) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError("values must have shape [batch, length, channels]")
        if values.shape[1] > self.max_length:
            raise ValueError(
                f"sequence length {values.shape[1]} exceeds max_length={self.max_length}"
            )
        if self.spatial_shape is not None and values.shape[1] != self.max_length:
            raise ValueError(
                "spatial inputs must contain the complete grid: "
                f"expected {self.max_length} values, got {values.shape[1]}"
            )
        if attention_mask is None:
            valid = torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
        else:
            valid = attention_mask.bool()
            if valid.shape != values.shape[:2]:
                raise ValueError("attention_mask must have shape [batch, length]")
        x = self.input_projection(values)
        x = x * valid.unsqueeze(-1).to(x.dtype)
        if self.local_temporal_stem is not None:
            x = self.local_temporal_stem(x, valid)
        x = self.embedding_dropout(x + self._positions(values.shape[1]))
        x = x * valid.unsqueeze(-1).to(x.dtype)
        for block_index, block in enumerate(self.blocks):
            if self.local_spatial_blocks:
                x = self.local_spatial_blocks[block_index](x, valid, self.spatial_shape)
            x = block(
                x,
                valid,
                padding_ratio_hint=padding_ratio_hint,
            )
        x = self.norm(x)
        return self.projection(self._pool(x, valid))


class SequenceClassifier(nn.Module):
    """Classification head shared by token and continuous sequence encoders."""

    def __init__(self, encoder: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        output_dim = encoder.projection.out_features
        self.head = nn.Linear(output_dim, num_classes)

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        padding_ratio_hint: float | None = None,
    ) -> torch.Tensor:
        return self.head(
            self.encoder(
                inputs,
                attention_mask,
                padding_ratio_hint=padding_ratio_hint,
            )
        )


class ReverseComplementSequenceClassifier(SequenceClassifier):
    """DNA classifier with reproducible on-device reverse-complement handling.

    Training augmentation draws from PyTorch's checkpointed RNG. Evaluation can
    optionally average the original and reverse-complement logits. Right padding
    remains on the right, so absolute position indices retain their semantics.
    """

    def __init__(
        self,
        encoder: SequenceMixerEncoder,
        num_classes: int,
        *,
        complement_ids: torch.Tensor,
        reverse_complement_probability: float = 0.0,
        reverse_complement_eval: bool = False,
        mutation_probability: float = 0.0,
        mutation_stop_epoch: int = 0,
    ) -> None:
        super().__init__(encoder, num_classes)
        if not 0.0 <= reverse_complement_probability <= 1.0:
            raise ValueError("reverse_complement_probability must be in [0, 1]")
        if complement_ids.ndim != 1:
            raise ValueError("complement_ids must be a one-dimensional lookup table")
        self.reverse_complement_probability = float(reverse_complement_probability)
        self.reverse_complement_eval = bool(reverse_complement_eval)
        if not 0.0 <= mutation_probability <= 1.0:
            raise ValueError("mutation_probability must be in [0, 1]")
        self.mutation_probability = float(mutation_probability)
        self.mutation_stop_epoch = int(mutation_stop_epoch)
        self.augmentation_epoch = 0
        self.register_buffer("complement_ids", complement_ids.long(), persistent=True)

    def set_augmentation_epoch(self, epoch: int) -> None:
        self.augmentation_epoch = int(epoch)

    def mutate(self, inputs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if (
            self.mutation_probability <= 0.0
            or self.augmentation_epoch >= self.mutation_stop_epoch
        ):
            return inputs
        # NucleotideTokenizer assigns contiguous ids 2..5 to A/C/G/T. N,
        # unknown, and padding are deliberately left untouched.
        canonical = attention_mask & inputs.ge(2) & inputs.le(5)
        selected = canonical & (
            torch.rand(inputs.shape, device=inputs.device) < self.mutation_probability
        )
        offset = torch.randint(1, 4, inputs.shape, device=inputs.device)
        replacements = 2 + (inputs.long() - 2 + offset) % 4
        return torch.where(selected, replacements.to(inputs.dtype), inputs)

    def reverse_complement(
        self, inputs: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = inputs.ne(self.encoder.pad_token_id) if attention_mask is None else attention_mask.bool()
        length = inputs.shape[1]
        positions = torch.arange(length, device=inputs.device).unsqueeze(0)
        lengths = valid.sum(dim=1, keepdim=True)
        reversed_positions = (lengths - 1 - positions).clamp_min(0)
        source = torch.where(valid, reversed_positions, positions).long()
        reversed_inputs = inputs.gather(1, source)
        complemented = self.complement_ids[reversed_inputs.long()]
        complemented = torch.where(valid, complemented, inputs.long())
        return complemented.to(inputs.dtype), valid

    def _classify(
        self,
        inputs: torch.Tensor,
        attention_mask: torch.Tensor,
        padding_ratio_hint: float | None = None,
    ) -> torch.Tensor:
        return self.head(
            self.encoder(
                inputs,
                attention_mask,
                padding_ratio_hint=padding_ratio_hint,
            )
        )

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        padding_ratio_hint: float | None = None,
    ) -> torch.Tensor:
        valid = inputs.ne(self.encoder.pad_token_id) if attention_mask is None else attention_mask.bool()
        if self.training:
            inputs = self.mutate(inputs, valid)
        reverse_inputs, reverse_mask = self.reverse_complement(inputs, valid)
        if self.training and self.reverse_complement_probability > 0.0:
            selected = torch.rand(inputs.shape[0], device=inputs.device) < self.reverse_complement_probability
            inputs = torch.where(selected.unsqueeze(1), reverse_inputs, inputs)
            return self._classify(inputs, valid, padding_ratio_hint)
        logits = self._classify(inputs, valid, padding_ratio_hint)
        if not self.training and self.reverse_complement_eval:
            reverse_logits = self._classify(
                reverse_inputs, reverse_mask, padding_ratio_hint
            )
            logits = 0.5 * (logits + reverse_logits)
        return logits


class SequencePairClassifier(nn.Module):
    """Symmetric shared-encoder head used by LRA document matching."""

    def __init__(
        self, encoder: SequenceMixerEncoder, num_classes: int = 2,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        dim = encoder.projection.out_features
        hidden_dim = hidden_dim or dim
        self.head = nn.Sequential(
            nn.Linear(4 * dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        first: torch.Tensor,
        first_mask: torch.Tensor,
        second: torch.Tensor,
        second_mask: torch.Tensor,
        *,
        first_padding_ratio_hint: float | None = None,
        second_padding_ratio_hint: float | None = None,
    ) -> torch.Tensor:
        first_embedding = self.encoder(
            first,
            first_mask,
            padding_ratio_hint=first_padding_ratio_hint,
        )
        second_embedding = self.encoder(
            second,
            second_mask,
            padding_ratio_hint=second_padding_ratio_hint,
        )
        features = torch.cat(
            [
                first_embedding,
                second_embedding,
                first_embedding - second_embedding,
                first_embedding * second_embedding,
            ],
            dim=-1,
        )
        return self.head(features)
