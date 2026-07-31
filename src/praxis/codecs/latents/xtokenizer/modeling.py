# SPDX-FileCopyrightText: 2026 Xiaoshen Han
#
# SPDX-License-Identifier: Apache-2.0

"""Perceiver-style observation-relative action modeling blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor

from praxis.codecs.latents.xtokenizer.config import XTokenizerModelConfig
from praxis.nn.embeddings.positional import RotaryPosEmbed

__all__ = [
    "ActionDecoder",
    "DeltaActionEncoder",
    "EncoderOutput",
    "RelativeActionEncoder",
]


@dataclass(frozen=True)
class EncoderOutput:
    """Latent action slots and their validity mask."""

    latents: Tensor
    latent_mask: Tensor


class _ActionProjection(nn.Sequential):
    """Paper input stem with compatibility access to the input linear weight."""

    def __init__(self, action_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    @property
    def weight(self) -> Tensor:
        return cast(nn.Linear, self[0]).weight


class _RotaryAttention(nn.Module):
    """Multi-head attention with independently rotary-encoded queries and keys."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads, "
                f"got {hidden_dim} and {num_heads}."
            )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: Tensor,
        context: Tensor,
        context_padding_mask: Tensor,
        *,
        query_rope: RotaryPosEmbed | None,
        key_rope: RotaryPosEmbed | None,
    ) -> Tensor:
        batch_size, query_length, _ = queries.shape
        query = self._split_heads(self.query_projection(queries))
        key = self._split_heads(self.key_projection(context))
        value = self._split_heads(self.value_projection(context))
        if query_rope is not None:
            query = query_rope(query)
        if key_rope is not None:
            key = key_rope(key)

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(
            context_padding_mask[:, None, None, :],
            torch.finfo(scores.dtype).min,
        )
        weights = torch.softmax(scores.float(), dim=-1).to(dtype=scores.dtype)
        attended = torch.matmul(self.dropout(weights), value)
        attended = attended.transpose(1, 2).reshape(
            batch_size,
            query_length,
            self.hidden_dim,
        )
        return self.output_projection(attended)

    def _split_heads(self, tensor: Tensor) -> Tensor:
        return tensor.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)


class _FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        return self.layers(hidden)


class _SelfAttentionBlock(nn.Module):
    """Pre-normalized rotary self-attention and position-wise FFN."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = _RotaryAttention(hidden_dim, num_heads, dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = _FeedForward(hidden_dim, dropout)
        self.mlp_dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: Tensor,
        padding_mask: Tensor,
        rope: RotaryPosEmbed,
    ) -> Tensor:
        normalized = self.attention_norm(hidden)
        attended = self.attention(
            normalized,
            normalized,
            padding_mask,
            query_rope=rope,
            key_rope=rope,
        )
        hidden = hidden + self.attention_dropout(attended)
        return hidden + self.mlp_dropout(self.mlp(self.mlp_norm(hidden)))


class _CrossAttentionBlock(nn.Module):
    """Pre-normalized cross-attention with an optional position-wise FFN."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        *,
        with_mlp: bool,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.attention = _RotaryAttention(hidden_dim, num_heads, dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.mlp_norm = nn.LayerNorm(hidden_dim) if with_mlp else None
        self.mlp = _FeedForward(hidden_dim, dropout) if with_mlp else None
        self.mlp_dropout = nn.Dropout(dropout) if with_mlp else None

    def forward(
        self,
        queries: Tensor,
        context: Tensor,
        context_padding_mask: Tensor,
        *,
        query_rope: RotaryPosEmbed | None,
        key_rope: RotaryPosEmbed | None,
    ) -> Tensor:
        attended = self.attention(
            self.query_norm(queries),
            self.context_norm(context),
            context_padding_mask,
            query_rope=query_rope,
            key_rope=key_rope,
        )
        hidden = self.output_norm(queries + self.attention_dropout(attended))
        if self.mlp is None or self.mlp_norm is None or self.mlp_dropout is None:
            return hidden
        return hidden + self.mlp_dropout(self.mlp(self.mlp_norm(hidden)))


class RelativeActionEncoder(nn.Module):
    """Encode observation-relative actions into compressed latent slots."""

    def __init__(self, config: XTokenizerModelConfig) -> None:
        super().__init__()
        geometry = _resolve_geometry(config)
        self.action_dim = geometry.action_dim
        self.action_horizon = geometry.action_horizon
        self.compression_ratio = geometry.compression_ratio
        self.hidden_dim = geometry.hidden_dim
        self.latent_dim = geometry.latent_dim
        self.latent_horizon = geometry.latent_horizon
        self.embodiment_vocab_size = geometry.embodiment_vocab_size

        self.action_projection = _ActionProjection(
            self.action_dim,
            self.hidden_dim,
            geometry.dropout,
        )
        self.anchor_projection = nn.Linear(self.action_dim, self.hidden_dim)
        self.embodiment_embedding = nn.Embedding(
            self.embodiment_vocab_size,
            self.hidden_dim,
        )
        encoder_head_dim = self.hidden_dim // geometry.encoder_num_heads
        self.action_rope = RotaryPosEmbed(encoder_head_dim, base=10_000.0)
        self.latent_rope = RotaryPosEmbed(encoder_head_dim, base=10_000.0)
        self.temporal_encoder = nn.ModuleList(
            [
                _SelfAttentionBlock(
                    self.hidden_dim,
                    geometry.encoder_num_heads,
                    geometry.dropout,
                )
                for _ in range(geometry.encoder_depth)
            ]
        )
        self.state_cross_attention = _CrossAttentionBlock(
            self.hidden_dim,
            geometry.encoder_num_heads,
            geometry.dropout,
            with_mlp=False,
        )
        self.latent_queries = nn.Parameter(
            torch.empty(geometry.max_latent_horizon, self.hidden_dim)
        )
        self.latent_cross_attention = _CrossAttentionBlock(
            self.hidden_dim,
            geometry.encoder_num_heads,
            geometry.dropout,
            with_mlp=True,
        )
        self.latent_projection = nn.Linear(self.hidden_dim, self.latent_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        _initialize_transformer_weights(self)
        nn.init.trunc_normal_(self.latent_queries, std=0.02)

    def forward(
        self,
        actions: Tensor,
        anchor: Tensor,
        robot_type: Tensor,
        action_mask: Tensor | None,
        *,
        embodiment_valid: Tensor | None = None,
    ) -> EncoderOutput:
        batch_size = _validate_sequence(
            actions,
            name="actions",
            length=self.action_horizon,
            dimension=self.action_dim,
        )
        _validate_conditioning(
            anchor,
            robot_type,
            batch_size=batch_size,
            action_dim=self.action_dim,
            embodiment_vocab_size=self.embodiment_vocab_size,
            device=actions.device,
        )
        valid_mask = _resolve_action_mask(
            action_mask,
            batch_size=batch_size,
            action_horizon=self.action_horizon,
            device=actions.device,
        )
        latent_mask = _compress_mask(valid_mask, self.compression_ratio)
        action_padding_mask = _safe_padding_mask(~valid_mask)

        clean_actions = actions.masked_fill(~valid_mask.unsqueeze(-1), 0)
        embodiment = _embodiment_conditioning(
            self.embodiment_embedding,
            robot_type,
            embodiment_valid,
        )
        action_tokens = self.action_projection(clean_actions) + embodiment
        action_tokens = action_tokens.masked_fill(~valid_mask.unsqueeze(-1), 0)
        for block in self.temporal_encoder:
            action_tokens = block(
                action_tokens,
                action_padding_mask,
                self.action_rope,
            )
            action_tokens = action_tokens.masked_fill(~valid_mask.unsqueeze(-1), 0)

        state_padding_mask = torch.zeros(
            batch_size,
            1,
            dtype=torch.bool,
            device=actions.device,
        )
        action_tokens = self.state_cross_attention(
            action_tokens,
            self.anchor_projection(anchor),
            state_padding_mask,
            query_rope=None,
            key_rope=None,
        )
        action_tokens = action_tokens.masked_fill(~valid_mask.unsqueeze(-1), 0)

        latent_queries = self.latent_queries[: self.latent_horizon]
        latent_queries = latent_queries.unsqueeze(0).expand(batch_size, -1, -1)
        hidden = self.latent_cross_attention(
            latent_queries,
            action_tokens,
            action_padding_mask,
            query_rope=self.latent_rope,
            key_rope=self.action_rope,
        )
        latents = self.latent_projection(hidden)
        latents = latents.masked_fill(~latent_mask.unsqueeze(-1), 0)
        return EncoderOutput(latents=latents, latent_mask=latent_mask)


class ActionDecoder(nn.Module):
    """Decode latent slots into observation-relative action sequences."""

    def __init__(self, config: XTokenizerModelConfig) -> None:
        super().__init__()
        geometry = _resolve_geometry(config)
        self.action_dim = geometry.action_dim
        self.action_horizon = geometry.action_horizon
        self.compression_ratio = geometry.compression_ratio
        self.hidden_dim = geometry.hidden_dim
        self.latent_dim = geometry.latent_dim
        self.latent_horizon = geometry.latent_horizon
        self.embodiment_vocab_size = geometry.embodiment_vocab_size

        self.latent_projection = nn.Linear(self.latent_dim, self.hidden_dim)
        self.anchor_projection = nn.Linear(self.action_dim, self.hidden_dim)
        self.embodiment_embedding = nn.Embedding(
            self.embodiment_vocab_size,
            self.hidden_dim,
        )
        decoder_head_dim = self.hidden_dim // geometry.decoder_num_heads
        self.latent_rope = RotaryPosEmbed(decoder_head_dim, base=10_000.0)
        self.action_rope = RotaryPosEmbed(decoder_head_dim, base=10_000.0)
        self.timestep_queries = nn.Parameter(
            torch.empty(geometry.max_action_horizon, self.hidden_dim)
        )
        self.timestep_cross_attention = _CrossAttentionBlock(
            self.hidden_dim,
            geometry.decoder_num_heads,
            geometry.dropout,
            with_mlp=False,
        )
        self.state_cross_attention = _CrossAttentionBlock(
            self.hidden_dim,
            geometry.decoder_num_heads,
            geometry.dropout,
            with_mlp=False,
        )
        self.temporal_decoder = nn.ModuleList(
            [
                _SelfAttentionBlock(
                    self.hidden_dim,
                    geometry.decoder_num_heads,
                    geometry.dropout,
                )
                for _ in range(geometry.decoder_depth)
            ]
        )
        self.output_projection = nn.Linear(self.hidden_dim, self.action_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        _initialize_transformer_weights(self)
        nn.init.trunc_normal_(self.timestep_queries, std=0.02)

    def forward(
        self,
        latents: Tensor,
        anchor: Tensor,
        robot_type: Tensor,
        action_mask: Tensor | None,
        *,
        embodiment_valid: Tensor | None = None,
    ) -> Tensor:
        batch_size = _validate_sequence(
            latents,
            name="latents",
            length=self.latent_horizon,
            dimension=self.latent_dim,
        )
        _validate_conditioning(
            anchor,
            robot_type,
            batch_size=batch_size,
            action_dim=self.action_dim,
            embodiment_vocab_size=self.embodiment_vocab_size,
            device=latents.device,
        )
        valid_mask = _resolve_action_mask(
            action_mask,
            batch_size=batch_size,
            action_horizon=self.action_horizon,
            device=latents.device,
        )
        latent_mask = _compress_mask(valid_mask, self.compression_ratio)
        latent_padding_mask = _safe_padding_mask(~latent_mask)

        clean_latents = latents.masked_fill(~latent_mask.unsqueeze(-1), 0)
        latent_tokens = self.latent_projection(clean_latents)
        latent_tokens = latent_tokens.masked_fill(~latent_mask.unsqueeze(-1), 0)
        embodiment = _embodiment_conditioning(
            self.embodiment_embedding,
            robot_type,
            embodiment_valid,
        )
        timestep_queries = self.timestep_queries[: self.action_horizon]
        timestep_queries = timestep_queries.unsqueeze(0).expand(batch_size, -1, -1)
        timestep_queries = timestep_queries + embodiment
        hidden = self.timestep_cross_attention(
            timestep_queries,
            latent_tokens,
            latent_padding_mask,
            query_rope=self.action_rope,
            key_rope=self.latent_rope,
        )

        state_padding_mask = torch.zeros(
            batch_size,
            1,
            dtype=torch.bool,
            device=latents.device,
        )
        hidden = self.state_cross_attention(
            hidden,
            self.anchor_projection(anchor),
            state_padding_mask,
            query_rope=None,
            key_rope=None,
        )

        timestep_padding_mask = _safe_padding_mask(~valid_mask)
        hidden = hidden.masked_fill(~valid_mask.unsqueeze(-1), 0)
        for block in self.temporal_decoder:
            hidden = block(hidden, timestep_padding_mask, self.action_rope)
            hidden = hidden.masked_fill(~valid_mask.unsqueeze(-1), 0)
        reconstructed = self.output_projection(hidden)
        return reconstructed.masked_fill(~valid_mask.unsqueeze(-1), 0)


@dataclass(frozen=True)
class _ModelGeometry:
    action_dim: int
    action_horizon: int
    max_action_horizon: int
    compression_ratio: int
    hidden_dim: int
    latent_dim: int
    latent_horizon: int
    max_latent_horizon: int
    encoder_num_heads: int
    encoder_depth: int
    decoder_num_heads: int
    decoder_depth: int
    dropout: float
    embodiment_vocab_size: int


def _resolve_geometry(config: XTokenizerModelConfig) -> _ModelGeometry:
    if config.action_dim is None:
        raise ValueError("action_dim must be configured before constructing the model.")

    values = {
        "action_dim": int(config.action_dim),
        "action_horizon": int(config.action_horizon),
        "max_action_horizon": int(config.max_action_horizon),
        "compression_ratio": int(config.compression_ratio),
        "hidden_dim": int(config.hidden_dim),
        "latent_dim": int(config.latent_dim),
        "encoder_num_heads": int(config.encoder_num_heads),
        "encoder_depth": int(config.encoder_depth),
        "decoder_num_heads": int(config.decoder_num_heads),
        "decoder_depth": int(config.decoder_depth),
        "embodiment_vocab_size": int(config.embodiment_vocab_size),
    }
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}.")
    action_horizon = values["action_horizon"]
    max_action_horizon = values["max_action_horizon"]
    compression_ratio = values["compression_ratio"]
    if action_horizon % compression_ratio != 0:
        raise ValueError(
            "action_horizon must be divisible by compression_ratio, "
            f"got {action_horizon} and {compression_ratio}."
        )
    if max_action_horizon % compression_ratio != 0:
        raise ValueError(
            "max_action_horizon must be divisible by compression_ratio, "
            f"got {max_action_horizon} and {compression_ratio}."
        )
    if action_horizon > max_action_horizon:
        raise ValueError(
            "action_horizon must not exceed max_action_horizon, "
            f"got {action_horizon} and {max_action_horizon}."
        )
    hidden_dim = values["hidden_dim"]
    for name in ("encoder_num_heads", "decoder_num_heads"):
        num_heads = values[name]
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim must be divisible by {name}, "
                f"got {hidden_dim} and {num_heads}."
            )

    dropout = float(config.dropout)
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
    return _ModelGeometry(
        action_dim=values["action_dim"],
        action_horizon=action_horizon,
        max_action_horizon=max_action_horizon,
        compression_ratio=compression_ratio,
        hidden_dim=hidden_dim,
        latent_dim=values["latent_dim"],
        latent_horizon=action_horizon // compression_ratio,
        max_latent_horizon=max_action_horizon // compression_ratio,
        encoder_num_heads=values["encoder_num_heads"],
        encoder_depth=values["encoder_depth"],
        decoder_num_heads=values["decoder_num_heads"],
        decoder_depth=values["decoder_depth"],
        dropout=dropout,
        embodiment_vocab_size=values["embodiment_vocab_size"],
    )


def _validate_sequence(
    sequence: Tensor,
    *,
    name: str,
    length: int,
    dimension: int,
) -> int:
    if sequence.ndim != 3:
        raise ValueError(
            f"{name} must have shape [B, {length}, {dimension}], "
            f"got {tuple(sequence.shape)}."
        )
    if sequence.shape[1:] != (length, dimension):
        raise ValueError(
            f"{name} must have shape [B, {length}, {dimension}], "
            f"got {tuple(sequence.shape)}."
        )
    if not sequence.is_floating_point():
        raise ValueError(f"{name} must have a floating-point dtype.")
    return int(sequence.shape[0])


def _validate_conditioning(
    anchor: Tensor,
    robot_type: Tensor,
    *,
    batch_size: int,
    action_dim: int,
    embodiment_vocab_size: int,
    device: torch.device,
) -> None:
    expected_anchor_shape = (batch_size, 1, action_dim)
    if anchor.shape != expected_anchor_shape:
        raise ValueError(
            f"anchor must have shape {expected_anchor_shape}, "
            f"got {tuple(anchor.shape)}."
        )
    if not anchor.is_floating_point():
        raise ValueError("anchor must have a floating-point dtype.")
    if anchor.device != device:
        raise ValueError("anchor and sequence inputs must be on the same device.")

    if robot_type.shape != (batch_size,):
        raise ValueError(
            f"robot_type must have shape ({batch_size},), "
            f"got {tuple(robot_type.shape)}."
        )
    if (
        robot_type.is_floating_point()
        or robot_type.is_complex()
        or robot_type.dtype == torch.bool
    ):
        raise ValueError("robot_type must have an integer dtype.")
    if robot_type.device != device:
        raise ValueError("robot_type and sequence inputs must be on the same device.")
    if robot_type.numel() > 0 and (
        bool((robot_type < 0).any())
        or bool((robot_type >= embodiment_vocab_size).any())
    ):
        raise ValueError(f"robot_type ids must be in [0, {embodiment_vocab_size}).")


def _resolve_action_mask(
    action_mask: Tensor | None,
    *,
    batch_size: int,
    action_horizon: int,
    device: torch.device,
) -> Tensor:
    expected_shape = (batch_size, action_horizon)
    if action_mask is None:
        return torch.ones(expected_shape, dtype=torch.bool, device=device)
    if action_mask.shape != expected_shape:
        raise ValueError(
            f"action_mask must have shape {expected_shape}, "
            f"got {tuple(action_mask.shape)}."
        )
    if action_mask.dtype != torch.bool:
        raise ValueError("action_mask must have boolean dtype.")
    if action_mask.device != device:
        raise ValueError("action_mask and sequence inputs must be on the same device.")
    return action_mask


def _compress_mask(action_mask: Tensor, compression_ratio: int) -> Tensor:
    return action_mask.unflatten(1, (-1, compression_ratio)).any(dim=-1)


def _safe_padding_mask(padding_mask: Tensor) -> Tensor:
    all_padded = padding_mask.all(dim=1)
    if not bool(all_padded.any()):
        return padding_mask
    safe_mask = padding_mask.clone()
    safe_mask[all_padded, 0] = False
    return safe_mask


def _embodiment_conditioning(
    embedding: nn.Embedding,
    robot_type: Tensor,
    embodiment_valid: Tensor | None,
) -> Tensor:
    conditioning = embedding(robot_type.long()).unsqueeze(1)
    if embodiment_valid is None:
        return conditioning
    if embodiment_valid.shape != robot_type.shape:
        raise ValueError(
            "embodiment_valid must match robot_type shape, got "
            f"{tuple(embodiment_valid.shape)} and {tuple(robot_type.shape)}."
        )
    if embodiment_valid.dtype != torch.bool:
        raise ValueError("embodiment_valid must have boolean dtype.")
    if embodiment_valid.device != robot_type.device:
        raise ValueError("embodiment_valid and robot_type must share a device.")
    return conditioning.masked_fill(~embodiment_valid[:, None, None], 0)


def _initialize_transformer_weights(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.xavier_uniform_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
        elif isinstance(child, nn.Embedding):
            nn.init.xavier_uniform_(child.weight)
        elif isinstance(child, nn.LayerNorm):
            if child.weight is not None:
                nn.init.ones_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)


# Compatibility export for callers using the earlier, ambiguous name. New code
# should use RelativeActionEncoder; temporal deltas mean adjacent-frame differences.
DeltaActionEncoder = RelativeActionEncoder
