# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Shared VLM prefix-stream contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch
from torch import Tensor

VlmInputBatch = dict[str, Tensor]


@dataclass(frozen=True)
class VlmPrefixState:
    """Mutable-by-replacement state for one manually advanced VLM stream."""

    hidden: Tensor
    mask: Tensor
    attention_mask: Tensor | dict[str, Tensor] | None
    position_ids: Tensor
    cache_position: Tensor
    position_embeddings: tuple[Tensor, Tensor]
    suffix_position_offsets: Tensor | None = None
    extra: dict[str, Any] | None = None


@dataclass(frozen=True)
class LayerKV:
    """One layer of projected VLM key/value context."""

    key: Tensor
    value: Tensor

    def detach(self) -> LayerKV:
        return LayerKV(key=self.key.detach(), value=self.value.detach())


@dataclass(frozen=True)
class VlmAttentionGeometry:
    """Attention dimensions exposed by a concrete VLM backbone."""

    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def attention_dim(self) -> int:
        return self.num_attention_heads * self.head_dim


class VlmPrefixAdapter(Protocol):
    """Minimal VLM prefix stream used by sparse CE."""

    @property
    def num_layers(self) -> int: ...

    def prepare_prefix(
        self,
        vlm_inputs: VlmInputBatch,
        *,
        layer_attention_mask: Tensor | None = None,
    ) -> VlmPrefixState: ...

    def forward_prefix_layer(
        self,
        state: VlmPrefixState,
        *,
        layer_idx: int,
    ) -> VlmPrefixState: ...


class VlmPrefixLogitsAdapter(VlmPrefixAdapter, Protocol):
    """VLM prefix stream that can project final hidden states to LM logits."""

    @property
    def hidden_size(self) -> int: ...

    def final_hidden_to_logits(
        self,
        hidden: Tensor,
        *,
        candidate_token_ids: Tensor | None = None,
    ) -> Tensor: ...


def compact_position_ids_from_mask(
    attention_mask: Tensor,
    *,
    first_position: int = 0,
) -> Tensor:
    """Return compact token positions for valid prefix tokens."""
    mask = attention_mask.to(dtype=torch.bool)
    if mask.ndim != 2:
        raise ValueError(
            f"attention_mask must have shape (B, S), got {tuple(mask.shape)}."
        )
    positions = torch.cumsum(mask.to(dtype=torch.long), dim=1) - 1 + int(first_position)
    return positions.clamp_min(int(first_position))


def with_compact_position_ids(
    vlm_inputs: VlmInputBatch,
    *,
    first_position: int = 0,
) -> VlmInputBatch:
    """Return VLM inputs with compact position ids derived from attention mask."""
    attention_mask = cast(Tensor | None, vlm_inputs.get("attention_mask"))
    if attention_mask is None:
        input_ids = cast(Tensor | None, vlm_inputs.get("input_ids"))
        if input_ids is None:
            raise ValueError("VLM inputs must include attention_mask or input_ids.")
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    out = dict(vlm_inputs)
    out["position_ids"] = compact_position_ids_from_mask(
        attention_mask,
        first_position=first_position,
    )
    return out


def compact_suffix_position_offsets(
    attention_mask: Tensor,
    *,
    first_position: int = 0,
) -> Tensor:
    """Return the first suffix RoPE position after compact valid prefix tokens."""
    mask = attention_mask.to(dtype=torch.bool)
    if mask.ndim != 2:
        raise ValueError(
            f"attention_mask must have shape (B, S), got {tuple(mask.shape)}."
        )
    return mask.to(dtype=torch.long).sum(dim=1) + int(first_position)


def suffix_position_offsets_from_vlm_inputs(vlm_inputs: VlmInputBatch) -> Tensor:
    """Infer suffix RoPE offsets from VLM prefill inputs."""
    attention_mask = cast(Tensor | None, vlm_inputs.get("attention_mask"))
    input_ids = cast(Tensor | None, vlm_inputs.get("input_ids"))
    if attention_mask is None:
        if input_ids is None:
            raise ValueError("VLM inputs must include attention_mask or input_ids.")
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    mask = attention_mask.to(dtype=torch.bool)
    position_ids = cast(Tensor | None, vlm_inputs.get("position_ids"))
    if position_ids is None:
        return compact_suffix_position_offsets(mask)
    positions = position_ids
    if positions.ndim == 3:
        if positions.shape[1:] != mask.shape:
            raise ValueError(
                "rank-3 position_ids must have shape (N, B, S) matching "
                f"attention_mask (B, S), got {tuple(positions.shape)} and {tuple(mask.shape)}."
            )
        positions = positions.to(device=mask.device, dtype=torch.long)
        valid_positions = positions.masked_fill(
            ~mask[None, :, :],
            torch.iinfo(torch.long).min,
        )
        if not bool(mask.any(dim=1).all()):
            raise ValueError(
                "Every sample must expose at least one valid VLM prefix token."
            )
        return valid_positions.amax(dim=(0, 2)) + 1
    if positions.ndim != 2:
        raise ValueError(
            f"position_ids must have shape (B, S) or (N, B, S), got {tuple(positions.shape)}."
        )
    if positions.shape != mask.shape:
        raise ValueError(
            "rank-2 position_ids must have shape (B, S) matching attention_mask, "
            f"got {tuple(positions.shape)} and {tuple(mask.shape)}."
        )
    positions = positions.to(device=mask.device, dtype=torch.long)
    valid_positions = positions.masked_fill(~mask, torch.iinfo(torch.long).min)
    if not bool(mask.any(dim=1).all()):
        raise ValueError(
            "Every sample must expose at least one valid VLM prefix token."
        )
    return valid_positions.max(dim=1).values + 1


def attention_geometry_from_config(config: Any) -> VlmAttentionGeometry:
    """Read attention geometry from a concrete backbone text config."""
    num_heads = getattr(config, "num_attention_heads", None)
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = getattr(config, "head_dim", None)
    hidden_size = getattr(config, "hidden_size", None)
    if num_heads is None:
        raise RuntimeError("Text config does not define num_attention_heads.")
    if num_kv_heads is None:
        raise RuntimeError("Text config does not define num_key_value_heads.")
    if head_dim is None:
        if hidden_size is None:
            raise RuntimeError("Text config does not define head_dim or hidden_size.")
        head_dim = int(hidden_size) // int(num_heads)
    if int(head_dim) % 2 != 0:
        raise RuntimeError(
            f"Text config head_dim must be even for RoPE, got {head_dim}."
        )
    return VlmAttentionGeometry(
        num_attention_heads=int(num_heads),
        num_key_value_heads=int(num_kv_heads),
        head_dim=int(head_dim),
    )
