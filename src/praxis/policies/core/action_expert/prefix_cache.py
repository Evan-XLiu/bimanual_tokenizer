"""Prefix K/V condition wrappers for flow-matching action experts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from praxis.policies.core.vlm.prefix import LayerKV


@dataclass(frozen=True)
class PrefixCache:
    """VLM K/V cache plus a prefix-only validity mask."""

    layers: tuple[LayerKV, ...]
    mask: Tensor
    suffix_position_offsets: Tensor | None = None

    def detach(self) -> PrefixCache:
        return PrefixCache(
            layers=tuple(layer.detach() for layer in self.layers),
            mask=self.mask.detach(),
            suffix_position_offsets=None
            if self.suffix_position_offsets is None
            else self.suffix_position_offsets.detach(),
        )


def build_prefix_cache(
    *,
    past_key_values: Any,
    prefix_mask: Tensor,
    detach: bool,
    suffix_position_offsets: Tensor | None = None,
) -> PrefixCache:
    """Normalize HF cache outputs into prefix-only cache tensors.

    The cache keeps the original padded sequence shape so batched expert
    attention can stay dense, but all non-prefix positions are zeroed before
    they reach the expert. The mask is still carried as the authoritative
    attention constraint. ``detach`` is explicit because KI and FM differ
    exactly at this gradient seam.
    """
    mask = prefix_mask.bool()
    if detach:
        mask = mask.detach()
    layers = tuple(
        _mask_layer_to_prefix(layer, mask, detach=detach)
        for layer in _iter_layer_kv(past_key_values)
    )
    if not layers:
        raise RuntimeError("VLM forward did not return any past_key_values layers.")
    if suffix_position_offsets is not None:
        suffix_position_offsets = suffix_position_offsets.to(
            device=mask.device,
            dtype=torch.long,
        )
        if detach:
            suffix_position_offsets = suffix_position_offsets.detach()
    return PrefixCache(
        layers=layers,
        mask=mask,
        suffix_position_offsets=suffix_position_offsets,
    )


def _iter_layer_kv(past_key_values: Any) -> list[LayerKV]:
    if past_key_values is None:
        return []
    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()
    if not isinstance(past_key_values, (list, tuple)):
        raise TypeError(
            "Unsupported VLM cache type. Expected a sequence of layer K/V pairs "
            f"or a cache with to_legacy_cache(), got {type(past_key_values)!r}."
        )

    layers: list[LayerKV] = []
    for item in past_key_values:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            raise TypeError(
                "Each VLM cache layer must be a tuple/list containing key and value tensors."
            )
        key, value = item[0], item[1]
        if not isinstance(key, Tensor) or not isinstance(value, Tensor):
            raise TypeError("VLM cache keys and values must be tensors.")
        layers.append(LayerKV(key=key, value=value))
    return layers


def _mask_layer_to_prefix(
    layer: LayerKV, prefix_mask: Tensor, *, detach: bool
) -> LayerKV:
    return LayerKV(
        key=_zero_non_prefix_positions(layer.key, prefix_mask, detach=detach),
        value=_zero_non_prefix_positions(layer.value, prefix_mask, detach=detach),
    )


def _zero_non_prefix_positions(
    tensor: Tensor, prefix_mask: Tensor, *, detach: bool
) -> Tensor:
    """Zero positions outside ``prefix_mask`` while preserving requested gradients."""
    if detach:
        tensor = tensor.detach()
    mask = prefix_mask.to(device=tensor.device, dtype=torch.bool)
    if tensor.ndim == 4:
        if tensor.shape[0] != mask.shape[0] or tensor.shape[2] != mask.shape[1]:
            raise ValueError(
                "Rank-4 cache tensor must have shape (B, H, S, D) matching "
                f"prefix mask (B, S); got tensor={tuple(tensor.shape)} "
                f"mask={tuple(mask.shape)}."
            )
        return tensor.masked_fill(~mask[:, None, :, None], 0)
    if tensor.ndim == 3:
        if tensor.shape[0] != mask.shape[0] or tensor.shape[1] != mask.shape[1]:
            raise ValueError(
                "Rank-3 cache tensor must have shape (B, S, C) matching "
                f"prefix mask (B, S); got tensor={tuple(tensor.shape)} "
                f"mask={tuple(mask.shape)}."
            )
        return tensor.masked_fill(~mask[:, :, None], 0)
    raise ValueError(
        f"Expected VLM cache tensor with rank 3 or 4, got shape={tuple(tensor.shape)}."
    )


def flatten_cache_tensor(tensor: Tensor) -> Tensor:
    """Convert cache tensor to ``(B, S, C)`` for adapter projections."""
    if tensor.ndim == 4:
        # Common HF shape: (B, H_kv, S, D).
        return tensor.transpose(1, 2).reshape(tensor.shape[0], tensor.shape[2], -1)
    if tensor.ndim == 3:
        return tensor
    raise ValueError(
        f"Expected VLM cache tensor with rank 3 or 4, got shape={tuple(tensor.shape)}."
    )


def validate_cache_matches_mask(cache: PrefixCache) -> None:
    """Fail fast if cache sequence length and mask length diverge."""
    if not cache.layers:
        raise RuntimeError("Prefix cache has no layers.")
    seq_len = int(flatten_cache_tensor(cache.layers[0].key).shape[1])
    if cache.mask.ndim != 2 or int(cache.mask.shape[1]) != seq_len:
        raise ValueError(
            "Prefix cache mask must have shape (B, S) matching cache sequence "
            f"length {seq_len}, got {tuple(cache.mask.shape)}."
        )
    if not bool(cache.mask.any(dim=1).all()):
        raise ValueError(
            "Every sample must expose at least one valid VLM prefix token."
        )


def cache_to_device(cache: PrefixCache, *, device: torch.device) -> PrefixCache:
    """Move cache tensors to the expert device if needed."""
    return PrefixCache(
        layers=tuple(
            LayerKV(key=layer.key.to(device), value=layer.value.to(device))
            for layer in cache.layers
        ),
        mask=cache.mask.to(device),
        suffix_position_offsets=None
        if cache.suffix_position_offsets is None
        else cache.suffix_position_offsets.to(device),
    )
