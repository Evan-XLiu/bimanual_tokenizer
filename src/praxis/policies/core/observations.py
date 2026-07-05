# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Observation helpers shared by policy families."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor


def extract_instructions(
    batch: dict[str, Any], *, language_key: str | None
) -> list[str] | None:
    """Extract per-sample language instructions from a batch."""
    if language_key is None:
        return None
    raw = batch.get(language_key)
    if raw is None or isinstance(raw, Tensor):
        return None
    if isinstance(raw, str):
        return [raw]
    return [str(item) for item in raw]


def stack_observation_images(
    obs: dict[str, Tensor],
    *,
    image_features: Sequence[Any],
    as_uint8: bool,
) -> Tensor:
    """Convert observation image tensors to a ``(B, N, C, H, W)`` batch."""
    camera_keys = [feat.key for feat in image_features]
    if not camera_keys:
        first_tensor = next(
            (value for value in obs.values() if isinstance(value, Tensor)), None
        )
        if first_tensor is None:
            raise RuntimeError(
                "Cannot infer batch size for observations without tensor values."
            )
        dtype = torch.uint8 if as_uint8 else torch.float32
        return torch.empty(
            (int(first_tensor.shape[0]), 0, 3, 1, 1),
            dtype=dtype,
            device=first_tensor.device,
        )

    camera_batches: list[Tensor] = []
    for key in camera_keys:
        imgs = obs[key]
        if imgs.dim() == 5:
            imgs = imgs[:, -1]
        if as_uint8:
            imgs = (
                (imgs * 255.0).round().clamp(0, 255)
                if imgs.is_floating_point()
                else imgs.clamp(0, 255)
            )
            imgs = imgs.to(torch.uint8)
        camera_batches.append(imgs)
    return torch.stack(camera_batches, dim=1)


def concatenate_state_features(
    obs: dict[str, Tensor],
    *,
    state_features: Sequence[Any],
) -> Tensor | None:
    """Return last-step continuous state features as one vector."""
    parts: list[Tensor] = []
    for feat in state_features:
        value = obs.get(feat.key)
        if value is None:
            continue
        feature_shape = tuple(int(dim) for dim in getattr(feat, "shape", ()))
        rank_without_time = 1 + len(feature_shape)
        rank_with_time = rank_without_time + 1
        if value.dim() == rank_with_time:
            value = value[:, -1]
        elif value.dim() != rank_without_time:
            raise ValueError(
                f"State feature {feat.key!r} expected rank {rank_without_time} "
                f"or {rank_with_time}, got shape {tuple(value.shape)}."
            )
        if tuple(int(dim) for dim in value.shape[1:]) != feature_shape:
            raise ValueError(
                f"State feature {feat.key!r} expected trailing shape "
                f"{feature_shape}, got {tuple(value.shape[1:])}."
            )
        parts.append(value.reshape(int(value.shape[0]), -1))
    if not parts:
        return None
    return torch.cat(parts, dim=-1)
