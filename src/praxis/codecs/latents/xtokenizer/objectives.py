# SPDX-FileCopyrightText: 2026 Xiaoshen Han
#
# SPDX-License-Identifier: Apache-2.0

"""Masked reconstruction and semantic objectives for X-Tokenizer."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from praxis.codecs.latents.xtokenizer.action_space import (
    rotation_6d_to_matrix as _rotation_6d_to_matrix,
)
from praxis.codecs.latents.xtokenizer.config import _resolve_action_layout
from praxis.core.types import NormMode

__all__ = [
    "masked_action_l1",
    "rotation_geodesic_loss",
    "symmetric_global_local_infonce",
]


def masked_action_l1(
    reconstruction: Tensor,
    target: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    """Return elementwise L1 over valid rows or individual action DoFs."""
    _validate_feature_pair(reconstruction, target)
    feature_mask = _resolve_feature_mask(valid_mask, reconstruction)
    valid_reconstruction = reconstruction[feature_mask]
    valid_target = target[feature_mask]
    if valid_reconstruction.numel() == 0:
        return reconstruction.masked_fill(~feature_mask, 0).sum() * 0.0
    return F.l1_loss(valid_reconstruction, valid_target)


def rotation_geodesic_loss(
    reconstruction: Tensor,
    target: Tensor,
    stats: Mapping[str, object] | None,
    valid_mask: Tensor,
    *,
    normalization_mode: NormMode | None = None,
    rotation_slice: slice | None = None,
) -> Tensor:
    """Return masked, pi-normalized SO(3) distance for a named 6D rotation."""
    _validate_feature_pair(reconstruction, target)
    resolved_rotation_slice = _resolve_rotation_slice(
        int(reconstruction.shape[-1]),
        rotation_slice,
    )
    if valid_mask.shape == reconstruction.shape:
        _validate_boolean_mask(valid_mask, reconstruction.device)
        rotation_valid = valid_mask[..., resolved_rotation_slice].all(dim=-1)
    else:
        _validate_valid_mask(
            valid_mask,
            reconstruction.shape[:-1],
            reconstruction.device,
        )
        rotation_valid = valid_mask
    reconstruction_rotation = _rotation_channels(
        reconstruction,
        resolved_rotation_slice,
    )
    target_rotation = _rotation_channels(target, resolved_rotation_slice)
    valid_reconstruction = reconstruction_rotation[rotation_valid]
    valid_target = target_rotation[rotation_valid]
    if valid_reconstruction.numel() == 0:
        return (
            reconstruction_rotation.masked_fill(
                ~rotation_valid.unsqueeze(-1),
                0,
            ).sum()
            * 0.0
        )

    resolved_stats = _resolve_stats(stats)
    with torch.autocast(device_type=reconstruction.device.type, enabled=False):
        valid_reconstruction = _denormalize_rotation(
            valid_reconstruction.float(),
            resolved_stats,
            action_dim=int(reconstruction.shape[-1]),
            normalization_mode=normalization_mode,
            rotation_slice=resolved_rotation_slice,
        )
        valid_target = _denormalize_rotation(
            valid_target.float(),
            resolved_stats,
            action_dim=int(target.shape[-1]),
            normalization_mode=normalization_mode,
            rotation_slice=resolved_rotation_slice,
        )
        reconstruction_matrix = _rotation_6d_to_matrix(valid_reconstruction)
        target_matrix = _rotation_6d_to_matrix(valid_target)
        relative = reconstruction_matrix @ target_matrix.transpose(-1, -2)
        cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(
            -1.0,
            1.0,
        )

        epsilon = torch.finfo(cosine.dtype).eps * 4
        interior = cosine.clamp(-1.0 + epsilon, 1.0 - epsilon)
        angle = torch.acos(interior)
        angle = torch.where(cosine >= 1.0 - epsilon, torch.zeros_like(angle), angle)
        angle = torch.where(
            cosine <= -1.0 + epsilon,
            torch.full_like(angle, torch.pi),
            angle,
        )
    return angle.mean() / torch.pi


def symmetric_global_local_infonce(
    action_features: Tensor,
    vl_features: Tensor,
    valid_mask: Tensor,
    logit_scale: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute symmetric trajectory-level and slot-level InfoNCE losses."""
    _validate_feature_pair(action_features, vl_features)
    if action_features.ndim != 3:
        raise ValueError(
            "action_features and vl_features must have shape [B, M, H], "
            f"got {tuple(action_features.shape)}."
        )
    _validate_valid_mask(
        valid_mask,
        action_features.shape[:2],
        action_features.device,
    )
    if logit_scale.numel() != 1 or not logit_scale.is_floating_point():
        raise ValueError("logit_scale must be a floating-point scalar tensor.")
    if logit_scale.device != action_features.device:
        raise ValueError("logit_scale and feature tensors must be on the same device.")

    clean_action = action_features.masked_fill(~valid_mask.unsqueeze(-1), 0)
    clean_vl = vl_features.masked_fill(~valid_mask.unsqueeze(-1), 0)
    counts = valid_mask.sum(dim=1, keepdim=True)
    valid_trajectories = counts.squeeze(-1) > 0
    global_action = clean_action.sum(dim=1) / counts.clamp_min(1)
    global_vl = clean_vl.sum(dim=1) / counts.clamp_min(1)
    global_loss = _symmetric_infonce(
        global_action[valid_trajectories],
        global_vl[valid_trajectories],
        logit_scale,
        zero_source=clean_action,
    )
    local_loss = _symmetric_infonce(
        action_features[valid_mask],
        vl_features[valid_mask],
        logit_scale,
        zero_source=clean_action,
    )
    return global_loss, local_loss


def _symmetric_infonce(
    first: Tensor,
    second: Tensor,
    logit_scale: Tensor,
    *,
    zero_source: Tensor,
) -> Tensor:
    first, second = _gather_distributed_pairs(first, second)
    if first.shape[0] == 0:
        return zero_source.sum() * 0.0 + logit_scale.sum() * 0.0
    first = F.normalize(first.float(), dim=-1, eps=1e-6)
    second = F.normalize(second.float(), dim=-1, eps=1e-6)
    scale = logit_scale.float().clamp(max=math.log(100.0)).exp()
    logits = first @ second.transpose(0, 1) * scale
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.transpose(0, 1), labels)
    )


def _gather_distributed_pairs(first: Tensor, second: Tensor) -> tuple[Tensor, Tensor]:
    if not dist.is_available() or not dist.is_initialized():
        return first, second
    world_size = dist.get_world_size()
    if world_size == 1:
        return first, second

    local_size = torch.tensor([first.shape[0]], dtype=torch.int64, device=first.device)
    gathered_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(gathered_sizes, local_size)
    sizes = [int(size.item()) for size in gathered_sizes]
    max_size = max(sizes)
    if max_size == 0:
        return first, second

    def _pad(value: Tensor) -> Tensor:
        if value.shape[0] == max_size:
            return value
        padding = value.new_zeros((max_size - value.shape[0], value.shape[1]))
        return torch.cat((value, padding), dim=0)

    from torch.distributed.nn.functional import all_gather

    gathered_first = cast(tuple[Tensor, ...], all_gather(_pad(first)))
    gathered_second = cast(tuple[Tensor, ...], all_gather(_pad(second)))
    return (
        torch.cat(
            [value[:size] for value, size in zip(gathered_first, sizes, strict=True)]
        ),
        torch.cat(
            [value[:size] for value, size in zip(gathered_second, sizes, strict=True)]
        ),
    )


def _resolve_rotation_slice(
    action_dim: int,
    rotation_slice: slice | None,
) -> slice:
    if action_dim == 6:
        resolved = slice(0, 6) if rotation_slice is None else rotation_slice
    elif rotation_slice is None:
        _, _, resolved_slices, _ = _resolve_action_layout(action_dim, None)
        if not resolved_slices:
            raise ValueError(
                "rotation inputs require a named six-channel rotation DoF."
            )
        if len(resolved_slices) != 1:
            raise ValueError(
                "rotation_slice is required when the action layout has multiple "
                "rotation groups."
            )
        resolved = resolved_slices[0]
    else:
        resolved = rotation_slice
    start, stop, step = resolved.indices(action_dim)
    if step != 1 or stop - start != 6:
        raise ValueError(
            "rotation_slice must select exactly six contiguous action channels."
        )
    return slice(start, stop)


def _rotation_channels(actions: Tensor, rotation_slice: slice) -> Tensor:
    return actions[..., rotation_slice]


def _resolve_stats(
    stats: Mapping[str, object] | None,
) -> Mapping[str, object]:
    if stats is None:
        return {}
    statistic_keys = {"mean", "std", "min", "max", "q01", "q99"}
    if statistic_keys.intersection(stats):
        return stats
    nested = [value for value in stats.values() if isinstance(value, Mapping)]
    if len(nested) == 1:
        return nested[0]
    return stats


def _denormalize_rotation(
    rotation: Tensor,
    stats: Mapping[str, object],
    *,
    action_dim: int,
    normalization_mode: NormMode | None,
    rotation_slice: slice,
) -> Tensor:
    if normalization_mode == NormMode.NONE:
        return rotation

    mean = _rotation_stat(
        stats.get("mean"),
        rotation,
        action_dim=action_dim,
        rotation_slice=rotation_slice,
    )
    std = _rotation_stat(
        stats.get("std"),
        rotation,
        action_dim=action_dim,
        rotation_slice=rotation_slice,
    )
    if (
        normalization_mode in (None, NormMode.MEAN_STD)
        and mean is not None
        and std is not None
    ):
        return rotation * std + mean

    minimum = _rotation_stat(
        stats.get("min", stats.get("q01")),
        rotation,
        action_dim=action_dim,
        rotation_slice=rotation_slice,
    )
    maximum = _rotation_stat(
        stats.get("max", stats.get("q99")),
        rotation,
        action_dim=action_dim,
        rotation_slice=rotation_slice,
    )
    if (
        normalization_mode in (None, NormMode.MIN_MAX)
        and minimum is not None
        and maximum is not None
    ):
        return (rotation + 1.0) * 0.5 * (maximum - minimum) + minimum
    return rotation


def _rotation_stat(
    value: object | None,
    reference: Tensor,
    *,
    action_dim: int,
    rotation_slice: slice,
) -> Tensor | None:
    if value is None:
        return None
    tensor = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    tensor = tensor.squeeze()
    if tensor.ndim != 1:
        raise ValueError(
            "action normalization statistics must resolve to one-dimensional "
            f"channel vectors, got {tuple(tensor.shape)}."
        )
    if tensor.numel() == 6:
        return tensor
    if tensor.numel() == action_dim:
        return tensor[rotation_slice]
    raise ValueError(
        "action normalization statistics must contain either six rotation "
        f"channels or all {action_dim} action channels, got {tensor.numel()}."
    )


def _validate_feature_pair(first: Tensor, second: Tensor) -> None:
    if first.shape != second.shape:
        raise ValueError(
            f"feature tensors must have matching shapes, got {tuple(first.shape)} "
            f"and {tuple(second.shape)}."
        )
    if first.ndim < 2:
        raise ValueError(
            f"feature tensors must include sample and feature dimensions, got {first.ndim}."
        )
    if not first.is_floating_point() or not second.is_floating_point():
        raise ValueError("feature tensors must have floating-point dtypes.")
    if first.device != second.device:
        raise ValueError("feature tensors must be on the same device.")


def _validate_valid_mask(
    valid_mask: Tensor,
    expected_shape: torch.Size,
    expected_device: torch.device,
) -> None:
    if valid_mask.shape != expected_shape:
        raise ValueError(
            f"valid_mask must have shape {tuple(expected_shape)}, "
            f"got {tuple(valid_mask.shape)}."
        )
    if valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must have boolean dtype.")
    if valid_mask.device != expected_device:
        raise ValueError("valid_mask and feature tensors must be on the same device.")


def _resolve_feature_mask(valid_mask: Tensor, features: Tensor) -> Tensor:
    if valid_mask.shape == features.shape:
        _validate_boolean_mask(valid_mask, features.device)
        return valid_mask
    _validate_valid_mask(valid_mask, features.shape[:-1], features.device)
    return valid_mask.unsqueeze(-1).expand_as(features)


def _validate_boolean_mask(valid_mask: Tensor, expected_device: torch.device) -> None:
    if valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must have boolean dtype.")
    if valid_mask.device != expected_device:
        raise ValueError("valid_mask and feature tensors must be on the same device.")
