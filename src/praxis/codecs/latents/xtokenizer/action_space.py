# SPDX-FileCopyrightText: 2026 Xiaoshen Han
#
# SPDX-License-Identifier: Apache-2.0

"""Observation-relative action geometry for X-Tokenizer."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def to_observation_relative_actions(
    actions: Tensor,
    observation: Tensor,
    *,
    position_slice: slice | None = None,
    rotation_slice: slice | None = None,
    position_slices: tuple[slice, ...] | None = None,
    rotation_slices: tuple[slice, ...] | None = None,
) -> Tensor:
    """Express configuration channels relative to the pre-chunk observation."""
    _validate_inputs(actions, observation)
    relative = actions.clone()
    for segment in _resolve_slices(position_slice, position_slices):
        relative[..., segment] = actions[..., segment] - observation[..., segment]
    for segment in _resolve_slices(rotation_slice, rotation_slices):
        action_rotation = rotation_6d_to_matrix(actions[..., segment])
        observation_rotation = rotation_6d_to_matrix(observation[..., segment])
        relative[..., segment] = matrix_to_rotation_6d(
            action_rotation @ observation_rotation.transpose(-1, -2)
        )
    return relative


def from_observation_relative_actions(
    relative_actions: Tensor,
    observation: Tensor,
    *,
    position_slice: slice | None = None,
    rotation_slice: slice | None = None,
    position_slices: tuple[slice, ...] | None = None,
    rotation_slices: tuple[slice, ...] | None = None,
) -> Tensor:
    """Recover absolute configuration channels from observation-relative actions."""
    _validate_inputs(relative_actions, observation)
    absolute = relative_actions.clone()
    for segment in _resolve_slices(position_slice, position_slices):
        absolute[..., segment] = (
            relative_actions[..., segment] + observation[..., segment]
        )
    for segment in _resolve_slices(rotation_slice, rotation_slices):
        relative_rotation = rotation_6d_to_matrix(relative_actions[..., segment])
        observation_rotation = rotation_6d_to_matrix(observation[..., segment])
        absolute[..., segment] = matrix_to_rotation_6d(
            relative_rotation @ observation_rotation
        )
    return absolute


def rotation_6d_to_matrix(rotation: Tensor) -> Tensor:
    """Convert the repository's row-major 6D rotation representation to SO(3)."""
    if rotation.shape[-1] != 6:
        raise ValueError(
            f"rotation must end in six channels, got {tuple(rotation.shape)}."
        )
    first = rotation[..., :3]
    second = rotation[..., 3:]
    basis_1 = F.normalize(first, dim=-1, eps=1e-6)
    projection = (basis_1 * second).sum(dim=-1, keepdim=True)
    basis_2 = F.normalize(second - projection * basis_1, dim=-1, eps=1e-6)
    basis_3 = basis_1.cross(basis_2, dim=-1)
    return torch.stack((basis_1, basis_2, basis_3), dim=-2)


def matrix_to_rotation_6d(rotation: Tensor) -> Tensor:
    """Convert SO(3) matrices to the repository's row-major 6D representation."""
    if rotation.shape[-2:] != (3, 3):
        raise ValueError(
            f"rotation must end in a 3x3 matrix, got {tuple(rotation.shape)}."
        )
    return rotation[..., :2, :].reshape(*rotation.shape[:-2], 6)


def _validate_inputs(actions: Tensor, observation: Tensor) -> None:
    if actions.ndim != 3:
        raise ValueError(
            f"actions must have shape [B, T, D], got {tuple(actions.shape)}."
        )
    if observation.shape != (actions.shape[0], 1, actions.shape[2]):
        raise ValueError(
            "observation must have shape [B, 1, D] matching actions, got "
            f"{tuple(observation.shape)} and {tuple(actions.shape)}."
        )
    if not actions.is_floating_point() or not observation.is_floating_point():
        raise ValueError("actions and observation must have floating-point dtypes.")
    if actions.device != observation.device:
        raise ValueError("actions and observation must be on the same device.")


def _resolve_slices(
    singular: slice | None,
    plural: tuple[slice, ...] | None,
) -> tuple[slice, ...]:
    if plural is not None:
        if singular is not None:
            raise ValueError("Pass either a singular slice or plural slices, not both.")
        return tuple(plural)
    return () if singular is None else (singular,)


__all__ = [
    "from_observation_relative_actions",
    "matrix_to_rotation_6d",
    "rotation_6d_to_matrix",
    "to_observation_relative_actions",
]
