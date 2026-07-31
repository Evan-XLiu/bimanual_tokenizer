# SPDX-License-Identifier: Apache-2.0

"""Bridge action binding for X-Tokenizer's canonical absolute layout."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

_BRIDGE_XYZ_COMMAND_LIMIT = 0.05
_BRIDGE_RPY_COMMAND_LIMIT = 0.25


@dataclass(frozen=True)
class BridgeXTokenizerEpisode:
    """One absolute command-target trajectory shared by all episode chunks."""

    absolute_actions: np.ndarray
    observation_states_absolute: np.ndarray
    dof_mask: np.ndarray


@dataclass(frozen=True)
class BridgeXTokenizerLayout:
    """Named slices for Bridge's active DoFs in an ordered X layout."""

    action_dim: int
    position: slice
    rotation: slice
    gripper: slice
    active_dofs: tuple[str, ...]


@dataclass(frozen=True)
class BridgeXTokenizerStatisticsSamples:
    """Fixed-horizon prediction endpoints and their absolute anchors."""

    predict_action: np.ndarray
    observation_state: np.ndarray


def resolve_bridge_xtokenizer_layout(
    dof_dims: Mapping[str, int],
    *,
    dof_order: Sequence[str] | None = None,
    arm: str = "left",
) -> BridgeXTokenizerLayout:
    """Resolve Bridge channels from explicit DoF names and ordering."""
    if arm not in {"left", "right"}:
        raise ValueError(f"arm must be 'left' or 'right', got {arm!r}.")
    order = list(dof_dims) if dof_order is None else list(dof_order)
    if (
        not order
        or not all(isinstance(name, str) and name for name in order)
        or len(order) != len(set(order))
        or set(order) != set(dof_dims)
    ):
        raise ValueError("dof_order must list every dof_dims key exactly once.")

    slices: dict[str, slice] = {}
    offset = 0
    for raw_key in order:
        dim = int(dof_dims[raw_key])
        if dim <= 0:
            raise ValueError(f"dof_dims[{raw_key!r}] must be positive, got {dim}.")
        key = raw_key.replace("_relative", "")
        if key in slices:
            raise ValueError(f"Duplicate canonical X-Tokenizer DoF name {key!r}.")
        slices[key] = slice(offset, offset + dim)
        offset += dim

    position_key = f"follow_{arm}_ee_cartesian_pos"
    rotation_key = f"follow_{arm}_ee_rotation_6D"
    gripper_key = f"follow_{arm}_gripper"
    required = {
        position_key: 3,
        rotation_key: 6,
        gripper_key: 1,
    }
    for key, expected_dim in required.items():
        segment = slices.get(key)
        if segment is None or segment.stop - segment.start != expected_dim:
            raise ValueError(
                f"X-Tokenizer dof_dims must contain {key!r} with dim {expected_dim}."
            )
    return BridgeXTokenizerLayout(
        action_dim=offset,
        position=slices[position_key],
        rotation=slices[rotation_key],
        gripper=slices[gripper_key],
        active_dofs=tuple(
            raw_key.replace("_relative", "")
            for raw_key in order
            if raw_key.replace("_relative", "") in required
        ),
    )


def compute_relative_rotation_6d_batch(
    absolute_rotation_6d: np.ndarray,
    anchor_rotation_6d: np.ndarray,
) -> np.ndarray:
    """Express absolute 6D rotations relative to one pre-chunk observation."""
    absolute = np.asarray(absolute_rotation_6d, dtype=np.float32)
    anchors = np.asarray(anchor_rotation_6d, dtype=np.float32)
    if absolute.ndim != 3 or absolute.shape[-1] != 6:
        raise ValueError(
            f"absolute_rotation_6d must have shape (B, T, 6), got {absolute.shape}."
        )
    if anchors.shape != (absolute.shape[0], 6):
        raise ValueError(
            f"anchor_rotation_6d must have shape ({absolute.shape[0]}, 6), "
            f"got {anchors.shape}."
        )
    if not np.isfinite(absolute).all() or not np.isfinite(anchors).all():
        raise ValueError("6D rotations must contain only finite values.")
    absolute_matrices = _rotation_6d_to_matrix(absolute)
    anchor_matrices = _rotation_6d_to_matrix(anchors)
    relative_matrices = absolute_matrices @ np.swapaxes(
        anchor_matrices[:, None],
        -1,
        -2,
    )
    return _matrix_to_rotation_6d(relative_matrices)


def build_bridge_xtokenizer_episode(
    states: np.ndarray,
    actions: np.ndarray,
    dof_dims: Mapping[str, int],
    *,
    dof_order: Sequence[str] | None = None,
    arm: str = "left",
) -> BridgeXTokenizerEpisode:
    """Build one absolute command-target trajectory for a complete episode."""
    episode_states = np.asarray(states, dtype=np.float32)
    episode_actions = np.asarray(actions, dtype=np.float32)
    if episode_states.ndim != 2 or episode_states.shape[1] != 7:
        raise ValueError(f"states must have shape (T, 7), got {episode_states.shape}.")
    if episode_actions.ndim != 2 or episode_actions.shape[1] != 7:
        raise ValueError(
            f"actions must have shape (T, 7), got {episode_actions.shape}."
        )
    if len(episode_states) != len(episode_actions):
        raise ValueError("states and actions must have the same episode length.")
    if len(episode_states) == 0:
        raise ValueError("Bridge episodes must contain at least one frame.")
    if not np.isfinite(episode_states).all() or not np.isfinite(episode_actions).all():
        raise ValueError("Bridge states and actions must contain only finite values.")

    layout = resolve_bridge_xtokenizer_layout(
        dof_dims,
        dof_order=dof_order,
        arm=arm,
    )
    clipped_actions = episode_actions.copy()
    clipped_actions[:, :3] = np.clip(
        clipped_actions[:, :3],
        -_BRIDGE_XYZ_COMMAND_LIMIT,
        _BRIDGE_XYZ_COMMAND_LIMIT,
    )
    clipped_actions[:, 3:6] = np.clip(
        clipped_actions[:, 3:6],
        -_BRIDGE_RPY_COMMAND_LIMIT,
        _BRIDGE_RPY_COMMAND_LIMIT,
    )
    clipped_actions[:, 6] = np.clip(clipped_actions[:, 6], 0.0, 1.0)
    absolute_actions = np.zeros(
        (len(episode_actions), layout.action_dim),
        dtype=np.float32,
    )
    target_position = episode_states[0, :3].copy()
    target_rotation = _euler_xyz_to_matrix(episode_states[0, 3:6])
    delta_rotations = _euler_xyz_to_matrix_batch(clipped_actions[:, 3:6])
    for index, delta_rotation in enumerate(delta_rotations):
        measured_position = episode_states[index, :3]
        target_position = (
            measured_position
            + clipped_actions[index, :3]
            - delta_rotation @ measured_position
            + delta_rotation @ target_position
        )
        target_rotation = delta_rotation @ target_rotation
        absolute_actions[index, layout.position] = target_position
        absolute_actions[index, layout.rotation] = _matrix_to_rotation_6d(
            target_rotation
        )
        absolute_actions[index, layout.gripper] = clipped_actions[index, 6]

    observation_states = np.zeros(
        (len(episode_states), layout.action_dim),
        dtype=np.float32,
    )
    observation_states[:, layout.position] = episode_states[:, :3]
    observation_states[:, layout.rotation] = _matrix_to_rotation_6d(
        _euler_xyz_to_matrix_batch(episode_states[:, 3:6])
    )
    observation_states[:, layout.gripper] = episode_states[:, 6:7]
    dof_mask = np.zeros_like(absolute_actions, dtype=np.bool_)
    for segment in (layout.position, layout.rotation, layout.gripper):
        dof_mask[:, segment] = True
    return BridgeXTokenizerEpisode(
        absolute_actions=absolute_actions,
        observation_states_absolute=observation_states,
        dof_mask=dof_mask,
    )


def build_bridge_xtokenizer_statistics_samples(
    states: np.ndarray,
    actions: np.ndarray,
    dof_dims: Mapping[str, int],
    *,
    stats_horizon: int,
    dof_order: Sequence[str] | None = None,
    arm: str = "left",
) -> BridgeXTokenizerStatisticsSamples:
    """Build one endpoint and anchor sample for every valid episode start."""
    if (
        not isinstance(stats_horizon, int)
        or isinstance(stats_horizon, bool)
        or stats_horizon <= 0
    ):
        raise ValueError("stats_horizon must be a positive integer.")
    episode_states = np.asarray(states, dtype=np.float32)
    episode_actions = np.asarray(actions, dtype=np.float32)
    if episode_states.ndim != 2 or episode_states.shape[1] != 7:
        raise ValueError(f"states must have shape (T, 7), got {episode_states.shape}.")
    if episode_actions.ndim != 2 or episode_actions.shape[1] != 7:
        raise ValueError(
            f"actions must have shape (T, 7), got {episode_actions.shape}."
        )
    if len(episode_states) != len(episode_actions):
        raise ValueError("states and actions must have the same episode length.")
    if not np.isfinite(episode_states).all() or not np.isfinite(episode_actions).all():
        raise ValueError("Bridge states and actions must contain only finite values.")

    layout = resolve_bridge_xtokenizer_layout(
        dof_dims,
        dof_order=dof_order,
        arm=arm,
    )
    sample_count = max(len(episode_states) - stats_horizon + 1, 0)
    predict_action = np.zeros((sample_count, layout.action_dim), dtype=np.float32)
    observation_state = np.zeros_like(predict_action)
    if sample_count == 0:
        return BridgeXTokenizerStatisticsSamples(
            predict_action=predict_action,
            observation_state=observation_state,
        )

    episode = build_bridge_xtokenizer_episode(
        episode_states,
        episode_actions,
        dof_dims,
        dof_order=dof_order,
        arm=arm,
    )
    anchors = episode.observation_states_absolute[:sample_count]
    endpoints = episode.absolute_actions[
        stats_horizon - 1 : stats_horizon - 1 + sample_count,
    ]
    predict_action[:, layout.position] = (
        endpoints[:, layout.position] - anchors[:, layout.position]
    )
    predict_action[:, layout.rotation] = compute_relative_rotation_6d_batch(
        endpoints[:, None, layout.rotation],
        anchors[:, layout.rotation],
    )[:, 0]
    predict_action[:, layout.gripper] = endpoints[:, layout.gripper]
    observation_state[:] = anchors
    return BridgeXTokenizerStatisticsSamples(
        predict_action=predict_action,
        observation_state=observation_state,
    )


def _euler_xyz_to_matrix(euler: np.ndarray) -> np.ndarray:
    return _euler_xyz_to_matrix_batch(np.asarray(euler)[None])[0]


def _euler_xyz_to_matrix_batch(euler: np.ndarray) -> np.ndarray:
    euler = np.asarray(euler, dtype=np.float32)
    roll, pitch, yaw = np.moveaxis(euler, -1, 0)
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    shape = euler.shape[:-1]
    rotation_x = np.zeros((*shape, 3, 3), dtype=np.float32)
    rotation_y = np.zeros_like(rotation_x)
    rotation_z = np.zeros_like(rotation_x)
    rotation_x[..., 0, 0] = 1
    rotation_x[..., 1, 1] = cx
    rotation_x[..., 1, 2] = -sx
    rotation_x[..., 2, 1] = sx
    rotation_x[..., 2, 2] = cx
    rotation_y[..., 0, 0] = cy
    rotation_y[..., 0, 2] = sy
    rotation_y[..., 1, 1] = 1
    rotation_y[..., 2, 0] = -sy
    rotation_y[..., 2, 2] = cy
    rotation_z[..., 0, 0] = cz
    rotation_z[..., 0, 1] = -sz
    rotation_z[..., 1, 0] = sz
    rotation_z[..., 1, 1] = cz
    rotation_z[..., 2, 2] = 1
    return rotation_z @ rotation_y @ rotation_x


def _matrix_to_rotation_6d(rotation: np.ndarray) -> np.ndarray:
    return rotation[..., :2, :].reshape(*rotation.shape[:-2], 6).astype(np.float32)


def _rotation_6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation_6d, dtype=np.float32)
    first = rotation[..., :3]
    second = rotation[..., 3:]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_norm <= 1e-8):
        raise ValueError("6D rotation first axis must be non-zero.")
    first = first / first_norm
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(second_norm <= 1e-8):
        raise ValueError("6D rotation axes must be linearly independent.")
    second = second / second_norm
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=-2).astype(np.float32)
