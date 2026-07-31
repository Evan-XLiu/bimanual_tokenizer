# SPDX-FileCopyrightText: 2026 Xiaoshen Han
#
# SPDX-License-Identifier: Apache-2.0

"""Paper-alignment tests for X-Tokenizer observation-relative actions."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
from torch import Tensor

from praxis.codecs.latents.xtokenizer import (
    XTokenizerCodec,
    XTokenizerModelConfig,
)
from praxis.codecs.latents.xtokenizer.action_space import (
    from_observation_relative_actions,
    to_observation_relative_actions,
)

REORDERED_BRIDGE_LAYOUT: dict[str, Any] = {
    "convention": "absolute",
    "dof_dims": {
        "follow_left_gripper": 1,
        "follow_left_ee_cartesian_pos": 3,
        "follow_left_ee_rotation_6D": 6,
    },
    "dof_order": [
        "follow_left_gripper",
        "follow_left_ee_cartesian_pos",
        "follow_left_ee_rotation_6D",
    ],
}


def _reordered_bridge_config(**overrides: object) -> XTokenizerModelConfig:
    values: dict[str, Any] = {
        "action_dim": 10,
        "action_horizon": 2,
        "max_action_horizon": 2,
        "compression_ratio": 2,
        "action_layout": REORDERED_BRIDGE_LAYOUT,
        "hidden_dim": 8,
        "latent_dim": 8,
        "encoder_num_heads": 2,
        "encoder_depth": 1,
        "decoder_num_heads": 2,
        "decoder_depth": 1,
        "mam_num_heads": 2,
        "mam_depth": 1,
        "prediction_num_heads": 2,
        "prediction_depth": 1,
        "dropout": 0.0,
        "codebook_size": 8,
        "num_quantizers": 2,
        "dead_code_threshold": 0,
        "kmeans_init": False,
        "embodiment_vocab_size": 2,
        "vl_feature_dim": 4,
    }
    values.update(overrides)
    return XTokenizerModelConfig.from_dict(values)


def _reordered_slices() -> tuple[slice, slice]:
    config = _reordered_bridge_config()
    assert config.position_slice == slice(1, 4)
    assert config.rotation_slice == slice(4, 10)
    return config.position_slice, config.rotation_slice


def _rotation_x(angle: float) -> Tensor:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=torch.float64,
    )


def _rotation_y(angle: float) -> Tensor:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return torch.tensor(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=torch.float64,
    )


def _rotation_z(angle: float) -> Tensor:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return torch.tensor(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )


def _rotation_6d(rotation: Tensor) -> Tensor:
    return rotation[:2].reshape(6)


def _bridge_action(
    *,
    gripper: float,
    position: tuple[float, float, float],
    rotation: Tensor,
) -> Tensor:
    return torch.cat(
        (
            torch.tensor([gripper], dtype=torch.float64),
            torch.tensor(position, dtype=torch.float64),
            _rotation_6d(rotation),
        )
    )


def test_to_observation_relative_actions_uses_fixed_anchor_and_paper_so3_order() -> (
    None
):
    position_slice, rotation_slice = _reordered_slices()
    observation_rotation = _rotation_x(0.4) @ _rotation_y(-0.3)
    expected_relative_rotation = _rotation_z(0.6) @ _rotation_x(-0.2)
    absolute_rotation = expected_relative_rotation @ observation_rotation
    observation = _bridge_action(
        gripper=0.25,
        position=(10.0, 20.0, 30.0),
        rotation=observation_rotation,
    ).reshape(1, 1, 10)
    actions = _bridge_action(
        gripper=0.75,
        position=(11.5, 18.0, 34.0),
        rotation=absolute_rotation,
    ).reshape(1, 1, 10)

    relative = to_observation_relative_actions(
        actions,
        observation,
        position_slice=position_slice,
        rotation_slice=rotation_slice,
    )

    torch.testing.assert_close(
        relative[..., position_slice],
        torch.tensor([[[1.5, -2.0, 4.0]]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        relative[..., rotation_slice],
        _rotation_6d(expected_relative_rotation).reshape(1, 1, 6),
    )
    torch.testing.assert_close(relative[..., :1], actions[..., :1])

    wrong_left_multiplication = observation_rotation.T @ absolute_rotation
    assert not torch.allclose(
        relative[0, 0, rotation_slice],
        _rotation_6d(wrong_left_multiplication),
    )


def test_from_observation_relative_actions_uses_paper_so3_order() -> None:
    position_slice, rotation_slice = _reordered_slices()
    observation_rotation = _rotation_x(0.4) @ _rotation_y(-0.3)
    relative_rotation = _rotation_z(0.6) @ _rotation_x(-0.2)
    expected_absolute_rotation = relative_rotation @ observation_rotation
    observation = _bridge_action(
        gripper=0.25,
        position=(10.0, 20.0, 30.0),
        rotation=observation_rotation,
    ).reshape(1, 1, 10)
    relative = _bridge_action(
        gripper=-0.5,
        position=(1.5, -2.0, 4.0),
        rotation=relative_rotation,
    ).reshape(1, 1, 10)

    absolute = from_observation_relative_actions(
        relative,
        observation,
        position_slice=position_slice,
        rotation_slice=rotation_slice,
    )

    torch.testing.assert_close(
        absolute[..., position_slice],
        torch.tensor([[[11.5, 18.0, 34.0]]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        absolute[..., rotation_slice],
        _rotation_6d(expected_absolute_rotation).reshape(1, 1, 6),
    )
    torch.testing.assert_close(absolute[..., :1], relative[..., :1])

    wrong_left_multiplication = observation_rotation @ relative_rotation
    assert not torch.allclose(
        absolute[0, 0, rotation_slice],
        _rotation_6d(wrong_left_multiplication),
    )


def test_observation_relative_action_transform_roundtrips_all_reordered_channels() -> (
    None
):
    position_slice, rotation_slice = _reordered_slices()
    observation_rotation = _rotation_y(0.35) @ _rotation_z(-0.25)
    observation = _bridge_action(
        gripper=0.1,
        position=(-2.0, 4.0, 8.0),
        rotation=observation_rotation,
    ).reshape(1, 1, 10)
    relative_rotations = (
        _rotation_x(0.2) @ _rotation_z(0.3),
        _rotation_z(-0.5) @ _rotation_y(0.45),
    )
    actions = torch.stack(
        (
            _bridge_action(
                gripper=0.4,
                position=(-1.0, 3.0, 10.0),
                rotation=relative_rotations[0] @ observation_rotation,
            ),
            _bridge_action(
                gripper=0.9,
                position=(3.0, 9.0, 7.0),
                rotation=relative_rotations[1] @ observation_rotation,
            ),
        )
    ).unsqueeze(0)

    relative = to_observation_relative_actions(
        actions,
        observation,
        position_slice=position_slice,
        rotation_slice=rotation_slice,
    )
    restored = from_observation_relative_actions(
        relative,
        observation,
        position_slice=position_slice,
        rotation_slice=rotation_slice,
    )

    torch.testing.assert_close(restored, actions, atol=1e-12, rtol=1e-12)


def test_relative_action_uses_one_pre_chunk_observation_not_temporal_delta() -> None:
    position_slice, rotation_slice = _reordered_slices()
    identity = torch.eye(3, dtype=torch.float64)
    observation = _bridge_action(
        gripper=0.0,
        position=(10.0, 20.0, 30.0),
        rotation=identity,
    ).reshape(1, 1, 10)
    actions = torch.stack(
        (
            _bridge_action(
                gripper=0.2,
                position=(11.0, 20.0, 30.0),
                rotation=identity,
            ),
            _bridge_action(
                gripper=0.8,
                position=(13.0, 20.0, 30.0),
                rotation=identity,
            ),
        )
    ).unsqueeze(0)

    relative = to_observation_relative_actions(
        actions,
        observation,
        position_slice=position_slice,
        rotation_slice=rotation_slice,
    )
    temporal_delta = torch.stack(
        (
            actions[:, 0, position_slice] - observation[:, 0, position_slice],
            actions[:, 1, position_slice] - actions[:, 0, position_slice],
        ),
        dim=1,
    )

    torch.testing.assert_close(
        relative[..., position_slice],
        torch.tensor([[[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        temporal_delta,
        torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], dtype=torch.float64),
    )
    assert not torch.equal(relative[..., position_slice], temporal_delta)
    torch.testing.assert_close(relative[..., :1], actions[..., :1])


def test_observation_relative_transform_applies_every_named_pose_group() -> None:
    identity = _rotation_6d(torch.eye(3, dtype=torch.float64))
    observation = torch.cat(
        (
            torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
            identity,
            torch.tensor([0.25], dtype=torch.float64),
            torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64),
            identity,
            torch.tensor([0.75], dtype=torch.float64),
        )
    ).reshape(1, 1, 20)
    actions = observation.expand(1, 2, 20).clone()
    actions[..., 0:3] += torch.tensor([1.0, 3.0, 5.0])
    actions[..., 10:13] += torch.tensor([2.0, 4.0, 6.0])

    relative = to_observation_relative_actions(
        actions,
        observation,
        position_slice=None,
        rotation_slice=None,
        position_slices=(slice(0, 3), slice(10, 13)),
        rotation_slices=(slice(3, 9), slice(13, 19)),
    )

    torch.testing.assert_close(
        relative[..., 0:3],
        torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64).expand(1, 2, 3),
    )
    torch.testing.assert_close(
        relative[..., 10:13],
        torch.tensor([2.0, 4.0, 6.0], dtype=torch.float64).expand(1, 2, 3),
    )
    torch.testing.assert_close(relative[..., 9:10], actions[..., 9:10])
    torch.testing.assert_close(relative[..., 19:20], actions[..., 19:20])


@pytest.mark.parametrize("method", ["encode", "decode"])
def test_codec_inference_requires_observation_anchor(method: str) -> None:
    codec = XTokenizerCodec(config=_reordered_bridge_config()).eval()
    actions = torch.zeros(1, codec.action_horizon, codec.action_dim)
    actions[..., 4:10] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    indices = torch.zeros(
        1,
        codec.latent_horizon,
        codec.num_quantizers,
        dtype=torch.long,
    )

    with pytest.raises(ValueError, match="anchor|observation.state"):
        if method == "encode":
            codec.encode(actions)
        else:
            codec.decode(indices)
