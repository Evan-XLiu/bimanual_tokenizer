# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from praxis.data import bridge_xtokenizer

DOF_DIMS = {
    "follow_left_ee_cartesian_pos_relative": 3,
    "follow_left_ee_rotation_6D_relative": 6,
    "follow_left_gripper": 1,
    "follow_right_ee_cartesian_pos_relative": 3,
    "follow_right_ee_rotation_6D_relative": 6,
    "follow_right_gripper": 1,
    "velocity_decomposed": 3,
    "height": 1,
    "head_actions": 2,
}


def test_bridge_episode_trajectory_is_shared_across_overlapping_chunks() -> None:
    states = np.zeros((4, 7), dtype=np.float32)
    states[:, 0] = [0.0, 0.5, 0.7, 0.9]
    actions = np.zeros((4, 7), dtype=np.float32)
    actions[:, 0] = 0.01

    build_episode = getattr(
        bridge_xtokenizer,
        "build_bridge_xtokenizer_episode",
        None,
    )
    assert build_episode is not None, "Bridge conversion must operate episode-wide"

    episode = build_episode(states, actions, DOF_DIMS)
    chunk_at_zero = episode.absolute_actions[:3]
    chunk_at_one = episode.absolute_actions[1:4]

    np.testing.assert_allclose(chunk_at_zero[1:], chunk_at_one[:-1], atol=0)
    np.testing.assert_allclose(
        episode.absolute_actions[:, :3],
        [[0.01, 0.0, 0.0], [0.02, 0.0, 0.0], [0.03, 0.0, 0.0], [0.04, 0.0, 0.0]],
        atol=1e-7,
    )


def test_bridge_episode_rotation_uses_measured_ee_as_controller_pivot() -> None:
    states = np.zeros((2, 7), dtype=np.float32)
    states[0, :3] = [1.0, 0.0, 0.0]
    states[1, :3] = [0.0, 1.0, 0.0]
    actions = np.zeros((2, 7), dtype=np.float32)
    actions[1, 5] = 0.1

    episode = bridge_xtokenizer.build_bridge_xtokenizer_episode(
        states,
        actions,
        DOF_DIMS,
    )

    cosine = np.cos(0.1)
    sine = np.sin(0.1)
    delta_rotation = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    pivot = states[1, :3]
    previous_target = episode.absolute_actions[0, :3]
    expected = pivot - delta_rotation @ pivot + delta_rotation @ previous_target

    np.testing.assert_allclose(episode.absolute_actions[1, :3], expected, atol=1e-6)


def test_bridge_episode_world_axis_rotations_left_compose() -> None:
    states = np.zeros((2, 7), dtype=np.float32)
    actions = np.zeros((2, 7), dtype=np.float32)
    actions[0, 3] = 0.1
    actions[1, 4] = 0.2

    episode = bridge_xtokenizer.build_bridge_xtokenizer_episode(
        states,
        actions,
        DOF_DIMS,
    )

    roll = bridge_xtokenizer._euler_xyz_to_matrix(
        np.array([0.1, 0.0, 0.0], dtype=np.float32)
    )
    pitch = bridge_xtokenizer._euler_xyz_to_matrix(
        np.array([0.0, 0.2, 0.0], dtype=np.float32)
    )
    actual = _rotation_6d_to_matrix(episode.absolute_actions[1, 3:9])
    np.testing.assert_allclose(actual, pitch @ roll, atol=1e-6)


def test_bridge_episode_clips_raw_motion_commands_like_controller() -> None:
    states = np.zeros((1, 7), dtype=np.float32)
    actions = np.array(
        [[0.1, -0.1, 0.08, 0.5, -0.4, 0.3, 0.5]],
        dtype=np.float32,
    )

    episode = bridge_xtokenizer.build_bridge_xtokenizer_episode(
        states,
        actions,
        DOF_DIMS,
    )

    np.testing.assert_allclose(
        episode.absolute_actions[0, :3],
        [0.05, -0.05, 0.05],
        atol=1e-7,
    )
    expected_rotation = bridge_xtokenizer._euler_xyz_to_matrix(
        np.array([0.25, -0.25, 0.25], dtype=np.float32)
    )
    actual_rotation = _rotation_6d_to_matrix(episode.absolute_actions[0, 3:9])
    np.testing.assert_allclose(actual_rotation, expected_rotation, atol=1e-6)


def test_bridge_episode_gripper_is_clipped_absolute_target() -> None:
    states = np.zeros((3, 7), dtype=np.float32)
    actions = np.zeros((3, 7), dtype=np.float32)
    actions[:, 6] = [-0.2, 1.2, 0.4]

    episode = bridge_xtokenizer.build_bridge_xtokenizer_episode(
        states,
        actions,
        DOF_DIMS,
    )

    np.testing.assert_allclose(
        episode.absolute_actions[:, 9],
        [0.0, 1.0, 0.4],
        atol=0,
    )


def test_bridge_binding_rejects_non_7d_inputs() -> None:
    with pytest.raises(ValueError, match=r"states must have shape \(T, 7\)"):
        bridge_xtokenizer.build_bridge_xtokenizer_episode(
            np.zeros((2, 6)),
            np.zeros((2, 7)),
            DOF_DIMS,
        )

    with pytest.raises(ValueError, match=r"actions must have shape \(T, 7\)"):
        bridge_xtokenizer.build_bridge_xtokenizer_episode(
            np.zeros((2, 7)),
            np.zeros((2, 6)),
            DOF_DIMS,
        )


def test_bridge_episode_packs_observation_and_masks_only_active_dofs() -> None:
    states = np.array(
        [[0.3, -0.2, 0.1, 0.1, -0.2, 0.3, 0.4]],
        dtype=np.float32,
    )
    actions = np.zeros((1, 7), dtype=np.float32)

    episode = bridge_xtokenizer.build_bridge_xtokenizer_episode(
        states,
        actions,
        DOF_DIMS,
    )

    np.testing.assert_allclose(
        episode.observation_states_absolute[0, :3],
        states[0, :3],
        atol=0,
    )
    np.testing.assert_allclose(
        _rotation_6d_to_matrix(episode.observation_states_absolute[0, 3:9]),
        bridge_xtokenizer._euler_xyz_to_matrix(states[0, 3:6]),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        episode.observation_states_absolute[0, 9],
        states[0, 6],
        atol=0,
    )
    assert episode.dof_mask.shape == (1, 26)
    assert episode.dof_mask[:, :10].all()
    assert not episode.dof_mask[:, 10:].any()
    assert not episode.absolute_actions[:, 10:].any()


def test_relative_rotation_6d_recovers_world_axis_delta_from_absolute_pose() -> None:
    states = np.repeat(
        np.array([[0.0, 0.0, 0.0, 0.2, -0.3, 0.4, 0.5]], dtype=np.float32),
        2,
        axis=0,
    )
    actions = np.array(
        [
            [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.5],
            [0.0, 0.0, 0.0, 0.0, -0.2, 0.0, 0.5],
        ],
        dtype=np.float32,
    )
    episode = bridge_xtokenizer.build_bridge_xtokenizer_episode(
        states,
        actions,
        DOF_DIMS,
    )
    relative = bridge_xtokenizer.compute_relative_rotation_6d_batch(
        episode.absolute_actions[None, :, 3:9],
        episode.observation_states_absolute[None, 0, 3:9],
    )

    first_delta = bridge_xtokenizer._euler_xyz_to_matrix(actions[0, 3:6])
    second_delta = bridge_xtokenizer._euler_xyz_to_matrix(actions[1, 3:6])
    np.testing.assert_allclose(
        _rotation_6d_to_matrix(relative[0, 0]),
        first_delta,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        _rotation_6d_to_matrix(relative[0, 1]),
        second_delta @ first_delta,
        atol=2e-6,
    )


def test_statistics_samples_include_every_complete_fixed_horizon_window() -> None:
    states = np.array(
        [
            [0.0, 0.0, 0.0, 0.2, -0.3, 0.4, 0.1],
            [1.0, 2.0, 3.0, -0.4, 0.25, -0.1, 0.2],
            [4.0, -1.0, 2.0, 0.35, 0.15, -0.45, 0.3],
            [9.0, 8.0, -2.0, -0.2, -0.5, 0.3, 0.4],
        ],
        dtype=np.float32,
    )
    actions = np.array(
        [
            [10.0, -7.0, 4.0, 0.4, -0.2, 0.3, 0.11],
            [-9.0, 8.0, -3.0, -0.1, 0.35, -0.2, 0.22],
            [2.0, -5.0, 7.0, 0.25, -0.15, 0.4, 0.33],
            [-6.0, 1.0, -2.0, -0.3, 0.2, -0.1, 0.44],
        ],
        dtype=np.float32,
    )

    samples = bridge_xtokenizer.build_bridge_xtokenizer_statistics_samples(
        states,
        actions,
        DOF_DIMS,
        stats_horizon=2,
    )

    assert samples.predict_action.shape == (3, 26)
    assert samples.observation_state.shape == (3, 26)
    episode = bridge_xtokenizer.build_bridge_xtokenizer_episode(
        states,
        actions,
        DOF_DIMS,
    )
    for start in range(3):
        endpoint = episode.absolute_actions[start + 1]
        anchor = episode.observation_states_absolute[start]
        expected = endpoint.copy()
        expected[:3] = endpoint[:3] - anchor[:3]
        endpoint_rotation = _rotation_6d_to_matrix(endpoint[3:9])
        anchor_rotation = _rotation_6d_to_matrix(anchor[3:9])
        expected[3:9] = (endpoint_rotation @ anchor_rotation.T)[:2].reshape(6)

        np.testing.assert_allclose(samples.predict_action[start], expected, atol=2e-6)
        np.testing.assert_allclose(samples.observation_state[start], anchor, atol=2e-6)

    np.testing.assert_allclose(samples.predict_action[:, 9], [0.22, 0.33, 0.44])


@pytest.mark.parametrize(("episode_length", "expected_count"), [(2, 1), (3, 2)])
def test_statistics_sample_count_matches_native_complete_windows(
    episode_length: int,
    expected_count: int,
) -> None:
    samples = bridge_xtokenizer.build_bridge_xtokenizer_statistics_samples(
        np.zeros((episode_length, 7), dtype=np.float32),
        np.zeros((episode_length, 7), dtype=np.float32),
        DOF_DIMS,
        stats_horizon=2,
    )

    assert len(samples.predict_action) == expected_count
    assert len(samples.observation_state) == expected_count


@pytest.mark.parametrize("stats_horizon", [0, -1, True])
def test_statistics_samples_require_positive_integer_horizon(
    stats_horizon: int,
) -> None:
    with pytest.raises(ValueError, match="stats_horizon must be a positive integer"):
        bridge_xtokenizer.build_bridge_xtokenizer_statistics_samples(
            np.zeros((3, 7), dtype=np.float32),
            np.zeros((3, 7), dtype=np.float32),
            DOF_DIMS,
            stats_horizon=stats_horizon,
        )


def _rotation_6d_to_matrix(rotation: np.ndarray) -> np.ndarray:
    first = rotation[:3] / np.linalg.norm(rotation[:3])
    second = rotation[3:] - np.dot(first, rotation[3:]) * first
    second /= np.linalg.norm(second)
    return np.stack((first, second, np.cross(first, second)))
