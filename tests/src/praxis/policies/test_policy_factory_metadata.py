# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for metadata -> PolicyIO parsing."""

from __future__ import annotations

from omegaconf import OmegaConf

from praxis.core.types import FeatureType, NormMode
from praxis.policies.factory import make_io_from_metadata


class TestPolicyFactoryMetadata:
    def test_bicat_explicit_joint_action_and_observation_selection(self):
        features = {
            "observation.state_joint_angles": {"dtype": "float32", "shape": [14]},
            "observation.state_eef_absolute": {"dtype": "float32", "shape": [16]},
            "observation.images.top": {"dtype": "video", "shape": [360, 640, 3]},
            "action_eef_absolute": {"dtype": "float32", "shape": [16]},
            "action_eef_delta": {"dtype": "float32", "shape": [16]},
            "action_joint_angles": {"dtype": "float32", "shape": [14]},
        }
        cfg = OmegaConf.create({"n_obs_steps": 1, "action_horizon": 32})
        io = make_io_from_metadata(
            features,
            cfg,
            action_key="action_joint_angles",
            observation_keys=[
                "observation.state_joint_angles",
                "observation.images.top",
            ],
        )
        assert io.action_feature.key == "action_joint_angles"
        assert io.action_dim == 14
        assert io.obs_keys == [
            "observation.state_joint_angles",
            "observation.images.top",
        ]
        assert io.build_delta_timestamps()["action_joint_angles"] == [
            round(i / 30, 6) for i in range(32)
        ]

    def test_make_io_from_metadata_preserves_camera_keys(self):
        meta_features = {
            "observation.state": {"dtype": "float32", "shape": [8]},
            "observation.images.front_camera": {"dtype": "image", "shape": [96, 96, 3]},
            "observation.wrist_camera": {"dtype": "video", "shape": [84, 84, 3]},
            "action": {"dtype": "float32", "shape": [7]},
        }
        policy_cfg = OmegaConf.create(
            {"n_obs_steps": 2, "n_action_steps": 8, "action_horizon": 16}
        )

        io = make_io_from_metadata(meta_features, policy_cfg)

        assert io.n_obs_steps == 2
        assert io.n_action_steps == 16
        assert io.action_feature is not None
        assert io.action_feature.key == "action"
        assert io.action_feature.type == FeatureType.ACTION
        assert io.action_feature.shape == (7,)
        assert io.action_feature.norm_mode == NormMode.MIN_MAX

        by_key = {feat.key: feat for feat in io.obs_features}
        assert set(by_key) == {
            "observation.state",
            "observation.images.front_camera",
            "observation.wrist_camera",
        }
        assert by_key["observation.state"].type == FeatureType.STATE
        assert by_key["observation.state"].shape == (8,)
        assert by_key["observation.state"].norm_mode == NormMode.MIN_MAX

        assert by_key["observation.images.front_camera"].type == FeatureType.IMAGE
        assert by_key["observation.images.front_camera"].shape == (3, 96, 96)
        assert by_key["observation.images.front_camera"].norm_mode == NormMode.MEAN_STD

        assert by_key["observation.wrist_camera"].type == FeatureType.IMAGE
        assert by_key["observation.wrist_camera"].shape == (3, 84, 84)
        assert by_key["observation.wrist_camera"].norm_mode == NormMode.MEAN_STD

    def test_make_io_from_metadata_keeps_channel_first_image_shape(self):
        meta_features = {
            "observation.images.front_camera": {"dtype": "image", "shape": [3, 96, 96]},
            "action": {"dtype": "float32", "shape": [7]},
        }
        policy_cfg = OmegaConf.create(
            {"n_obs_steps": 2, "n_action_steps": 8, "action_horizon": 8}
        )

        io = make_io_from_metadata(meta_features, policy_cfg)
        by_key = {feat.key: feat for feat in io.obs_features}
        assert by_key["observation.images.front_camera"].shape == (3, 96, 96)

    def test_make_io_from_metadata_supports_nondefault_action_key(self):
        meta_features = {
            "observation.state": {"dtype": "float32", "shape": [8]},
            "robot_action": {"dtype": "float32", "shape": [7]},
            "episode_index": {"dtype": "int64", "shape": []},
        }
        policy_cfg = OmegaConf.create(
            {"n_obs_steps": 1, "n_action_steps": 8, "action_horizon": 8}
        )

        io = make_io_from_metadata(meta_features, policy_cfg)

        assert io.action_feature is not None
        assert io.action_feature.key == "robot_action"
        assert io.action_feature.shape == (7,)

    def test_make_io_from_metadata_uses_feature_norm_mode_when_available(self):
        meta_features = {
            "observation.state": {
                "dtype": "float32",
                "shape": [8],
                "normalization_mode": "mean_std",
            },
            "observation.images.front_camera": {
                "dtype": "image",
                "shape": [96, 96, 3],
                "norm_mode": "min_max",
            },
            "robot_action": {
                "dtype": "float32",
                "shape": [7],
                "normalization_mode": "MEAN_STD",
            },
        }
        policy_cfg = OmegaConf.create(
            {"n_obs_steps": 2, "n_action_steps": 8, "action_horizon": 8}
        )

        io = make_io_from_metadata(meta_features, policy_cfg)
        by_key = {feat.key: feat for feat in io.obs_features}

        assert by_key["observation.state"].norm_mode == NormMode.MEAN_STD
        assert by_key["observation.images.front_camera"].norm_mode == NormMode.MIN_MAX
        assert io.action_feature is not None
        assert io.action_feature.norm_mode == NormMode.MEAN_STD

    def test_make_io_from_metadata_filters_observation_keys(self):
        meta_features = {
            "observation.state": {"dtype": "float32", "shape": [8]},
            "observation.images.image_0": {"dtype": "image", "shape": [96, 96, 3]},
            "observation.images.image_1": {"dtype": "image", "shape": [96, 96, 3]},
            "action": {"dtype": "float32", "shape": [7]},
        }
        policy_cfg = OmegaConf.create(
            {"n_obs_steps": 2, "n_action_steps": 8, "action_horizon": 8}
        )

        io = make_io_from_metadata(
            meta_features,
            policy_cfg,
            observation_keys=["observation.state", "observation.images.image_0"],
        )

        assert [feature.key for feature in io.obs_features] == [
            "observation.state",
            "observation.images.image_0",
        ]
