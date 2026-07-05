# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for praxis.core.types."""

from __future__ import annotations

import pytest

from praxis.core.types import FeatureSpec, FeatureType, NormMode, PolicyIO


class TestFeatureSpec:
    def test_default_norm_mode(self):
        spec = FeatureSpec(key="x", type=FeatureType.STATE, shape=(4,))
        assert spec.norm_mode == NormMode.NONE

    def test_image_spec(self):
        spec = FeatureSpec(
            key="obs.img",
            type=FeatureType.IMAGE,
            shape=(3, 96, 96),
            norm_mode=NormMode.MEAN_STD,
        )
        assert spec.shape == (3, 96, 96)


class TestFeatureSpecSerialization:
    def test_roundtrip(self):
        spec = FeatureSpec(
            key="observation.images.top",
            type=FeatureType.IMAGE,
            shape=(3, 96, 96),
            norm_mode=NormMode.MEAN_STD,
        )
        d = spec.to_dict()
        loaded = FeatureSpec.from_dict(d)
        assert loaded.key == spec.key
        assert loaded.type == spec.type
        assert loaded.shape == spec.shape
        assert loaded.norm_mode == spec.norm_mode


class TestPolicyIOSerialization:
    def test_roundtrip(self, image_io: PolicyIO):
        d = image_io.to_dict()
        loaded = PolicyIO.from_dict(d)
        assert len(loaded.obs_features) == len(image_io.obs_features)
        assert loaded.action_feature.key == image_io.action_feature.key
        assert loaded.action_feature.shape == image_io.action_feature.shape
        assert loaded.n_obs_steps == image_io.n_obs_steps
        assert loaded.n_action_steps == image_io.n_action_steps
        assert loaded.fps == image_io.fps
        for orig, loaded_f in zip(
            image_io.obs_features, loaded.obs_features, strict=False
        ):
            assert orig.key == loaded_f.key
            assert orig.type == loaded_f.type
            assert orig.shape == loaded_f.shape
            assert orig.norm_mode == loaded_f.norm_mode

    def test_roundtrip_no_action(self):
        io = PolicyIO(obs_features=[], action_feature=None)
        d = io.to_dict()
        loaded = PolicyIO.from_dict(d)
        assert loaded.action_feature is None

    def test_image_dims(self, image_io: PolicyIO):
        h, w = image_io.image_dims
        assert h == 96
        assert w == 96

    def test_image_dims_no_images(self):
        io = PolicyIO(
            obs_features=[
                FeatureSpec(
                    key="observation.state", type=FeatureType.STATE, shape=(8,)
                ),
            ]
        )
        with pytest.raises(ValueError, match="no image features"):
            _ = io.image_dims


class TestPolicyIO:
    def test_state_and_image_filters(self, image_io: PolicyIO):
        assert len(image_io.state_features) == 1
        assert len(image_io.image_features) == 2
        assert image_io.state_features[0].key == "observation.state"

    def test_obs_keys(self, image_io: PolicyIO):
        keys = image_io.obs_keys
        assert "observation.images.image" in keys
        assert "observation.images.image2" in keys
        assert "observation.state" in keys

    def test_action_dim(self, simple_io: PolicyIO):
        assert simple_io.action_dim == 7

    def test_action_dim_raises_without_action(self):
        io = PolicyIO(obs_features=[], action_feature=None)
        with pytest.raises(ValueError):
            _ = io.action_dim

    def test_delta_timestamps_obs(self, simple_io: PolicyIO):
        dt = simple_io.build_delta_timestamps()
        # n_obs_steps=2, fps=10 → obs deltas = [-0.1, 0.0]
        assert dt["observation.state"] == pytest.approx([-0.1, 0.0])

    def test_delta_timestamps_action(self, simple_io: PolicyIO):
        dt = simple_io.build_delta_timestamps()
        # n_action_steps=8, fps=10 → [0.0, 0.1, ..., 0.7]
        expected = [round(i * 0.1, 6) for i in range(8)]
        assert dt["action"] == pytest.approx(expected)

    def test_delta_timestamps_keys(self, image_io: PolicyIO):
        dt = image_io.build_delta_timestamps()
        assert set(dt.keys()) == {
            "observation.images.image",
            "observation.images.image2",
            "observation.state",
            "action",
        }
