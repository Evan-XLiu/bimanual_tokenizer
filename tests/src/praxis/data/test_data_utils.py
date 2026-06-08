# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from praxis.core.types import FeatureSpec, FeatureType, NormMode, PolicyIO
from praxis.data.utils import resolve_normalization_modes_for_io


def _io() -> PolicyIO:
    return PolicyIO(
        obs_features=[
            FeatureSpec(
                key="observation.state",
                type=FeatureType.STATE,
                shape=(8,),
                norm_mode=NormMode.MIN_MAX,
            ),
            FeatureSpec(
                key="observation.images.front",
                type=FeatureType.IMAGE,
                shape=(3, 96, 96),
                norm_mode=NormMode.MEAN_STD,
            ),
        ],
        action_feature=FeatureSpec(
            key="robot_action",
            type=FeatureType.ACTION,
            shape=(7,),
            norm_mode=NormMode.MIN_MAX,
        ),
        n_obs_steps=1,
        n_action_steps=1,
    )


def test_resolve_normalization_modes_for_io_uses_metadata_then_fallback() -> None:
    io = _io()
    meta_features = {
        "observation.state": {
            "shape": [8],
            "dtype": "float32",
            "normalization_mode": "mean_std",
        },
        "observation.images.front": {
            "shape": [96, 96, 3],
            "dtype": "image",
            "norm_mode": "min_max",
        },
        "robot_action": {"shape": [7], "dtype": "float32"},
    }
    out = resolve_normalization_modes_for_io(meta_features, io)
    assert out["observation.state"] == NormMode.MEAN_STD
    assert out["observation.images.front"] == NormMode.MIN_MAX
    assert out["robot_action"] == NormMode.MIN_MAX


def test_resolve_normalization_modes_for_io_none_io() -> None:
    assert resolve_normalization_modes_for_io({}, None) == {}
