"""Tests for dataset feature metadata helpers."""

from __future__ import annotations

import pytest

from praxis.core.types import NormMode
from praxis.data.utils import (
    collect_action_feature_candidates,
    resolve_action_feature_from_metadata,
)


def test_collect_action_feature_candidates_filters_reserved_and_observations() -> None:
    features = {
        "observation.state": {"dtype": "float32", "shape": [7]},
        "episode_index": {"dtype": "int64", "shape": []},
        "next.reward": {"dtype": "float32", "shape": [1]},
        "action_is_pad": {"dtype": "bool", "shape": [8]},
        "robot_action": {
            "dtype": "float32",
            "shape": [3],
            "info": {"norm_mode": "mean_std"},
        },
        "trajectory": {"dtype": "float32", "shape": [2, 3]},
    }

    candidates = collect_action_feature_candidates(features)

    assert [candidate.key for candidate in candidates] == ["robot_action"]
    assert candidates[0].shape == (3,)
    assert candidates[0].norm_mode == NormMode.MEAN_STD


def test_resolve_action_feature_prefers_canonical_action_key() -> None:
    features = {
        "action": {"dtype": "float32", "shape": [7]},
        "robot_action": {"dtype": "float32", "shape": [3]},
    }

    action_feature = resolve_action_feature_from_metadata(features)

    assert action_feature.key == "action"
    assert action_feature.shape == (7,)


def test_resolve_action_feature_prefers_single_action_named_key() -> None:
    features = {
        "controls": {"dtype": "float32", "shape": [7]},
        "robot_action": {"dtype": "float32", "shape": [3]},
    }

    action_feature = resolve_action_feature_from_metadata(features)

    assert action_feature.key == "robot_action"


def test_resolve_action_feature_rejects_ambiguous_candidates() -> None:
    features = {
        "controls": {"dtype": "float32", "shape": [7]},
        "commands": {"dtype": "float32", "shape": [3]},
    }

    with pytest.raises(ValueError, match="Ambiguous action feature inference"):
        resolve_action_feature_from_metadata(features, context="unit test metadata")


def test_resolve_action_feature_rejects_missing_candidate() -> None:
    features = {
        "observation.state": {"dtype": "float32", "shape": [7]},
        "episode_index": {"dtype": "int64", "shape": []},
    }

    with pytest.raises(KeyError, match="Unable to infer action feature"):
        resolve_action_feature_from_metadata(features)
