"""Shared fixtures for praxis tests."""

from __future__ import annotations

import pytest
import torch

from praxis.core.types import FeatureSpec, FeatureType, NormMode, PolicyIO


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def simple_io() -> PolicyIO:
    """PolicyIO with 1 state feature and 1 action feature (no images). Good for fast unit tests."""
    return PolicyIO(
        obs_features=[
            FeatureSpec(
                key="observation.state",
                type=FeatureType.STATE,
                shape=(8,),
                norm_mode=NormMode.MIN_MAX,
            ),
        ],
        action_feature=FeatureSpec(
            key="action",
            type=FeatureType.ACTION,
            shape=(7,),
            norm_mode=NormMode.MIN_MAX,
        ),
        n_obs_steps=2,
        n_action_steps=8,
        fps=10,
    )


@pytest.fixture
def image_io() -> PolicyIO:
    """PolicyIO with state + 2 image features (like LIBERO)."""
    return PolicyIO(
        obs_features=[
            FeatureSpec(
                key="observation.images.image",
                type=FeatureType.IMAGE,
                shape=(3, 96, 96),
                norm_mode=NormMode.MEAN_STD,
            ),
            FeatureSpec(
                key="observation.images.image2",
                type=FeatureType.IMAGE,
                shape=(3, 96, 96),
                norm_mode=NormMode.MEAN_STD,
            ),
            FeatureSpec(
                key="observation.state",
                type=FeatureType.STATE,
                shape=(8,),
                norm_mode=NormMode.MIN_MAX,
            ),
        ],
        action_feature=FeatureSpec(
            key="action",
            type=FeatureType.ACTION,
            shape=(7,),
            norm_mode=NormMode.MIN_MAX,
        ),
        n_obs_steps=2,
        n_action_steps=8,
        fps=10,
    )


@pytest.fixture
def dummy_stats() -> dict[str, dict[str, torch.Tensor]]:
    """Normalization stats matching the image_io fixture."""
    return {
        "observation.state": {
            "min": torch.zeros(8),
            "max": torch.ones(8),
            "mean": torch.full((8,), 0.5),
            "std": torch.full((8,), 0.25),
        },
        "observation.images.image": {
            "min": torch.zeros(3, 1, 1),
            "max": torch.ones(3, 1, 1),
            "mean": torch.full((3, 1, 1), 0.5),
            "std": torch.full((3, 1, 1), 0.25),
        },
        "observation.images.image2": {
            "min": torch.zeros(3, 1, 1),
            "max": torch.ones(3, 1, 1),
            "mean": torch.full((3, 1, 1), 0.5),
            "std": torch.full((3, 1, 1), 0.25),
        },
        "action": {
            "min": -torch.ones(7),
            "max": torch.ones(7),
            "mean": torch.zeros(7),
            "std": torch.full((7,), 0.5),
        },
    }
