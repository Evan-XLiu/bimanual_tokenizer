# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Core type definitions for the policy research framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FeatureType(str, Enum):
    """Type of a feature in the dataset."""

    STATE = "state"
    IMAGE = "image"
    ACTION = "action"
    LANGUAGE = "language"


CANONICAL_ACTION_KEY = FeatureType.ACTION.value


class NormMode(str, Enum):
    """Normalization mode for features."""

    MIN_MAX = "min_max"  # Normalize to [-1, 1]
    MEAN_STD = "mean_std"  # Zero-mean, unit-variance
    NONE = "none"  # No normalization


@dataclass
class FeatureSpec:
    """Specification of a single feature (observation or action).

    Attributes:
        key: LeRobot-style key (e.g., "observation.state", "observation.images.top").
        type: Whether this is a state, image, or action feature.
        shape: Expected shape per timestep (e.g., (8,) for state, (3, 96, 96) for image).
        norm_mode: How to normalize this feature.
    """

    key: str
    type: FeatureType
    shape: tuple[int, ...]
    norm_mode: NormMode = NormMode.NONE

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "type": self.type.value,
            "shape": list(self.shape),
            "norm_mode": self.norm_mode.value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> FeatureSpec:
        return cls(
            key=d["key"],
            type=FeatureType(d["type"]),
            shape=tuple(d["shape"]),
            norm_mode=NormMode(d["norm_mode"]),
        )


@dataclass
class PolicyIO:
    """Describes the input/output interface of a policy.

    This defines what observations a policy expects and what actions it produces,
    along with temporal horizons for observation stacking and action chunking.

    Attributes:
        obs_features: List of observation feature specs.
        action_feature: The action feature spec.
        n_obs_steps: Number of observation frames to stack (temporal history).
        n_action_steps: Number of action steps to predict (action chunk size).
        fps: Dataset frame rate, used to compute delta_timestamps.
    """

    obs_features: list[FeatureSpec] = field(default_factory=list)
    action_feature: FeatureSpec | None = None
    n_obs_steps: int = 1
    n_action_steps: int = 1
    fps: int = 30
    language_key: str | None = None

    def to_dict(self) -> dict:
        return {
            "obs_features": [f.to_dict() for f in self.obs_features],
            "action_feature": self.action_feature.to_dict()
            if self.action_feature
            else None,
            "n_obs_steps": self.n_obs_steps,
            "n_action_steps": self.n_action_steps,
            "fps": self.fps,
            "language_key": self.language_key,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PolicyIO:
        return cls(
            obs_features=[FeatureSpec.from_dict(f) for f in d["obs_features"]],
            action_feature=FeatureSpec.from_dict(d["action_feature"])
            if d.get("action_feature")
            else None,
            n_obs_steps=d["n_obs_steps"],
            n_action_steps=d["n_action_steps"],
            fps=d["fps"],
            language_key=d.get("language_key"),
        )

    @property
    def image_dims(self) -> tuple[int, int]:
        """Extract (H, W) from the first image feature. Channel-first format assumed."""
        img_features = self.image_features
        if not img_features:
            raise ValueError(
                "PolicyIO has no image features — cannot determine observation dimensions."
            )
        shape = img_features[0].shape
        if len(shape) != 3:
            raise ValueError(
                f"Expected 3D image shape (C, H, W), got {shape} for feature {img_features[0].key}"
            )
        return shape[1], shape[2]

    @property
    def state_features(self) -> list[FeatureSpec]:
        return [f for f in self.obs_features if f.type == FeatureType.STATE]

    @property
    def image_features(self) -> list[FeatureSpec]:
        return [f for f in self.obs_features if f.type == FeatureType.IMAGE]

    @property
    def obs_keys(self) -> list[str]:
        return [f.key for f in self.obs_features]

    @property
    def action_dim(self) -> int:
        if self.action_feature is None:
            raise ValueError("No action feature defined.")
        return self.action_feature.shape[0]

    def build_delta_timestamps(self) -> dict[str, list[float]]:
        """Build delta_timestamps dict for LeRobotDataset.

        Observation keys get n_obs_steps frames ending at t=0.
        Action key gets n_action_steps frames starting at t=0.
        """
        dt = 1.0 / self.fps
        obs_deltas = [round(-i * dt, 6) for i in reversed(range(self.n_obs_steps))]

        timestamps: dict[str, list[float]] = {}
        for feat in self.obs_features:
            timestamps[feat.key] = obs_deltas

        if self.action_feature is not None:
            action_deltas = [round(i * dt, 6) for i in range(self.n_action_steps)]
            timestamps[self.action_feature.key] = action_deltas

        return timestamps
