# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""PolicyIO construction from dataset metadata."""

from __future__ import annotations

import logging

from omegaconf import DictConfig

from praxis.core.types import (
    CANONICAL_ACTION_KEY,
    FeatureSpec,
    FeatureType,
    NormMode,
    PolicyIO,
)
from praxis.data.utils import resolve_feature_norm_mode

try:
    from lerobot.configs.types import FeatureType as LeRobotFeatureType
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.utils.utils import (
        get_channel_first_image_shape as lerobot_get_channel_first_image_shape,
    )
except Exception:  # pragma: no cover - fallback for environments without lerobot
    LeRobotFeatureType = None
    dataset_to_policy_features = None
    lerobot_get_channel_first_image_shape = None

logger = logging.getLogger(__name__)

OBSERVATION_PREFIX = "observation."
RESERVED_NON_ACTION_KEYS = {
    "index",
    "frame_index",
    "episode_index",
    "timestamp",
    "task",
    "task_index",
    "reward",
    "done",
    "next.reward",
    "next.done",
}


def make_io_from_metadata(
    meta_features: dict,
    policy_cfg: DictConfig,
    *,
    observation_keys: list[str] | None = None,
) -> PolicyIO:
    """Build PolicyIO from LeRobot dataset metadata features."""
    obs_features: list[FeatureSpec] = []
    action_feature: FeatureSpec | None = None
    policy_features = _extract_policy_features(meta_features)

    if policy_features is not None and LeRobotFeatureType is not None:
        obs_from_policy, action_candidates = (
            _collect_obs_and_action_from_policy_features(meta_features, policy_features)
        )
        obs_features.extend(obs_from_policy)
        action_feature = _select_policy_action_feature(action_candidates)
    else:
        obs_features.extend(_collect_obs_from_metadata_fallback(meta_features))

    if action_feature is None:
        fallback_action_candidates = _collect_fallback_action_candidates(meta_features)
        action_feature = _infer_action_feature_from_fallback(fallback_action_candidates)

    return PolicyIO(
        obs_features=_filter_observation_features(
            obs_features, observation_keys=observation_keys
        ),
        action_feature=action_feature,
        n_obs_steps=policy_cfg.n_obs_steps,
        n_action_steps=policy_cfg.action_horizon,
    )


def _normalize_image_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return image shape in channel-first format when metadata looks channel-last."""
    if len(shape) != 3:
        return shape

    if lerobot_get_channel_first_image_shape is not None:
        try:
            return tuple(lerobot_get_channel_first_image_shape(shape))
        except Exception:
            pass

    h_or_c, _, c_or_w = shape
    if c_or_w in (1, 3, 4) and h_or_c not in (1, 3, 4):
        return (c_or_w, shape[0], shape[1])
    return shape


def _extract_policy_features(meta_features: dict) -> dict | None:
    """Try LeRobot feature conversion and return None when conversion is unavailable."""
    if dataset_to_policy_features is None or LeRobotFeatureType is None:
        return None
    try:
        converted = dataset_to_policy_features(meta_features)
    except Exception as exc:
        logger.warning(
            "Falling back to local metadata parsing because LeRobot feature conversion failed: %s",
            exc,
        )
        return None
    return converted


def _filter_observation_features(
    obs_features: list[FeatureSpec],
    *,
    observation_keys: list[str] | None,
) -> list[FeatureSpec]:
    if observation_keys is None:
        return obs_features
    if not observation_keys:
        raise ValueError("observation_keys must be null or a non-empty list.")

    by_key = {feature.key: feature for feature in obs_features}
    missing = [key for key in observation_keys if key not in by_key]
    if missing:
        raise KeyError(
            "Requested observation_keys are missing from dataset metadata: "
            f"{missing}. Available: {sorted(by_key)}"
        )

    filtered: list[FeatureSpec] = []
    seen: set[str] = set()
    for key in observation_keys:
        if key in seen:
            raise ValueError(f"Duplicate observation key requested: {key}")
        filtered.append(by_key[key])
        seen.add(key)
    return filtered


def _collect_obs_and_action_from_policy_features(
    meta_features: dict, policy_features: dict
) -> tuple[list[FeatureSpec], list[FeatureSpec]]:
    """Collect observation specs and action candidates from converted LeRobot features."""
    if LeRobotFeatureType is None:
        raise RuntimeError(
            "LeRobot feature conversion is unavailable; cannot parse policy features."
        )
    lerobot_feature_type = LeRobotFeatureType

    obs_features: list[FeatureSpec] = []
    action_candidates: list[FeatureSpec] = []

    for key, feat in policy_features.items():
        shape = tuple(feat.shape)
        if feat.type == lerobot_feature_type.VISUAL:
            obs_features.append(
                FeatureSpec(
                    key=key,
                    type=FeatureType.IMAGE,
                    shape=_normalize_image_shape(shape),
                    norm_mode=resolve_feature_norm_mode(
                        meta_features.get(key), NormMode.MEAN_STD
                    ),
                )
            )
            continue

        if feat.type in (lerobot_feature_type.STATE, lerobot_feature_type.ENV):
            if key.startswith(OBSERVATION_PREFIX):
                obs_features.append(
                    FeatureSpec(
                        key=key,
                        type=FeatureType.STATE,
                        shape=shape,
                        norm_mode=resolve_feature_norm_mode(
                            meta_features.get(key), NormMode.MIN_MAX
                        ),
                    )
                )
            continue

        if feat.type == lerobot_feature_type.ACTION:
            action_candidates.append(
                FeatureSpec(
                    key=key,
                    type=FeatureType.ACTION,
                    shape=shape,
                    norm_mode=resolve_feature_norm_mode(
                        meta_features.get(key), NormMode.MIN_MAX
                    ),
                )
            )

    return obs_features, action_candidates


def _collect_obs_from_metadata_fallback(meta_features: dict) -> list[FeatureSpec]:
    """Build observation specs directly from metadata when LeRobot conversion is unavailable."""
    obs_features: list[FeatureSpec] = []
    for key, info in meta_features.items():
        if not key.startswith(OBSERVATION_PREFIX):
            continue
        ftype = (
            FeatureType.IMAGE
            if info["dtype"] in ("image", "video")
            else FeatureType.STATE
        )
        shape = tuple(info["shape"])
        if ftype == FeatureType.IMAGE and len(shape) == 3:
            shape = _normalize_image_shape(shape)
        default_mode = (
            NormMode.MEAN_STD if ftype == FeatureType.IMAGE else NormMode.MIN_MAX
        )
        obs_features.append(
            FeatureSpec(
                key=key,
                type=ftype,
                shape=shape,
                norm_mode=resolve_feature_norm_mode(info, default_mode),
            )
        )
    return obs_features


def _collect_fallback_action_candidates(meta_features: dict) -> list[FeatureSpec]:
    """Infer 1D action-like features from raw metadata keys."""
    candidates: list[FeatureSpec] = []
    for key, info in meta_features.items():
        if key.startswith(OBSERVATION_PREFIX):
            continue
        if (
            key in RESERVED_NON_ACTION_KEYS
            or key.startswith("next.")
            or key.endswith("_is_pad")
        ):
            continue
        shape = (
            info.get("shape")
            if isinstance(info, dict)
            else getattr(info, "shape", None)
        )
        if shape is None:
            continue
        shape_tuple = tuple(shape)
        if len(shape_tuple) != 1:
            continue
        candidates.append(
            FeatureSpec(
                key=key,
                type=FeatureType.ACTION,
                shape=shape_tuple,
                norm_mode=resolve_feature_norm_mode(info, NormMode.MIN_MAX),
            )
        )
    return candidates


def _select_policy_action_feature(
    action_candidates: list[FeatureSpec],
) -> FeatureSpec | None:
    """Select action feature when LeRobot-converted action candidates are available."""
    if not action_candidates:
        return None
    if len(action_candidates) == 1:
        return action_candidates[0]
    canonical = [feat for feat in action_candidates if feat.key == CANONICAL_ACTION_KEY]
    if len(canonical) == 1:
        return canonical[0]
    raise ValueError(
        "Ambiguous action features in metadata: "
        f"{[feat.key for feat in action_candidates]}. "
        "Please keep a single action feature."
    )


def _infer_action_feature_from_fallback(
    fallback_action_candidates: list[FeatureSpec],
) -> FeatureSpec:
    """Infer action feature from fallback metadata candidates."""
    if not fallback_action_candidates:
        raise KeyError(
            "Unable to infer action feature from metadata. "
            "Expected at least one non-observation action-like key."
        )
    if len(fallback_action_candidates) == 1:
        return fallback_action_candidates[0]
    canonical = [
        feat for feat in fallback_action_candidates if feat.key == CANONICAL_ACTION_KEY
    ]
    if len(canonical) == 1:
        return canonical[0]
    action_named = [
        feat for feat in fallback_action_candidates if "action" in feat.key.lower()
    ]
    if len(action_named) == 1:
        return action_named[0]
    raise ValueError(
        "Ambiguous fallback action feature inference from metadata: "
        f"{[feat.key for feat in fallback_action_candidates]}"
    )
