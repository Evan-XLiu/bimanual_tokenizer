"""Data utilities: delta_timestamps builder, stats conversion, collation."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch

try:
    from lerobot.processor.converters import to_tensor as lerobot_to_tensor
except (
    Exception
):  # pragma: no cover - fallback for environments without lerobot processor package
    lerobot_to_tensor = None

from praxis.core.types import (
    CANONICAL_ACTION_KEY,
    FeatureSpec,
    FeatureType,
    NormMode,
    PolicyIO,
)

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


def stats_to_tensors(
    stats: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, torch.Tensor]]:
    """Convert numpy stats from LeRobotDataset.meta.stats to torch tensors."""
    if lerobot_to_tensor is not None:
        converted = lerobot_to_tensor(stats, dtype=torch.float32)
        if isinstance(converted, dict):
            return cast(dict[str, dict[str, torch.Tensor]], converted)

    result = {}
    for key, stat_dict in stats.items():
        result[key] = {}
        for stat_name, value in stat_dict.items():
            if isinstance(value, np.ndarray):
                result[key][stat_name] = torch.from_numpy(value).float()
            elif isinstance(value, torch.Tensor):
                result[key][stat_name] = value.float()
            else:
                result[key][stat_name] = torch.tensor(value, dtype=torch.float32)
    return result


def parse_norm_mode(value: Any) -> NormMode | None:
    """Parse a normalization mode value from metadata/config into ``NormMode``."""
    if value is None:
        return None
    if isinstance(value, NormMode):
        return value

    raw = value.value if hasattr(value, "value") else value
    if not isinstance(raw, str):
        return None

    normalized = raw.strip().lower().replace("-", "_")
    aliases = {
        "min_max": NormMode.MIN_MAX,
        "minmax": NormMode.MIN_MAX,
        "mean_std": NormMode.MEAN_STD,
        "meanstd": NormMode.MEAN_STD,
        "none": NormMode.NONE,
        "identity": NormMode.NONE,
    }
    return aliases.get(normalized)


def resolve_feature_norm_mode(feature_meta: Any, default: NormMode) -> NormMode:
    """Resolve per-feature normalization mode from metadata with fallback."""
    if not isinstance(feature_meta, dict):
        return default

    info = feature_meta.get("info")
    candidates = [
        feature_meta.get("norm_mode"),
        feature_meta.get("normalization_mode"),
        feature_meta.get("norm"),
        info.get("norm_mode") if isinstance(info, dict) else None,
        info.get("normalization_mode") if isinstance(info, dict) else None,
    ]
    for candidate in candidates:
        parsed = parse_norm_mode(candidate)
        if parsed is not None:
            return parsed
    return default


def resolve_normalization_modes_for_io(
    meta_features: dict[str, Any] | None,
    io: PolicyIO | None,
) -> dict[str, NormMode]:
    """Resolve per-feature normalization modes for the active PolicyIO keys."""
    if io is None:
        return {}
    meta_features = meta_features or {}
    features = list(io.obs_features)
    if io.action_feature is not None:
        features.append(io.action_feature)
    return {
        feat.key: resolve_feature_norm_mode(meta_features.get(feat.key), feat.norm_mode)
        for feat in features
    }


def _metadata_shape(feature_meta: Any) -> tuple[int, ...] | None:
    shape = (
        feature_meta.get("shape")
        if isinstance(feature_meta, dict)
        else getattr(feature_meta, "shape", None)
    )
    if shape is None:
        return None
    return tuple(int(dim) for dim in shape)


def collect_action_feature_candidates(
    meta_features: dict[str, Any],
) -> list[FeatureSpec]:
    """Collect one-dimensional non-observation features that can be actions."""
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
        shape = _metadata_shape(info)
        if shape is None or len(shape) != 1:
            continue
        candidates.append(
            FeatureSpec(
                key=key,
                type=FeatureType.ACTION,
                shape=shape,
                norm_mode=resolve_feature_norm_mode(info, NormMode.MIN_MAX),
            )
        )
    return candidates


def resolve_action_feature_from_metadata(
    meta_features: dict[str, Any],
    *,
    context: str = "dataset metadata",
) -> FeatureSpec:
    """Resolve the dataset action feature from LeRobot-style metadata."""
    candidates = collect_action_feature_candidates(meta_features)
    if not candidates:
        raise KeyError(
            f"Unable to infer action feature from {context}. Expected at least "
            "one one-dimensional non-observation feature."
        )
    if len(candidates) == 1:
        return candidates[0]

    canonical = [feat for feat in candidates if feat.key == CANONICAL_ACTION_KEY]
    if len(canonical) == 1:
        return canonical[0]

    action_named = [feat for feat in candidates if "action" in feat.key.lower()]
    if len(action_named) == 1:
        return action_named[0]

    raise ValueError(
        f"Ambiguous action feature inference from {context}: "
        f"candidates={[feat.key for feat in candidates]}. Expected one feature, "
        f"canonical {CANONICAL_ACTION_KEY!r}, or a single key containing 'action'."
    )
