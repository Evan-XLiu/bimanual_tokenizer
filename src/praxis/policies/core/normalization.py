# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""PolicyIO-driven normalization-stat selection helpers."""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

from praxis.core.types import FeatureSpec, FeatureType, NormMode, PolicyIO

logger = logging.getLogger(__name__)


def collect_observation_norm_stats(
    *,
    io: PolicyIO | None,
    stats: dict[str, dict[str, Tensor]],
    feature_modes: dict[str, NormMode] | None = None,
    features: list[FeatureSpec] | None = None,
    use_image_defaults: bool = True,
    missing_stats_warning: str = (
        "Missing normalization stats for %s; skipping normalization for this key."
    ),
) -> tuple[dict[str, dict[str, Tensor]], dict[str, NormMode]]:
    if io is None:
        return {}, {}

    target_features = io.obs_features if features is None else features
    relevant_stats: dict[str, dict[str, Tensor]] = {}
    modes: dict[str, NormMode] = {}
    for feat in target_features:
        feature_stats = stats.get(feat.key)
        if feature_stats is None:
            if use_image_defaults and feat.type == FeatureType.IMAGE:
                feature_stats = default_image_stats(feat)
                logger.warning(
                    "Missing normalization stats for %s; using default image min/max in [0,1].",
                    feat.key,
                )
            else:
                logger.warning(missing_stats_warning, feat.key)
                continue

        tensor_stats = coerce_tensor_stats(feature_stats)
        mode = select_norm_mode(
            key=feat.key,
            stats=tensor_stats,
            preferred_mode=(
                feature_modes.get(feat.key, feat.norm_mode)
                if feature_modes is not None
                else feat.norm_mode
            ),
        )
        if mode is None:
            continue

        modes[feat.key] = mode
        relevant_stats[feat.key] = tensor_stats

    return relevant_stats, modes


def collect_action_norm_stats(
    *,
    io: PolicyIO | None,
    stats: dict[str, dict[str, Tensor]],
    feature_modes: dict[str, NormMode] | None = None,
) -> tuple[dict[str, Tensor] | None, NormMode | None]:
    if io is None or io.action_feature is None:
        return None, None

    action_key = io.action_feature.key
    action_stats = stats.get(action_key)
    if action_stats is None:
        logger.warning(
            "Missing usable action stats for %s; action normalization disabled.",
            action_key,
        )
        return None, None

    tensor_stats = coerce_tensor_stats(action_stats)
    mode = select_norm_mode(
        key=action_key,
        stats=tensor_stats,
        preferred_mode=(
            feature_modes.get(action_key, io.action_feature.norm_mode)
            if feature_modes is not None
            else io.action_feature.norm_mode
        ),
        missing_warning=(
            "Missing usable action stats for %s; action normalization disabled."
        ),
    )
    if mode is None:
        return None, None

    return tensor_stats, mode


def select_norm_mode(
    *,
    key: str,
    stats: dict[str, Tensor],
    preferred_mode: NormMode,
    missing_warning: str = (
        "Stats for %s missing both (mean,std) and (min,max); skipping this key."
    ),
) -> NormMode | None:
    if preferred_mode == NormMode.NONE:
        return None

    has_minmax = "min" in stats and "max" in stats
    has_meanstd = "mean" in stats and "std" in stats
    if preferred_mode == NormMode.MIN_MAX and has_minmax:
        return NormMode.MIN_MAX
    if preferred_mode == NormMode.MEAN_STD and has_meanstd:
        return NormMode.MEAN_STD
    if has_minmax:
        return NormMode.MIN_MAX
    if has_meanstd:
        return NormMode.MEAN_STD

    logger.warning(missing_warning, key)
    return None


def coerce_tensor_stats(stats: dict[str, Any]) -> dict[str, Tensor]:
    tensor_stats: dict[str, Tensor] = {}
    for stat_name, value in stats.items():
        if isinstance(value, Tensor):
            tensor_stats[stat_name] = value.float()
        else:
            tensor_stats[stat_name] = torch.as_tensor(value, dtype=torch.float32)
    return tensor_stats


def default_image_stats(feat: FeatureSpec) -> dict[str, Tensor]:
    shape = feat.shape
    stat_shape = (
        (shape[0],) + (1,) * max(0, len(shape) - 1) if len(shape) >= 1 else (1,)
    )
    return {
        "min": torch.zeros(stat_shape, dtype=torch.float32),
        "max": torch.ones(stat_shape, dtype=torch.float32),
    }
