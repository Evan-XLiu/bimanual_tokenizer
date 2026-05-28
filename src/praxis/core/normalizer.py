"""Normalization modules backed by LeRobot normalization processors."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from praxis.core.types import CANONICAL_ACTION_KEY, NormMode

try:
    from lerobot.configs.types import (
        FeatureType as LeRobotFeatureType,
        NormalizationMode as LeRobotNormalizationMode,
        PolicyFeature as LeRobotPolicyFeature,
    )
    from lerobot.processor.normalize_processor import (
        NormalizerProcessorStep as LeRobotNormalizerStep,
        UnnormalizerProcessorStep as LeRobotUnnormalizerStep,
    )
except (
    Exception
):  # pragma: no cover - fallback for environments without lerobot processor package
    LeRobotFeatureType = None
    LeRobotNormalizationMode = None
    LeRobotPolicyFeature = None
    LeRobotNormalizerStep = None
    LeRobotUnnormalizerStep = None


def _infer_feature_type(key: str):
    if LeRobotFeatureType is None:
        return None
    key_lower = key.lower()
    if (
        key_lower == CANONICAL_ACTION_KEY
        or key_lower.startswith(f"{CANONICAL_ACTION_KEY}.")
        or CANONICAL_ACTION_KEY in key_lower
    ):
        return LeRobotFeatureType.ACTION
    if "image" in key:
        return LeRobotFeatureType.VISUAL
    return LeRobotFeatureType.STATE


def _infer_feature_shape(stat_dict: dict[str, Any]) -> tuple[int, ...]:
    for candidate in ("min", "max", "mean", "std", "q01", "q99", "q10", "q90"):
        if candidate in stat_dict:
            shape = tuple(torch.as_tensor(stat_dict[candidate]).shape)
            if len(shape) == 0:
                return (1,)
            return shape
    return (1,)


def _parse_to_device_dtype(
    *args, **kwargs
) -> tuple[torch.device | str | None, torch.dtype | None]:
    device = kwargs.get("device")
    dtype = kwargs.get("dtype")
    for arg in args:
        if isinstance(arg, (str, torch.device)):
            device = arg
        elif isinstance(arg, torch.dtype):
            dtype = arg
    return device, dtype


class _LeRobotNormalizerBase(nn.Module):
    """Runtime normalization module. No registered buffers — persistence is handled
    externally via ``norm_stats.pt``."""

    def __init__(
        self,
        stats: dict[str, dict[str, torch.Tensor]],
        modes: dict[str, NormMode],
        *,
        inverse: bool,
    ) -> None:
        super().__init__()
        self.keys = sorted(modes.keys())
        self.modes = modes
        self.stats = stats
        self.inverse = inverse
        self._steps: dict[str, tuple[Any, Any]] = {}
        self._runtime_device: torch.device | None = None

        self._build_steps(stats, modes)

    def _build_steps(
        self,
        stats: dict[str, dict[str, torch.Tensor]],
        modes: dict[str, NormMode],
    ) -> None:
        """Build LeRobot processor steps from stats/modes dicts."""
        if (
            LeRobotNormalizerStep is None
            or LeRobotUnnormalizerStep is None
            or LeRobotFeatureType is None
            or LeRobotNormalizationMode is None
            or LeRobotPolicyFeature is None
        ):
            raise RuntimeError("LeRobot normalization components are not available.")

        self._steps = {}
        step_cls = LeRobotUnnormalizerStep if self.inverse else LeRobotNormalizerStep
        for key in sorted(modes.keys()):
            mode = modes[key]
            if mode == NormMode.NONE:
                continue
            feature_type = _infer_feature_type(key)
            if feature_type is None:
                continue

            if mode == NormMode.MIN_MAX:
                norm_mode = LeRobotNormalizationMode.MIN_MAX
            elif mode == NormMode.MEAN_STD:
                norm_mode = LeRobotNormalizationMode.MEAN_STD
            else:
                norm_mode = LeRobotNormalizationMode.IDENTITY

            key_stats = stats.get(key, {})
            feature = LeRobotPolicyFeature(
                type=feature_type, shape=_infer_feature_shape(key_stats)
            )
            step = step_cls(
                features={key: feature},
                norm_map={feature_type: norm_mode},
                stats={key: key_stats},
            )
            self._steps[key] = (step, feature_type)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = dict(batch)
        target_device = next(
            (val.device for val in out.values() if isinstance(val, torch.Tensor)),
            None,
        )
        if target_device is not None and self._runtime_device != target_device:
            self.to(device=target_device)
        for key in self.keys:
            if key not in out:
                continue
            if out[key] is None or not isinstance(out[key], torch.Tensor):
                continue
            step_and_type = self._steps.get(key)
            if step_and_type is None:
                continue
            step, feature_type = step_and_type
            out[key] = step._apply_transform(  # noqa: SLF001
                out[key], key, feature_type, inverse=self.inverse
            )
        return out

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        device, dtype = _parse_to_device_dtype(*args, **kwargs)
        if isinstance(device, str):
            device = torch.device(device)
        for step, _feature_type in self._steps.values():
            step.to(device=device, dtype=dtype)
        if isinstance(device, torch.device):
            self._runtime_device = device
        return module


class Normalize(_LeRobotNormalizerBase):
    """Normalizes tensors using pre-computed statistics."""

    def __init__(
        self, stats: dict[str, dict[str, torch.Tensor]], modes: dict[str, NormMode]
    ) -> None:
        super().__init__(stats=stats, modes=modes, inverse=False)


class Unnormalize(_LeRobotNormalizerBase):
    """Inverse of Normalize — maps normalized tensors back to original scale."""

    def __init__(
        self, stats: dict[str, dict[str, torch.Tensor]], modes: dict[str, NormMode]
    ) -> None:
        super().__init__(stats=stats, modes=modes, inverse=True)
