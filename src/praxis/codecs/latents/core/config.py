# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Shared config objects for continuous latent codecs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeVar

from praxis.codecs.core import build_dataclass_config


@dataclass
class LatentRegularizationConfig:
    """Aggregate-prior regularization for standalone latent-codec training."""

    enabled: bool = True
    mean_weight: float = 1e-3
    variance_weight: float = 1e-3
    covariance_weight: float = 0.0
    norm_weight: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "mean_weight",
            "variance_weight",
            "covariance_weight",
            "norm_weight",
        ):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}.")
            setattr(self, name, value)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> LatentRegularizationConfig:
        return build_dataclass_config(cls, cfg)


@dataclass
class LatentSpaceConfig:
    """Standalone latent-space training controls."""

    regularization: LatentRegularizationConfig = field(
        default_factory=LatentRegularizationConfig
    )
    decoder_noise_std: float = 0.0

    def __post_init__(self) -> None:
        regularization: Any = self.regularization
        if isinstance(regularization, dict):
            self.regularization = LatentRegularizationConfig.from_dict(regularization)
        elif not isinstance(regularization, LatentRegularizationConfig):
            raise TypeError(
                "regularization must be a dict or LatentRegularizationConfig."
            )
        self.decoder_noise_std = _validate_decoder_noise_std(self.decoder_noise_std)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> LatentSpaceConfig:
        payload = dict(cfg)
        regularization = payload.get("regularization")
        if isinstance(regularization, dict):
            payload["regularization"] = LatentRegularizationConfig.from_dict(
                regularization
            )
        elif regularization is not None and not isinstance(
            regularization, LatentRegularizationConfig
        ):
            raise TypeError(
                "latent_space.regularization must be a dict, "
                "LatentRegularizationConfig, or None."
            )
        return build_dataclass_config(cls, payload)


def _validate_decoder_noise_std(value: float) -> float:
    """Validate train-time latent decoder noise standard deviation."""
    resolved = float(value)
    if resolved < 0.0:
        raise ValueError(f"decoder_noise_std must be >= 0, got {resolved}.")
    return resolved


T = TypeVar("T")


def build_latent_autoencoder_config(config_cls: type[T], cfg: dict[str, Any]) -> T:
    """Build a latent-autoencoder config with nested latent-space config."""
    payload = dict(cfg)
    value = payload.get("latent_space")
    if isinstance(value, dict):
        payload["latent_space"] = LatentSpaceConfig.from_dict(value)
    elif value is not None and not isinstance(value, LatentSpaceConfig):
        raise TypeError("latent_space must be a dict or LatentSpaceConfig.")
    return build_dataclass_config(config_cls, payload)
