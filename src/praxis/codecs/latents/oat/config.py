# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration dataclass for OAT continuous latent autoencoders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from praxis.codecs.core import build_dataclass_config, dataclass_config_to_dict
from praxis.codecs.latents.core import (
    LatentSpaceConfig,
    build_latent_autoencoder_config,
)
from praxis.codecs.latents.oat.schedule import unit_register_schedule


@dataclass
class OATModelConfig:
    """Shared OAT encoder/decoder geometry."""

    name: str = "oat"
    pretrained_path: str | None = None

    # Action/sample geometry
    action_dim: int | None = None
    action_horizon: int = 32
    latent_horizon: int = 16
    latent_dim: int = 4

    # Shared transformer dimensions
    emb_dim: int = 256
    head_dim: int = 32
    encoder_depth: int = 6
    decoder_depth: int = 6
    dropout: float = 0.1

    # Decoder prefix-dropout schedule over ordered latents.
    latent_dropout_mode: str = "pow2"

    @property
    def register_schedule(self) -> tuple[int, ...]:
        return unit_register_schedule(self.latent_horizon)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> OATModelConfig:
        payload = dict(cfg)
        legacy_dropout_mode = payload.pop("token_dropout_mode", None)
        if legacy_dropout_mode is not None:
            current_dropout_mode = payload.get("latent_dropout_mode")
            if (
                current_dropout_mode is not None
                and current_dropout_mode != legacy_dropout_mode
            ):
                raise ValueError(
                    "OAT config cannot specify conflicting token_dropout_mode and "
                    "latent_dropout_mode values."
                )
            payload["latent_dropout_mode"] = legacy_dropout_mode
        return build_dataclass_config(cls, payload)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_config_to_dict(self)


@dataclass
class OATAutoencoderConfig(OATModelConfig):
    """OAT model geometry plus standalone latent-space training controls."""

    latent_space: LatentSpaceConfig = field(default_factory=LatentSpaceConfig)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> OATAutoencoderConfig:
        return build_latent_autoencoder_config(cls, cfg)
