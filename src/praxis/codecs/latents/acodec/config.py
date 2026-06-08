# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration dataclass for ActionCodec continuous latent autoencoders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from praxis.codecs.core import build_dataclass_config, dataclass_config_to_dict
from praxis.codecs.latents.core import (
    LatentSpaceConfig,
    build_latent_autoencoder_config,
)


@dataclass
class ActionCodecModelConfig:
    """Shared ActionCodec-style encoder/decoder geometry."""

    name: str = "acodec"
    pretrained_path: str | None = None

    # Action/sample geometry
    action_dim: int | None = None
    action_horizon: int = 32
    latent_horizon: int = 16

    # Model dimensions
    emb_dim: int = 256
    latent_dim: int = 512
    head_dim: int = 32
    encoder_depth: int = 6
    decoder_depth: int = 6
    dropout: float = 0.1
    mlp_ratio: float = 4.0
    qk_norm: str = "ln"
    layer_scale: bool = True
    pos_scale: float = 0.1
    encoder_add_self_attn: bool = False
    encoder_add_causal_mask: bool = False
    decoder_add_self_attn: bool = False
    decoder_add_causal_mask: bool = False

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> ActionCodecModelConfig:
        return build_dataclass_config(cls, cfg)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_config_to_dict(self)


@dataclass
class ActionCodecAutoencoderConfig(ActionCodecModelConfig):
    """ActionCodec model geometry plus standalone latent-space training controls."""

    latent_space: LatentSpaceConfig = field(default_factory=LatentSpaceConfig)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> ActionCodecAutoencoderConfig:
        return build_latent_autoencoder_config(cls, cfg)
