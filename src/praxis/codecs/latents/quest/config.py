"""Configuration dataclass for QueST continuous latent autoencoders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from praxis.codecs.core import build_dataclass_config, dataclass_config_to_dict
from praxis.codecs.latents.core import (
    LatentSpaceConfig,
    build_latent_autoencoder_config,
)


@dataclass
class QueSTModelConfig:
    """Shared QueST encoder/decoder geometry."""

    name: str = "quest"
    pretrained_path: str | None = None

    # Action/sample geometry
    action_dim: int | None = None
    action_horizon: int = 32

    # Model dimensions
    encoder_dim: int = 256
    encoder_nhead: int = 8
    encoder_nlayer: int = 6
    decoder_dim: int = 256
    decoder_nhead: int = 8
    decoder_nlayer: int = 6
    dropout: float = 0.1

    # Temporal modeling
    downsample_factor: int = 2
    use_causal_encoder: bool = True
    use_causal_decoder: bool = True

    latent_dim: int = 4

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> QueSTModelConfig:
        return build_dataclass_config(cls, cfg)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_config_to_dict(self)


@dataclass
class QueSTAutoencoderConfig(QueSTModelConfig):
    """QueST model geometry plus standalone latent-space training controls."""

    latent_space: LatentSpaceConfig = field(default_factory=LatentSpaceConfig)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> QueSTAutoencoderConfig:
        return build_latent_autoencoder_config(cls, cfg)
