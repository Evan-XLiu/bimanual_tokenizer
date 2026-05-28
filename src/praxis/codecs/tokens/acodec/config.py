"""Configuration dataclass for ActionCodec tokenizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from praxis.codecs.latents.acodec.config import ActionCodecModelConfig


@dataclass
class ActionCodecTokenizerConfig(ActionCodecModelConfig):
    """ActionCodec latent autoencoder config plus VQ settings."""

    name: str = "acodec"

    # Vector quantizer
    vq_codebook_size: int = 2048
    vq_codebook_dim: int | None = None
    vq_commitment_weight: float = 0.25
    vq_decay: float = 0.99
    vq_epsilon: float = 1e-5
    vq_threshold_ema_dead_code: int = 2
    vq_kmeans_init: bool = True
    vq_kmeans_iters: int = 10

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> ActionCodecTokenizerConfig:
        return cast(ActionCodecTokenizerConfig, super().from_dict(cfg))
