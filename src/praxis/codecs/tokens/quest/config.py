"""Configuration dataclass for QueST tokenizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from praxis.codecs.latents.quest.config import QueSTModelConfig


@dataclass
class QueSTTokenizerConfig(QueSTModelConfig):
    """QueST latent autoencoder config plus finite-scalar quantizer settings."""

    name: str = "quest"
    latent_dim: int = 0

    # Quantizer
    fsq_levels: tuple[int, ...] = (8, 8, 6, 5)

    def __post_init__(self) -> None:
        if self.latent_dim == 0:
            self.latent_dim = len(self.fsq_levels)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> QueSTTokenizerConfig:
        payload = dict(cfg)
        if "fsq_levels" in payload and isinstance(payload["fsq_levels"], list):
            payload["fsq_levels"] = tuple(int(v) for v in payload["fsq_levels"])
        return cast(QueSTTokenizerConfig, super().from_dict(payload))
