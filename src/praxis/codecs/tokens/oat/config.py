# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration dataclass for OAT tokenizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from praxis.codecs.latents.oat.config import OATModelConfig


@dataclass
class OATTokenizerConfig(OATModelConfig):
    """OAT latent autoencoder config plus finite-scalar quantizer settings."""

    name: str = "oat"
    latent_dim: int = 0

    # Quantizer
    quantizer_levels: tuple[int, ...] = (8, 8, 6, 5)
    quantizer_drop_quant_p: float = 0.0
    quantizer_corrupt_tokens_p: float = 0.0
    quantizer_min_corrupt_tokens_p: float | None = None
    quantizer_apply_corrupt_tokens_p: float = 0.2

    def __post_init__(self) -> None:
        if self.latent_dim == 0:
            self.latent_dim = len(self.quantizer_levels)

    @property
    def block_schedule(self) -> tuple[int, ...]:
        return self.register_schedule

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> OATTokenizerConfig:
        payload = dict(cfg)
        if "quantizer_levels" in payload and isinstance(
            payload["quantizer_levels"], list
        ):
            payload["quantizer_levels"] = tuple(
                int(v) for v in payload["quantizer_levels"]
            )
        return cast(OATTokenizerConfig, super().from_dict(payload))
