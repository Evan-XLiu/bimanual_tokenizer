# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration dataclass for BOATTokenizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from praxis.codecs.latents.oat.schedule import is_power_of_two, pow2_register_schedule
from praxis.codecs.tokens.oat.config import OATTokenizerConfig


@dataclass
class BOATTokenizerConfig(OATTokenizerConfig):
    """BOAT specializes OAT with a coarse-to-fine register schedule."""

    name: str = "boat"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.latent_horizon = int(self.latent_horizon)
        if not is_power_of_two(self.latent_horizon):
            raise ValueError(
                "BOATTokenizer requires latent_horizon to be a positive power of 2, "
                f"got {self.latent_horizon}."
            )

    @property
    def register_schedule(self) -> tuple[int, ...]:
        return pow2_register_schedule(self.latent_horizon)

    @property
    def block_schedule(self) -> tuple[int, ...]:
        return self.register_schedule

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> BOATTokenizerConfig:
        return cast(BOATTokenizerConfig, super().from_dict(cfg))
