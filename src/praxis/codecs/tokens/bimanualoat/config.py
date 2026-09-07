# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the paired bimanual OAT tokenizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from praxis.codecs.latents.oat.config import OATModelConfig

DEFAULT_RIGHT_ARM_SIGN = (-1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0)


@dataclass
class BimanualOATTokenizerConfig(OATModelConfig):
    """OAT geometry with independent common and relative FSQ streams."""

    name: str = "bimanualoat"
    latent_dim: int = 0

    right_arm_sign: tuple[float, ...] = DEFAULT_RIGHT_ARM_SIGN
    common_quantizer_levels: tuple[int, ...] = (8, 6)
    relative_quantizer_levels: tuple[int, ...] = (8, 6)
    quantizer_drop_quant_p: float = 0.0
    quantizer_corrupt_tokens_p: float = 0.0
    quantizer_min_corrupt_tokens_p: float | None = None
    quantizer_apply_corrupt_tokens_p: float = 0.2

    def __post_init__(self) -> None:
        self.right_arm_sign = tuple(float(value) for value in self.right_arm_sign)
        self.common_quantizer_levels = tuple(
            int(value) for value in self.common_quantizer_levels
        )
        self.relative_quantizer_levels = tuple(
            int(value) for value in self.relative_quantizer_levels
        )
        if not self.common_quantizer_levels or not self.relative_quantizer_levels:
            raise ValueError("common and relative quantizer levels must be non-empty.")
        if any(value not in (-1.0, 1.0) for value in self.right_arm_sign):
            raise ValueError("right_arm_sign values must be either -1 or 1.")
        expected_latent_dim = len(self.common_quantizer_levels) + len(
            self.relative_quantizer_levels
        )
        if self.latent_dim == 0:
            self.latent_dim = expected_latent_dim
        elif self.latent_dim != expected_latent_dim:
            raise ValueError(
                "latent_dim must equal the combined common and relative FSQ "
                f"dimensions ({expected_latent_dim}), got {self.latent_dim}."
            )
        if self.action_dim is not None:
            action_dim = int(self.action_dim)
            if action_dim % 2 != 0:
                raise ValueError(
                    f"bimanualoat requires an even action_dim, got {action_dim}."
                )
            if len(self.right_arm_sign) != action_dim // 2:
                raise ValueError(
                    "right_arm_sign length must equal action_dim / 2; "
                    f"got {len(self.right_arm_sign)} and action_dim={action_dim}."
                )

    @property
    def block_schedule(self) -> tuple[int, ...]:
        return self.register_schedule

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> BimanualOATTokenizerConfig:
        payload = dict(cfg)
        for key in (
            "right_arm_sign",
            "common_quantizer_levels",
            "relative_quantizer_levels",
        ):
            if key in payload and isinstance(payload[key], list):
                payload[key] = tuple(payload[key])
        return cast(BimanualOATTokenizerConfig, super().from_dict(payload))
