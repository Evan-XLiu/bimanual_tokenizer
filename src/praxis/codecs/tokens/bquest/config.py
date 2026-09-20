# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the bimanual QueST tokenizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from praxis.codecs.latents.quest.config import QueSTModelConfig

DEFAULT_RIGHT_ARM_SIGN = (-1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0)


@dataclass
class BQuestTokenizerConfig(QueSTModelConfig):
    """Shared-arm QueST geometry and two-level residual FSQ settings."""

    name: str = "bquest"
    action_dim: int | None = 14
    action_horizon: int = 32
    downsample_factor: int = 4
    latent_dim: int = 4

    # The transform is optional so canonicalization can be ablated cleanly.
    canonicalize_right_arm: bool = True
    right_arm_sign: tuple[float, ...] = DEFAULT_RIGHT_ARM_SIGN

    # Both streams use the same scalar grid and vocabulary.
    fsq_levels: tuple[int, ...] = (8, 8, 6, 5)

    def __post_init__(self) -> None:
        if self.action_horizon != 32:
            raise ValueError(
                f"BQuestTokenizer requires action_horizon=32, got {self.action_horizon}."
            )
        if self.downsample_factor != 4:
            raise ValueError(
                "BQuestTokenizer requires downsample_factor=4, "
                f"got {self.downsample_factor}."
            )
        self.right_arm_sign = tuple(float(value) for value in self.right_arm_sign)
        self.fsq_levels = tuple(int(value) for value in self.fsq_levels)
        if not self.fsq_levels or any(value < 2 for value in self.fsq_levels):
            raise ValueError("fsq_levels must contain integers >= 2.")
        if self.latent_dim != len(self.fsq_levels):
            raise ValueError(
                "latent_dim must match fsq_levels length; "
                f"got latent_dim={self.latent_dim}, levels={len(self.fsq_levels)}."
            )
        if any(value not in (-1.0, 1.0) for value in self.right_arm_sign):
            raise ValueError("right_arm_sign values must be either -1 or 1.")
        if len(self.right_arm_sign) != 7:
            raise ValueError(
                "right_arm_sign must contain exactly 7 values for the right arm, "
                f"got {len(self.right_arm_sign)}."
            )
        if self.action_dim is not None:
            action_dim = int(self.action_dim)
            if action_dim != 14:
                raise ValueError(
                    f"BQuestTokenizer expects action_dim=14, got {action_dim}."
                )
            if len(self.right_arm_sign) != action_dim // 2:
                raise ValueError(
                    "right_arm_sign length must equal action_dim / 2; "
                    f"got {len(self.right_arm_sign)} and action_dim={action_dim}."
                )

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> BQuestTokenizerConfig:
        payload = dict(cfg)
        for key in ("fsq_levels", "right_arm_sign"):
            if key in payload and isinstance(payload[key], list):
                payload[key] = tuple(payload[key])
        return cast(BQuestTokenizerConfig, super().from_dict(payload))
