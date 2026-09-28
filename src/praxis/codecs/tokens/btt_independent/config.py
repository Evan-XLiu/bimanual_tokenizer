# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration for BTT with independent arm normalization and parameters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

from praxis.codecs.core import build_dataclass_config, dataclass_config_to_dict
from praxis.codecs.tokens.btt.config import DEFAULT_RIGHT_ARM_SIGN

# RoboTwin robotwin_new_demo_clean/meta/stats.json action extrema.
# Right-arm bounds are transformed into the original BTT convention first.
DEFAULT_LEFT_ACTION_MIN = (
    -1.3947148323059082,
    -0.0003164021181873977,
    -0.1190902441740036,
    -1.9268131256103516,
    -1.4216828346252441,
    -3.468787908554077,
    0.0,
)
DEFAULT_LEFT_ACTION_MAX = (
    0.4364263713359833,
    3.3889312744140625,
    3.4514999389648438,
    1.789766788482666,
    1.5632697343826294,
    3.010831117630005,
    1.0,
)
DEFAULT_RIGHT_ACTION_MIN = (
    -1.3892723321914673,
    -0.5742834806442261,
    -0.005311839282512665,
    -1.9893786907196045,
    -1.397123098373413,
    -3.4247944355010986,
    0.0,
)
DEFAULT_RIGHT_ACTION_MAX = (
    2.9455959796905518,
    2.9950597286224365,
    3.333350419998169,
    1.9530924558639526,
    1.6339466571807861,
    2.9398255348205566,
    1.0,
)


@dataclass
class BTTIndependentConfig:
    """Geometry for the bimanual temporal-tokenizer encoder frontend."""

    name: str = "btt_independent"

    action_dim: int = 14
    arm_dim: int = 7
    action_horizon: int = 32
    latent_horizon_per_arm: int = 8

    model_dim: int = 256
    num_heads: int = 8
    dropout: float = 0.1

    latent_dim: int = 4
    fsq_levels: tuple[int, ...] = (8, 8, 6, 5)

    decoder_nhead: int = 8
    decoder_nlayer: int = 6
    use_causal_decoder: bool = True

    conv_kernel_sizes: tuple[int, ...] = (5, 3)
    conv_strides: tuple[int, ...] = (2, 2)
    right_arm_sign: tuple[float, ...] = DEFAULT_RIGHT_ARM_SIGN
    left_action_min: tuple[float, ...] = DEFAULT_LEFT_ACTION_MIN
    left_action_max: tuple[float, ...] = DEFAULT_LEFT_ACTION_MAX
    right_action_min: tuple[float, ...] = DEFAULT_RIGHT_ACTION_MIN
    right_action_max: tuple[float, ...] = DEFAULT_RIGHT_ACTION_MAX
    clamp_normalized_actions: bool = False

    def __post_init__(self) -> None:
        self.conv_kernel_sizes = tuple(int(value) for value in self.conv_kernel_sizes)
        self.conv_strides = tuple(int(value) for value in self.conv_strides)
        self.fsq_levels = tuple(int(value) for value in self.fsq_levels)
        self.right_arm_sign = tuple(float(value) for value in self.right_arm_sign)
        if self.name != "btt_independent":
            raise ValueError(
                "Expected a btt_independent checkpoint/config, not another BTT variant."
            )
        if self.right_arm_sign != DEFAULT_RIGHT_ARM_SIGN:
            raise ValueError(
                "btt_independent preserves the original BTT right_arm_sign convention."
            )
        for arm in ("left", "right"):
            for suffix in ("min", "max"):
                key = f"{arm}_action_{suffix}"
                setattr(self, key, tuple(float(value) for value in getattr(self, key)))

        if self.action_dim != 2 * self.arm_dim:
            raise ValueError(
                "action_dim must contain exactly two arms; "
                f"got action_dim={self.action_dim}, arm_dim={self.arm_dim}."
            )
        if self.action_horizon != 32 or self.latent_horizon_per_arm != 8:
            raise ValueError(
                "BTT currently requires action_horizon=32 and latent_horizon_per_arm=8."
            )
        if self.conv_strides != (2, 2):
            raise ValueError(
                f"BTT requires two stride-2 temporal stages, got {self.conv_strides}."
            )
        if len(self.conv_kernel_sizes) != len(self.conv_strides):
            raise ValueError(
                "conv_kernel_sizes and conv_strides must have equal length."
            )
        if any(
            kernel < stride
            for kernel, stride in zip(
                self.conv_kernel_sizes, self.conv_strides, strict=True
            )
        ):
            raise ValueError("each causal kernel size must be at least its stride.")
        downsample_factor = 1
        for stride in self.conv_strides:
            downsample_factor *= stride
        if self.action_horizon // downsample_factor != self.latent_horizon_per_arm:
            raise ValueError(
                "convolution schedule does not produce the latent horizon."
            )
        if self.model_dim <= 0 or self.model_dim % 2 != 0:
            raise ValueError(
                "model_dim must be positive and even for sin/cos positions."
            )
        if self.num_heads <= 0 or self.model_dim % self.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads.")
        if self.decoder_nhead <= 0 or self.model_dim % self.decoder_nhead != 0:
            raise ValueError("model_dim must be divisible by decoder_nhead.")
        if self.decoder_nlayer <= 0:
            raise ValueError("decoder_nlayer must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if not self.fsq_levels or any(level < 2 for level in self.fsq_levels):
            raise ValueError("fsq_levels must contain integers >= 2.")
        if self.latent_dim != len(self.fsq_levels):
            raise ValueError(
                "latent_dim must match fsq_levels length; "
                f"got latent_dim={self.latent_dim}, levels={len(self.fsq_levels)}."
            )
        if len(self.right_arm_sign) != self.arm_dim:
            raise ValueError(
                "right_arm_sign length must equal arm_dim; "
                f"got {len(self.right_arm_sign)} and {self.arm_dim}."
            )
        if any(value not in (-1.0, 1.0) for value in self.right_arm_sign):
            raise ValueError("right_arm_sign values must be either -1 or 1.")
        for arm in ("left", "right"):
            lower = getattr(self, f"{arm}_action_min")
            upper = getattr(self, f"{arm}_action_max")
            if len(lower) != self.arm_dim or len(upper) != self.arm_dim:
                raise ValueError(
                    f"{arm} normalization bounds must have arm_dim entries."
                )
            if any(not math.isfinite(v) for v in (*lower, *upper)):
                raise ValueError(f"{arm} normalization bounds must be finite.")
            if any(lo >= hi for lo, hi in zip(lower, upper, strict=True)):
                raise ValueError(
                    f"Every {arm}_action_min must be smaller than {arm}_action_max."
                )

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> BTTIndependentConfig:
        payload = dict(cfg)
        for key in (
            "conv_kernel_sizes",
            "conv_strides",
            "fsq_levels",
            "right_arm_sign",
            "left_action_min",
            "left_action_max",
            "right_action_min",
            "right_action_max",
        ):
            if key in payload and isinstance(payload[key], list):
                payload[key] = tuple(payload[key])
        return cast(BTTIndependentConfig, build_dataclass_config(cls, payload))

    def to_dict(self) -> dict[str, Any]:
        return dataclass_config_to_dict(self)
