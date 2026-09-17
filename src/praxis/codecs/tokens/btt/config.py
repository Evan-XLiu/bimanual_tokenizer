# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the continuous frontend of the BTT tokenizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from praxis.codecs.core import build_dataclass_config, dataclass_config_to_dict

DEFAULT_RIGHT_ARM_SIGN = (-1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0)

# Joint-wise extrema after applying the right-arm convention and pooling each
# corresponding [left_joint_j, canonical_right_joint_j] pair.  These seven
# values are the only normalization parameters used by either arm.
DEFAULT_SHARED_ACTION_MIN = (
    -1.3947148323059082,
    -0.5742834806442261,
    -0.1190902441740036,
    -1.9893786907196045,
    -1.4216828346252441,
    -3.468787908554077,
    0.0,
)
DEFAULT_SHARED_ACTION_MAX = (
    2.9455959796905518,
    3.3889312744140625,
    3.4514999389648438,
    1.9530924558639526,
    1.6339466571807861,
    3.010831117630005,
    1.0,
)


@dataclass
class BTTEncoderConfig:
    """Geometry for the bimanual temporal-tokenizer encoder frontend."""

    name: str = "btt"

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
    shared_action_min: tuple[float, ...] = DEFAULT_SHARED_ACTION_MIN
    shared_action_max: tuple[float, ...] = DEFAULT_SHARED_ACTION_MAX
    clamp_normalized_actions: bool = False

    def __post_init__(self) -> None:
        self.conv_kernel_sizes = tuple(int(value) for value in self.conv_kernel_sizes)
        self.conv_strides = tuple(int(value) for value in self.conv_strides)
        self.fsq_levels = tuple(int(value) for value in self.fsq_levels)
        self.right_arm_sign = tuple(float(value) for value in self.right_arm_sign)
        self.shared_action_min = tuple(float(value) for value in self.shared_action_min)
        self.shared_action_max = tuple(float(value) for value in self.shared_action_max)

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
        if (
            len(self.shared_action_min) != self.arm_dim
            or len(self.shared_action_max) != self.arm_dim
        ):
            raise ValueError(
                "shared_action_min and shared_action_max lengths must equal arm_dim; "
                f"got {len(self.shared_action_min)}, {len(self.shared_action_max)}, "
                f"and arm_dim={self.arm_dim}."
            )
        if any(
            lower >= upper
            for lower, upper in zip(
                self.shared_action_min, self.shared_action_max, strict=True
            )
        ):
            raise ValueError(
                "every shared_action_min value must be smaller than shared_action_max."
            )

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> BTTEncoderConfig:
        payload = dict(cfg)
        for key in (
            "conv_kernel_sizes",
            "conv_strides",
            "fsq_levels",
            "right_arm_sign",
            "shared_action_min",
            "shared_action_max",
        ):
            if key in payload and isinstance(payload[key], list):
                payload[key] = tuple(payload[key])
        return cast(BTTEncoderConfig, build_dataclass_config(cls, payload))

    def to_dict(self) -> dict[str, Any]:
        return dataclass_config_to_dict(self)
