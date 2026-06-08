# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Smol config for knowledge-insulated policies."""

from __future__ import annotations

from dataclasses import dataclass

from praxis.policies.backbones.smol.vlm import SMOLVLM_MODEL_NAME
from praxis.policies.ki.core.config import PolicyConfig


@dataclass
class SmolPolicyConfig(PolicyConfig):
    """Hyperparameters shared by Smol-backed policies in this family."""

    vlm_model_name: str = SMOLVLM_MODEL_NAME
    resize_imgs_with_padding: tuple[int, int] = (512, 512)

    def __post_init__(self) -> None:
        super().__post_init__()
        value = self.resize_imgs_with_padding
        if isinstance(value, list):
            value = tuple(int(item) for item in value)
        if len(value) != 2:
            raise ValueError(
                "resize_imgs_with_padding must contain exactly two integers."
            )
        self.resize_imgs_with_padding = (int(value[0]), int(value[1]))
