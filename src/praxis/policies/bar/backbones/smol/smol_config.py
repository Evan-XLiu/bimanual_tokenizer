# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Shared config dataclass for SmolVLM action-token policies."""

from __future__ import annotations

from dataclasses import dataclass

from praxis.policies.backbones.smol.vlm import SMOLVLM_MODEL_NAME
from praxis.policies.bar.core.vlm.config import VlmActionTokenPolicyConfig


@dataclass
class SmolPolicyConfig(VlmActionTokenPolicyConfig):
    """Hyperparameters shared by all SmolVLM action-token policies."""

    name: str = "smol_action_token"
    vlm_model_name: str = SMOLVLM_MODEL_NAME
    resize_imgs_with_padding: tuple[int, int] = (512, 512)
