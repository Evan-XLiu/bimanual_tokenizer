# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Shared config dataclass for Qwen VLM action-token policies."""

from __future__ import annotations

from dataclasses import dataclass

from praxis.policies.backbones.qwen.vlm import QWEN3_VL_MODEL_NAME
from praxis.policies.bar.core.vlm.config import VlmActionTokenPolicyConfig


@dataclass
class QwenPolicyConfig(VlmActionTokenPolicyConfig):
    """Hyperparameters shared by all Qwen VLM action-token policies."""

    name: str = "qwen_action_token"
    vlm_model_name: str = QWEN3_VL_MODEL_NAME
