# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen config for flow-matching policies."""

from __future__ import annotations

from dataclasses import dataclass

from praxis.policies.backbones.qwen.vlm import QWEN3_VL_MODEL_NAME
from praxis.policies.fm.core.config import PolicyConfig


@dataclass
class QwenPolicyConfig(PolicyConfig):
    """Hyperparameters shared by Qwen-backed policies in this family."""

    vlm_model_name: str = QWEN3_VL_MODEL_NAME
