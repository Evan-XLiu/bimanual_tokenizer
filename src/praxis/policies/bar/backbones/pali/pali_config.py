# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Shared config dataclass for Pali VLM action-token policies."""

from __future__ import annotations

from dataclasses import dataclass

from praxis.policies.backbones.pali.vlm import PALIGEMMA_MODEL_NAME
from praxis.policies.bar.core.vlm.config import VlmActionTokenPolicyConfig


@dataclass
class PaliPolicyConfig(VlmActionTokenPolicyConfig):
    """Hyperparameters shared by Pali VLM action-token policies."""

    name: str = "pali_action_token"
    vlm_model_name: str = PALIGEMMA_MODEL_NAME
