# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Molmo config for knowledge-insulated policies."""

from __future__ import annotations

from dataclasses import dataclass

from praxis.policies.backbones.molmo.vlm import MOLMO2_ER_MODEL_NAME
from praxis.policies.ki.core.config import PolicyConfig


@dataclass
class MolmoPolicyConfig(PolicyConfig):
    """Hyperparameters shared by Molmo-backed policies in this family."""

    vlm_model_name: str = MOLMO2_ER_MODEL_NAME
    lora_target_modules: tuple[str, ...] = (
        "att_proj",
        "attn_out",
        "ff_proj",
        "ff_out",
    )
    lora_modules_to_save: tuple[str, ...] = ("lm_head", "wte")
