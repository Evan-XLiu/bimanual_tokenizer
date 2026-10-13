# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Molmo2-ER configuration for end-to-end flow matching."""

from __future__ import annotations

from dataclasses import dataclass

from praxis.policies.backbones.molmo.vlm import MOLMO2_ER_MODEL_NAME
from praxis.policies.fm.core.config import PolicyConfig


@dataclass
class MolmoPolicyConfig(PolicyConfig):
    """Molmo2-ER VLM and continuous-action expert without action tokens."""

    vlm_model_name: str = MOLMO2_ER_MODEL_NAME
    vlm_dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    lora_target_modules: tuple[str, ...] = (
        "att_proj",
        "attn_out",
        "ff_proj",
        "ff_out",
    )
