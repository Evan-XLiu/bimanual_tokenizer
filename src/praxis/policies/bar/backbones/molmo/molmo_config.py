"""Molmo2-ER configuration for BAR action-token policies."""

from __future__ import annotations

from dataclasses import dataclass

from praxis.policies.backbones.molmo.vlm import MOLMO2_ER_MODEL_NAME
from praxis.policies.bar.core.vlm.config import VlmActionTokenPolicyConfig


@dataclass
class MolmoPolicyConfig(VlmActionTokenPolicyConfig):
    vlm_model_name: str = MOLMO2_ER_MODEL_NAME
    vlm_dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    lora_target_modules: tuple[str, ...] = (
        "att_proj", "attn_out", "ff_proj", "ff_out"
    )
    lora_modules_to_save: tuple[str, ...] = ("lm_head", "wte")
