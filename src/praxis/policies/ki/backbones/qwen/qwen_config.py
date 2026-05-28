"""Qwen config for knowledge-insulated policies."""

from __future__ import annotations

from dataclasses import dataclass

from praxis.policies.backbones.qwen.vlm import QWEN3_VL_MODEL_NAME
from praxis.policies.ki.core.config import PolicyConfig


@dataclass
class QwenPolicyConfig(PolicyConfig):
    """Hyperparameters shared by Qwen-backed policies in this family."""

    vlm_model_name: str = QWEN3_VL_MODEL_NAME
