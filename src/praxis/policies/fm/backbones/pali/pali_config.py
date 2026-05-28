"""PaliGemma config for flow-matching policies."""

from __future__ import annotations

from dataclasses import dataclass

from praxis.policies.backbones.pali.vlm import PALIGEMMA_MODEL_NAME
from praxis.policies.fm.core.config import PolicyConfig


@dataclass
class PaliPolicyConfig(PolicyConfig):
    """Hyperparameters shared by PaliGemma-backed policies in this family."""

    vlm_model_name: str = PALIGEMMA_MODEL_NAME
