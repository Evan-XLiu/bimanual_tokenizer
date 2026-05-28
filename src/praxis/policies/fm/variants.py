"""Concrete flow-matching policy variants."""

from __future__ import annotations

from dataclasses import dataclass

from praxis.policies.fm.backbones.pali.pali_base import PaliPolicyBase
from praxis.policies.fm.backbones.pali.pali_config import (
    PaliPolicyConfig as BasePaliPolicyConfig,
)
from praxis.policies.fm.backbones.qwen.qwen_base import QwenPolicyBase
from praxis.policies.fm.backbones.qwen.qwen_config import (
    QwenPolicyConfig as BaseQwenPolicyConfig,
)
from praxis.policies.fm.backbones.smol.smol_base import SmolPolicyBase
from praxis.policies.fm.backbones.smol.smol_config import (
    SmolPolicyConfig as BaseSmolPolicyConfig,
)


@dataclass
class PaliPolicyConfig(BasePaliPolicyConfig):
    """Hyperparameters for Pali FM training."""

    name: str = "fm_pali"


class PaliPolicy(PaliPolicyBase):
    """PaliGemma VLM plus end-to-end flow-matching action expert."""

    CONFIG_CLASS = PaliPolicyConfig


@dataclass
class QwenPolicyConfig(BaseQwenPolicyConfig):
    """Hyperparameters for Qwen FM training."""

    name: str = "fm_qwen"


class QwenPolicy(QwenPolicyBase):
    """Qwen VLM plus end-to-end flow-matching action expert."""

    CONFIG_CLASS = QwenPolicyConfig


@dataclass
class SmolPolicyConfig(BaseSmolPolicyConfig):
    """Hyperparameters for Smol FM training."""

    name: str = "fm_smol"


class SmolPolicy(SmolPolicyBase):
    """Smol VLM plus end-to-end flow-matching action expert."""

    CONFIG_CLASS = SmolPolicyConfig


__all__ = [
    "PaliPolicy",
    "PaliPolicyConfig",
    "QwenPolicy",
    "QwenPolicyConfig",
    "SmolPolicy",
    "SmolPolicyConfig",
]
