# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Concrete flow-matching policy variants."""

from __future__ import annotations

from dataclasses import dataclass

from praxis.policies.fm.backbones.molmo.molmo_base import MolmoPolicyBase
from praxis.policies.fm.backbones.molmo.molmo_config import (
    MolmoPolicyConfig as BaseMolmoPolicyConfig,
)
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


@dataclass
class MolmoPolicyConfig(BaseMolmoPolicyConfig):
    """Hyperparameters for Molmo2-ER FM training."""

    name: str = "fm_molmo"


class MolmoPolicy(MolmoPolicyBase):
    """Molmo2-ER VLM plus end-to-end flow-matching action expert."""

    CONFIG_CLASS = MolmoPolicyConfig


__all__ = [
    "MolmoPolicy",
    "MolmoPolicyConfig",
    "PaliPolicy",
    "PaliPolicyConfig",
    "QwenPolicy",
    "QwenPolicyConfig",
    "SmolPolicy",
    "SmolPolicyConfig",
]
