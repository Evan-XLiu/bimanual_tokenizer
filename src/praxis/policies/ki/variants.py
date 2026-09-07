# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Concrete knowledge-insulated policy variants."""

from __future__ import annotations

from dataclasses import dataclass, field

from praxis.codecs.tokens.fast import default_fast_tokenizer_config
from praxis.policies.core.action_tokens import (
    VariableLengthActionTokenMixin,
    action_tokenizer_config_factory,
)
from praxis.policies.ki.backbones.pali.pali_base import PaliPolicyBase
from praxis.policies.ki.backbones.pali.pali_config import PaliPolicyConfig
from praxis.policies.ki.backbones.qwen.qwen_base import QwenPolicyBase
from praxis.policies.ki.backbones.qwen.qwen_config import QwenPolicyConfig
from praxis.policies.ki.backbones.smol.smol_base import SmolPolicyBase
from praxis.policies.ki.backbones.smol.smol_config import SmolPolicyConfig


@dataclass
class ACodecPaliPolicyConfig(PaliPolicyConfig):
    """Hyperparameters for ACodec Pali co-training."""

    name: str = "ki_acodec_pali"
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("acodec")
    )


class ACodecPaliPolicy(PaliPolicyBase):
    """PaliGemma VLM CE plus detached-prefix-K/V FM expert."""

    CONFIG_CLASS = ACodecPaliPolicyConfig


@dataclass
class BimanualOATPaliPolicyConfig(PaliPolicyConfig):
    """Hyperparameters for paired bimanual OAT Pali co-training."""

    name: str = "ki_bimanualoat_pali"
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("bimanualoat")
    )


class BimanualOATPaliPolicy(PaliPolicyBase):
    """PaliGemma co-training with paired bimanual OAT tokens."""

    CONFIG_CLASS = BimanualOATPaliPolicyConfig


@dataclass
class BinPaliPolicyConfig(PaliPolicyConfig):
    """Hyperparameters for Bin Pali co-training."""

    name: str = "ki_bin_pali"
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("bin")
    )


class BinPaliPolicy(PaliPolicyBase):
    """PaliGemma VLM CE plus detached-prefix-K/V FM expert."""

    CONFIG_CLASS = BinPaliPolicyConfig


@dataclass
class FASTPaliPolicyConfig(PaliPolicyConfig):
    """Hyperparameters for FAST variable-length Pali co-training."""

    name: str = "ki_fast_pali"
    ce_supervise_eos: bool = True
    action_tokenizer: dict[str, object] = field(
        default_factory=default_fast_tokenizer_config
    )


class FASTPaliPolicy(VariableLengthActionTokenMixin, PaliPolicyBase):
    """PaliGemma VLM CE plus detached-prefix-K/V FM expert."""

    CONFIG_CLASS = FASTPaliPolicyConfig


@dataclass
class OATPaliPolicyConfig(PaliPolicyConfig):
    """Hyperparameters for OAT Pali co-training."""

    name: str = "ki_oat_pali"
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("oat")
    )


class OATPaliPolicy(PaliPolicyBase):
    """PaliGemma VLM CE plus detached-prefix-K/V FM expert."""

    CONFIG_CLASS = OATPaliPolicyConfig


@dataclass
class QueSTPaliPolicyConfig(PaliPolicyConfig):
    """Hyperparameters for QueST Pali co-training."""

    name: str = "ki_quest_pali"
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("quest")
    )


class QueSTPaliPolicy(PaliPolicyBase):
    """PaliGemma VLM CE plus detached-prefix-K/V FM expert."""

    CONFIG_CLASS = QueSTPaliPolicyConfig


@dataclass
class ACodecQwenPolicyConfig(QwenPolicyConfig):
    """Hyperparameters for ACodec Qwen co-training."""

    name: str = "ki_acodec_qwen"
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("acodec")
    )


class ACodecQwenPolicy(QwenPolicyBase):
    """Qwen VLM CE plus detached-prefix-K/V FM expert."""

    CONFIG_CLASS = ACodecQwenPolicyConfig


@dataclass
class BimanualOATQwenPolicyConfig(QwenPolicyConfig):
    """Hyperparameters for paired bimanual OAT Qwen co-training."""

    name: str = "ki_bimanualoat_qwen"
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("bimanualoat")
    )


class BimanualOATQwenPolicy(QwenPolicyBase):
    """Qwen co-training with paired bimanual OAT tokens."""

    CONFIG_CLASS = BimanualOATQwenPolicyConfig


@dataclass
class BinQwenPolicyConfig(QwenPolicyConfig):
    """Hyperparameters for Bin Qwen co-training."""

    name: str = "ki_bin_qwen"
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("bin")
    )


class BinQwenPolicy(QwenPolicyBase):
    """Qwen VLM CE plus detached-prefix-K/V FM expert."""

    CONFIG_CLASS = BinQwenPolicyConfig


@dataclass
class FASTQwenPolicyConfig(QwenPolicyConfig):
    """Hyperparameters for FAST variable-length Qwen co-training."""

    name: str = "ki_fast_qwen"
    ce_supervise_eos: bool = True
    action_tokenizer: dict[str, object] = field(
        default_factory=default_fast_tokenizer_config
    )


class FASTQwenPolicy(VariableLengthActionTokenMixin, QwenPolicyBase):
    """Qwen VLM CE plus detached-prefix-K/V FM expert."""

    CONFIG_CLASS = FASTQwenPolicyConfig


@dataclass
class OATQwenPolicyConfig(QwenPolicyConfig):
    """Hyperparameters for OAT Qwen co-training."""

    name: str = "ki_oat_qwen"
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("oat")
    )


class OATQwenPolicy(QwenPolicyBase):
    """Qwen VLM CE plus detached-prefix-K/V FM expert."""

    CONFIG_CLASS = OATQwenPolicyConfig


def _xtokenizer_target_config() -> dict[str, object]:
    return {
        "name": "offline_targets",
        "kind": "tokens",
        "codebook_size": 2048,
        "block_schedule": [1] * 32,
    }


@dataclass
class XTokenizerQwenPolicyConfig(QwenPolicyConfig):
    """KI Qwen co-training from offline X-Tokenizer targets."""

    name: str = "ki_xtokenizer_qwen"
    action_tokenizer: dict[str, object] = field(
        default_factory=_xtokenizer_target_config
    )
    action_token_target_key: str | None = "action_token_ids"
    action_token_length_key: str | None = "action_token_length"


class XTokenizerQwenPolicy(QwenPolicyBase):
    CONFIG_CLASS = XTokenizerQwenPolicyConfig


@dataclass
class QueSTQwenPolicyConfig(QwenPolicyConfig):
    """Hyperparameters for QueST Qwen co-training."""

    name: str = "ki_quest_qwen"
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("quest")
    )


class QueSTQwenPolicy(QwenPolicyBase):
    """Qwen VLM CE plus detached-prefix-K/V FM expert."""

    CONFIG_CLASS = QueSTQwenPolicyConfig


@dataclass
class ACodecSmolPolicyConfig(SmolPolicyConfig):
    """Hyperparameters for ACodec Smol co-training."""

    name: str = "ki_acodec_smol"
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("acodec")
    )


class ACodecSmolPolicy(SmolPolicyBase):
    """Smol VLM CE plus detached-prefix-K/V FM expert."""

    CONFIG_CLASS = ACodecSmolPolicyConfig


@dataclass
class BimanualOATSmolPolicyConfig(SmolPolicyConfig):
    """Hyperparameters for paired bimanual OAT Smol co-training."""

    name: str = "ki_bimanualoat_smol"
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("bimanualoat")
    )


class BimanualOATSmolPolicy(SmolPolicyBase):
    """Smol co-training with paired bimanual OAT tokens."""

    CONFIG_CLASS = BimanualOATSmolPolicyConfig


@dataclass
class BinSmolPolicyConfig(SmolPolicyConfig):
    """Hyperparameters for Bin Smol co-training."""

    name: str = "ki_bin_smol"
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("bin")
    )


class BinSmolPolicy(SmolPolicyBase):
    """Smol VLM CE plus detached-prefix-K/V FM expert."""

    CONFIG_CLASS = BinSmolPolicyConfig


@dataclass
class FASTSmolPolicyConfig(SmolPolicyConfig):
    """Hyperparameters for FAST variable-length Smol co-training."""

    name: str = "ki_fast_smol"
    ce_supervise_eos: bool = True
    action_tokenizer: dict[str, object] = field(
        default_factory=default_fast_tokenizer_config
    )


class FASTSmolPolicy(VariableLengthActionTokenMixin, SmolPolicyBase):
    """Smol VLM CE plus detached-prefix-K/V FM expert."""

    CONFIG_CLASS = FASTSmolPolicyConfig


@dataclass
class OATSmolPolicyConfig(SmolPolicyConfig):
    """Hyperparameters for OAT Smol co-training."""

    name: str = "ki_oat_smol"
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("oat")
    )


class OATSmolPolicy(SmolPolicyBase):
    """Smol VLM CE plus detached-prefix-K/V FM expert."""

    CONFIG_CLASS = OATSmolPolicyConfig


@dataclass
class QueSTSmolPolicyConfig(SmolPolicyConfig):
    """Hyperparameters for QueST Smol co-training."""

    name: str = "ki_quest_smol"
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("quest")
    )


class QueSTSmolPolicy(SmolPolicyBase):
    """Smol VLM CE plus detached-prefix-K/V FM expert."""

    CONFIG_CLASS = QueSTSmolPolicyConfig


__all__ = [
    "ACodecPaliPolicy",
    "ACodecPaliPolicyConfig",
    "ACodecQwenPolicy",
    "ACodecQwenPolicyConfig",
    "ACodecSmolPolicy",
    "ACodecSmolPolicyConfig",
    "BinPaliPolicy",
    "BinPaliPolicyConfig",
    "BinQwenPolicy",
    "BinQwenPolicyConfig",
    "BinSmolPolicy",
    "BinSmolPolicyConfig",
    "BimanualOATPaliPolicy",
    "BimanualOATPaliPolicyConfig",
    "BimanualOATQwenPolicy",
    "BimanualOATQwenPolicyConfig",
    "BimanualOATSmolPolicy",
    "BimanualOATSmolPolicyConfig",
    "FASTPaliPolicy",
    "FASTPaliPolicyConfig",
    "FASTQwenPolicy",
    "FASTQwenPolicyConfig",
    "FASTSmolPolicy",
    "FASTSmolPolicyConfig",
    "OATPaliPolicy",
    "OATPaliPolicyConfig",
    "OATQwenPolicy",
    "OATQwenPolicyConfig",
    "OATSmolPolicy",
    "OATSmolPolicyConfig",
    "QueSTPaliPolicy",
    "QueSTPaliPolicyConfig",
    "QueSTQwenPolicy",
    "QueSTQwenPolicyConfig",
    "QueSTSmolPolicy",
    "QueSTSmolPolicyConfig",
]
