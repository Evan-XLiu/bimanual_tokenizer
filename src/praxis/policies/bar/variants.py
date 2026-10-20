# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Concrete BAR policy variants.

The public policy matrix is method x codec x backbone, but those matrix cells do
not need one package each. This module keeps the concrete BAR variants together
while BAR core and backbone modules own the actual behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from praxis.policies.bar.backbones.molmo.molmo_base import MolmoPolicyBase
from praxis.policies.bar.backbones.molmo.molmo_config import MolmoPolicyConfig
from praxis.policies.bar.backbones.pali.pali_base import PaliPolicyBase
from praxis.policies.bar.backbones.pali.pali_config import (
    PaliPolicyConfig,
)
from praxis.policies.bar.backbones.qwen.qwen_base import QwenPolicyBase
from praxis.policies.bar.backbones.qwen.qwen_config import QwenPolicyConfig
from praxis.policies.bar.backbones.smol.smol_base import SmolPolicyBase
from praxis.policies.bar.backbones.smol.smol_config import SmolPolicyConfig
from praxis.policies.bar.codecs.fast.config import (
    FAST_RUNTIME_OVERRIDE_KEYS,
    default_fast_action_tokenizer_config,
)
from praxis.policies.bar.codecs.fast.vlm import FastVlmPolicyMixin
from praxis.policies.bar.core.stop import DecodeStopMode
from praxis.policies.core.action_tokens import action_tokenizer_config_factory


@dataclass
class BTTMolmoPolicyConfig(MolmoPolicyConfig):
    """BTT BAR with Molmo2-ER and discrete-text proprioception."""

    name: str = "bar_btt_molmo"
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("btt")
    )


class BTTMolmoPolicy(MolmoPolicyBase):
    CONFIG_CLASS = BTTMolmoPolicyConfig


@dataclass
class OATMolmoPolicyConfig(MolmoPolicyConfig):
    name: str = "bar_oat_molmo"
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("oat")
    )


class OATMolmoPolicy(MolmoPolicyBase):
    CONFIG_CLASS = OATMolmoPolicyConfig


@dataclass
class FASTMolmoPolicyConfig(MolmoPolicyConfig):
    PRETRAINED_RUNTIME_OVERRIDE_KEYS: ClassVar[frozenset[str]] = (
        MolmoPolicyConfig.PRETRAINED_RUNTIME_OVERRIDE_KEYS | FAST_RUNTIME_OVERRIDE_KEYS
    )
    name: str = "bar_fast_molmo"
    decode_stop_mode: DecodeStopMode = "eos"
    decode_strict: bool | None = None
    decode_failure_mode: str = "raise"
    action_tokenizer: dict[str, object] = field(
        default_factory=default_fast_action_tokenizer_config
    )


class FASTMolmoPolicy(FastVlmPolicyMixin, MolmoPolicyBase):
    CONFIG_CLASS = FASTMolmoPolicyConfig

    def _post_set_io(self) -> None:
        super()._post_set_io()
        # FAST's variable-length hook replaces the common fixed-length hook.
        self._build_state_prefix_projector()
        from praxis.policies.core.vlm.dtypes import parse_vlm_dtype

        self.to(dtype=parse_vlm_dtype(self.config.vlm_dtype))


@dataclass
class ACodecPaliPolicyConfig(PaliPolicyConfig):
    """Hyperparameters for ACodec-backed block Pali VLM policy."""

    name: str = "bar_acodec_pali"
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("acodec")
    )


class ACodecPaliPolicy(PaliPolicyBase):
    """ACodec Pali policy routed through the shared blockwise VLM path."""

    CONFIG_CLASS = ACodecPaliPolicyConfig


@dataclass
class BimanualOATPaliPolicyConfig(PaliPolicyConfig):
    """Hyperparameters for paired bimanual OAT Pali VLM policy."""

    name: str = "bar_bimanualoat_pali"
    decode_keep_k: int | None = None
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("bimanualoat")
    )


class BimanualOATPaliPolicy(PaliPolicyBase):
    """Paired bimanual OAT Pali policy."""

    CONFIG_CLASS = BimanualOATPaliPolicyConfig


@dataclass
class BinPaliPolicyConfig(PaliPolicyConfig):
    """Hyperparameters for Bin blockwise action-token Pali VLM policy."""

    name: str = "bar_bin_pali"
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("bin")
    )


class BinPaliPolicy(PaliPolicyBase):
    """Bin Pali policy routed through the shared blockwise VLM path."""

    CONFIG_CLASS = BinPaliPolicyConfig


@dataclass
class BOATPaliPolicyConfig(PaliPolicyConfig):
    """Hyperparameters for BOAT Pali VLM policy."""

    name: str = "bar_boat_pali"
    decode_keep_k: int | None = None
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("boat")
    )


class BOATPaliPolicy(PaliPolicyBase):
    """BOAT Pali policy routed through the shared fixed-budget VLM path."""

    CONFIG_CLASS = BOATPaliPolicyConfig


@dataclass
class FASTPaliPolicyConfig(PaliPolicyConfig):
    """Hyperparameters for FAST variable-length Pali VLM policy."""

    PRETRAINED_RUNTIME_OVERRIDE_KEYS: ClassVar[frozenset[str]] = (
        PaliPolicyConfig.PRETRAINED_RUNTIME_OVERRIDE_KEYS | FAST_RUNTIME_OVERRIDE_KEYS
    )

    name: str = "bar_fast_pali"
    decode_stop_mode: DecodeStopMode = "eos"
    decode_strict: bool | None = None
    decode_failure_mode: str = "raise"
    action_tokenizer: dict[str, object] = field(
        default_factory=default_fast_action_tokenizer_config
    )


class FASTPaliPolicy(FastVlmPolicyMixin, PaliPolicyBase):
    """Pali adapter for the shared FAST variable-length VLM codec."""

    CONFIG_CLASS = FASTPaliPolicyConfig


@dataclass
class OATPaliPolicyConfig(PaliPolicyConfig):
    """Hyperparameters for OAT blockwise action-token Pali VLM policy."""

    name: str = "bar_oat_pali"
    decode_keep_k: int | None = None
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("oat")
    )


class OATPaliPolicy(PaliPolicyBase):
    """OAT Pali policy routed through the shared blockwise action-token path."""

    CONFIG_CLASS = OATPaliPolicyConfig


@dataclass
class QueSTPaliPolicyConfig(PaliPolicyConfig):
    """Hyperparameters for QueST blockwise action-token Pali VLM policy."""

    name: str = "bar_quest_pali"
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("quest")
    )


class QueSTPaliPolicy(PaliPolicyBase):
    """QueST Pali policy routed through the shared blockwise VLM path."""

    CONFIG_CLASS = QueSTPaliPolicyConfig


@dataclass
class BQuestPaliPolicyConfig(PaliPolicyConfig):
    """Hyperparameters for bimanual QueST blockwise Pali VLM policy."""

    name: str = "bar_bquest_pali"
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("bquest")
    )


class BQuestPaliPolicy(PaliPolicyBase):
    """Bimanual QueST Pali policy routed through the blockwise VLM path."""

    CONFIG_CLASS = BQuestPaliPolicyConfig


@dataclass
class BTTPaliPolicyConfig(PaliPolicyConfig):
    """Hyperparameters for bimanual temporal-tokenizer Pali VLM policy."""

    name: str = "bar_btt_pali"
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("btt")
    )


class BTTPaliPolicy(PaliPolicyBase):
    """BTT Pali policy routed through the blockwise VLM path."""

    CONFIG_CLASS = BTTPaliPolicyConfig


@dataclass
class BTTIndependentPaliPolicyConfig(PaliPolicyConfig):
    """Hyperparameters for the independent-arm BTT ablation with Pali."""

    name: str = "bar_btt_independent_pali"
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("btt_independent")
    )


class BTTIndependentPaliPolicy(PaliPolicyBase):
    """Independent-arm BTT ablation routed through the Pali VLM path."""

    CONFIG_CLASS = BTTIndependentPaliPolicyConfig


@dataclass
class ACodecQwenPolicyConfig(QwenPolicyConfig):
    """Hyperparameters for ACodec-backed blockwise Qwen VLM policy."""

    name: str = "bar_acodec_qwen"
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("acodec")
    )


class ACodecQwenPolicy(QwenPolicyBase):
    """ACodec Qwen policy routed through the shared blockwise VLM path."""

    CONFIG_CLASS = ACodecQwenPolicyConfig


@dataclass
class BimanualOATQwenPolicyConfig(QwenPolicyConfig):
    """Hyperparameters for paired bimanual OAT Qwen VLM policy."""

    name: str = "bar_bimanualoat_qwen"
    decode_keep_k: int | None = None
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("bimanualoat")
    )


class BimanualOATQwenPolicy(QwenPolicyBase):
    """Paired bimanual OAT Qwen policy."""

    CONFIG_CLASS = BimanualOATQwenPolicyConfig


@dataclass
class BinQwenPolicyConfig(QwenPolicyConfig):
    """Hyperparameters for Bin blockwise action-token Qwen VLM policy."""

    name: str = "bar_bin_qwen"
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("bin")
    )


class BinQwenPolicy(QwenPolicyBase):
    """Bin Qwen policy routed through the shared blockwise VLM path."""

    CONFIG_CLASS = BinQwenPolicyConfig


@dataclass
class BOATQwenPolicyConfig(QwenPolicyConfig):
    """Hyperparameters for BOAT Qwen VLM policy."""

    name: str = "bar_boat_qwen"
    decode_keep_k: int | None = None
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("boat")
    )


class BOATQwenPolicy(QwenPolicyBase):
    """BOAT Qwen policy routed through the shared fixed-budget VLM path."""

    CONFIG_CLASS = BOATQwenPolicyConfig


@dataclass
class FASTQwenPolicyConfig(QwenPolicyConfig):
    """Hyperparameters for FAST variable-length Qwen VLM policy."""

    PRETRAINED_RUNTIME_OVERRIDE_KEYS: ClassVar[frozenset[str]] = (
        QwenPolicyConfig.PRETRAINED_RUNTIME_OVERRIDE_KEYS | FAST_RUNTIME_OVERRIDE_KEYS
    )

    name: str = "bar_fast_qwen"
    decode_stop_mode: DecodeStopMode = "eos"
    decode_strict: bool | None = None
    decode_failure_mode: str = "raise"
    action_tokenizer: dict[str, object] = field(
        default_factory=default_fast_action_tokenizer_config
    )


class FASTQwenPolicy(FastVlmPolicyMixin, QwenPolicyBase):
    """Qwen adapter for the shared FAST variable-length VLM codec."""

    CONFIG_CLASS = FASTQwenPolicyConfig


@dataclass
class OATQwenPolicyConfig(QwenPolicyConfig):
    """Hyperparameters for OAT bounded-token Qwen VLM policy."""

    name: str = "bar_oat_qwen"
    decode_keep_k: int | None = None
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("oat")
    )


class OATQwenPolicy(QwenPolicyBase):
    """OAT Qwen policy routed through the shared blockwise action-token path."""

    CONFIG_CLASS = OATQwenPolicyConfig


@dataclass
class QueSTQwenPolicyConfig(QwenPolicyConfig):
    """Hyperparameters for QueST blockwise action-token Qwen VLM policy."""

    name: str = "bar_quest_qwen"
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("quest")
    )


class QueSTQwenPolicy(QwenPolicyBase):
    """QueST Qwen policy routed through the shared blockwise VLM path."""

    CONFIG_CLASS = QueSTQwenPolicyConfig


@dataclass
class BQuestQwenPolicyConfig(QwenPolicyConfig):
    """Hyperparameters for bimanual QueST blockwise Qwen VLM policy."""

    name: str = "bar_bquest_qwen"
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("bquest")
    )


class BQuestQwenPolicy(QwenPolicyBase):
    """Bimanual QueST Qwen policy routed through the blockwise VLM path."""

    CONFIG_CLASS = BQuestQwenPolicyConfig


@dataclass
class BTTQwenPolicyConfig(QwenPolicyConfig):
    """Hyperparameters for bimanual temporal-tokenizer Qwen VLM policy."""

    name: str = "bar_btt_qwen"
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("btt")
    )


class BTTQwenPolicy(QwenPolicyBase):
    """BTT Qwen policy routed through the blockwise VLM path."""

    CONFIG_CLASS = BTTQwenPolicyConfig


@dataclass
class BTTIndependentQwenPolicyConfig(QwenPolicyConfig):
    """Hyperparameters for the independent-arm BTT ablation with Qwen."""

    name: str = "bar_btt_independent_qwen"
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("btt_independent")
    )


class BTTIndependentQwenPolicy(QwenPolicyBase):
    """Independent-arm BTT ablation routed through the Qwen VLM path."""

    CONFIG_CLASS = BTTIndependentQwenPolicyConfig


@dataclass
class ACodecSmolPolicyConfig(SmolPolicyConfig):
    """Hyperparameters for ACodec-backed blockwise Smol VLM policy."""

    name: str = "bar_acodec_smol"
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("acodec")
    )


class ACodecSmolPolicy(SmolPolicyBase):
    """ACodec Smol policy routed through the shared blockwise VLM path."""

    CONFIG_CLASS = ACodecSmolPolicyConfig


@dataclass
class BimanualOATSmolPolicyConfig(SmolPolicyConfig):
    """Hyperparameters for paired bimanual OAT Smol VLM policy."""

    name: str = "bar_bimanualoat_smol"
    decode_keep_k: int | None = None
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("bimanualoat")
    )


class BimanualOATSmolPolicy(SmolPolicyBase):
    """Paired bimanual OAT Smol policy."""

    CONFIG_CLASS = BimanualOATSmolPolicyConfig


@dataclass
class BinSmolPolicyConfig(SmolPolicyConfig):
    """Hyperparameters for Bin blockwise action-token Smol VLM policy."""

    name: str = "bar_bin_smol"
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("bin")
    )


class BinSmolPolicy(SmolPolicyBase):
    """Bin Smol policy routed through the shared blockwise VLM path."""

    CONFIG_CLASS = BinSmolPolicyConfig


@dataclass
class BOATSmolPolicyConfig(SmolPolicyConfig):
    """Hyperparameters for BOAT Smol VLM policy."""

    name: str = "bar_boat_smol"
    decode_keep_k: int | None = None
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("boat")
    )


class BOATSmolPolicy(SmolPolicyBase):
    """BOAT Smol policy routed through the shared fixed-budget VLM path."""

    CONFIG_CLASS = BOATSmolPolicyConfig


@dataclass
class FASTSmolPolicyConfig(SmolPolicyConfig):
    """Hyperparameters for FAST variable-length Smol VLM policy."""

    PRETRAINED_RUNTIME_OVERRIDE_KEYS: ClassVar[frozenset[str]] = (
        SmolPolicyConfig.PRETRAINED_RUNTIME_OVERRIDE_KEYS | FAST_RUNTIME_OVERRIDE_KEYS
    )

    name: str = "bar_fast_smol"
    decode_stop_mode: DecodeStopMode = "eos"
    decode_strict: bool | None = None
    decode_failure_mode: str = "raise"
    action_tokenizer: dict[str, object] = field(
        default_factory=default_fast_action_tokenizer_config
    )


class FASTSmolPolicy(FastVlmPolicyMixin, SmolPolicyBase):
    """Smol adapter for the shared FAST variable-length VLM codec."""

    CONFIG_CLASS = FASTSmolPolicyConfig


@dataclass
class OATSmolPolicyConfig(SmolPolicyConfig):
    """Hyperparameters for OAT bounded-token Smol VLM policy."""

    name: str = "bar_oat_smol"
    decode_keep_k: int | None = None
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("oat")
    )


class OATSmolPolicy(SmolPolicyBase):
    """OAT Smol policy routed through the shared blockwise action-token path."""

    CONFIG_CLASS = OATSmolPolicyConfig


@dataclass
class QueSTSmolPolicyConfig(SmolPolicyConfig):
    """Hyperparameters for QueST blockwise action-token Smol VLM policy."""

    name: str = "bar_quest_smol"
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("quest")
    )


class QueSTSmolPolicy(SmolPolicyBase):
    """QueST Smol policy routed through the shared blockwise VLM path."""

    CONFIG_CLASS = QueSTSmolPolicyConfig


@dataclass
class BQuestSmolPolicyConfig(SmolPolicyConfig):
    """Hyperparameters for bimanual QueST blockwise Smol VLM policy."""

    name: str = "bar_bquest_smol"
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("bquest")
    )


class BQuestSmolPolicy(SmolPolicyBase):
    """Bimanual QueST Smol policy routed through the blockwise VLM path."""

    CONFIG_CLASS = BQuestSmolPolicyConfig


@dataclass
class BTTSmolPolicyConfig(SmolPolicyConfig):
    """Hyperparameters for bimanual temporal-tokenizer Smol VLM policy."""

    name: str = "bar_btt_smol"
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("btt")
    )


class BTTSmolPolicy(SmolPolicyBase):
    """BTT Smol policy routed through the blockwise VLM path."""

    CONFIG_CLASS = BTTSmolPolicyConfig


@dataclass
class BTTIndependentSmolPolicyConfig(SmolPolicyConfig):
    """Hyperparameters for the independent-arm BTT ablation with Smol."""

    name: str = "bar_btt_independent_smol"
    action_tokenizer: dict[str, object] = field(
        default_factory=action_tokenizer_config_factory("btt_independent")
    )


class BTTIndependentSmolPolicy(SmolPolicyBase):
    """Independent-arm BTT ablation routed through the Smol VLM path."""

    CONFIG_CLASS = BTTIndependentSmolPolicyConfig


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
    "BQuestPaliPolicy",
    "BQuestPaliPolicyConfig",
    "BQuestQwenPolicy",
    "BQuestQwenPolicyConfig",
    "BQuestSmolPolicy",
    "BQuestSmolPolicyConfig",
    "BTTPaliPolicy",
    "BTTPaliPolicyConfig",
    "BTTIndependentPaliPolicy",
    "BTTIndependentPaliPolicyConfig",
    "BTTIndependentQwenPolicy",
    "BTTIndependentQwenPolicyConfig",
    "BTTIndependentSmolPolicy",
    "BTTIndependentSmolPolicyConfig",
    "BTTQwenPolicy",
    "BTTQwenPolicyConfig",
    "BTTSmolPolicy",
    "BTTSmolPolicyConfig",
    "BOATPaliPolicy",
    "BOATPaliPolicyConfig",
    "BOATQwenPolicy",
    "BOATQwenPolicyConfig",
    "BOATSmolPolicy",
    "BOATSmolPolicyConfig",
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
