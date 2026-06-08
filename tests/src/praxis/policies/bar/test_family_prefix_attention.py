# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Regression guard for family-scoped prefix attention defaults."""

from praxis.policies.bar.backbones.pali.pali_base import PaliPolicyBase
from praxis.policies.bar.backbones.qwen.qwen_base import QwenPolicyBase
from praxis.policies.bar.backbones.smol.smol_base import SmolPolicyBase
from praxis.policies.ki.backbones.pali.pali_base import (
    PaliPolicyBase as KiPaliPolicyBase,
)
from praxis.policies.ki.backbones.qwen.qwen_base import (
    QwenPolicyBase as KiQwenPolicyBase,
)
from praxis.policies.ki.backbones.smol.smol_base import (
    SmolPolicyBase as KiSmolPolicyBase,
)


def test_family_prefix_attention_scopes():
    assert PaliPolicyBase._PREFIX_ATTENTION == "bidirectional"
    assert SmolPolicyBase._PREFIX_ATTENTION == "causal"
    assert QwenPolicyBase._PREFIX_ATTENTION == "causal"
    assert KiPaliPolicyBase._PREFIX_ATTENTION == "bidirectional"
    assert KiSmolPolicyBase._PREFIX_ATTENTION == "causal"
    assert KiQwenPolicyBase._PREFIX_ATTENTION == "causal"
