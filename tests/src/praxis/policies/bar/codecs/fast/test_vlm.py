# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared FAST VLM codec wiring."""

from __future__ import annotations

import pytest

from praxis.policies.bar.codecs.fast.vlm import FastVlmPolicyMixin
from praxis.policies.bar.variants import FASTPaliPolicy, FASTQwenPolicy, FASTSmolPolicy


@pytest.mark.parametrize(
    "policy_cls",
    [FASTPaliPolicy, FASTQwenPolicy, FASTSmolPolicy],
)
def test_fast_vlm_backbone_adapters_use_shared_codec_mixin(policy_cls) -> None:
    assert issubclass(policy_cls, FastVlmPolicyMixin)
    for method_name in (
        "_post_set_io",
        "_init_token_modeling",
        "_resolve_block_schedule",
        "tokenize_actions_to_policy_tokens",
        "_decode_generated_action_tokens",
    ):
        assert method_name not in policy_cls.__dict__
