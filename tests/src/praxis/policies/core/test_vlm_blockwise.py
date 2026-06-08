# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for blockwise VLM token helpers."""

from __future__ import annotations

from praxis.policies.core.vlm.blockwise import (
    render_block_token_string,
)
from praxis.policies.core.vlm.vocab import ActionTokenVocabulary


def test_render_block_token_string_keeps_mask_sentinel():
    vocab = ActionTokenVocabulary(
        tokens=("a0", "a1", "a2"),
        token_ids=(100, 101, 102),
        mask_token="<|action_mask|>",
        mask_token_id=42,
    )
    out = render_block_token_string(
        [0, 9, 2],
        mask_sentinel=9,
        vocab=vocab,
    )
    assert out == "a0<|action_mask|>a2"
