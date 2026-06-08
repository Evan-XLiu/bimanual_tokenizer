# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for BAR generated-token decode helpers."""

from __future__ import annotations

import torch

from praxis.policies.bar.core.vlm.decode_tokens import (
    build_allowed_decode_token_ids,
    build_prediction_inputs,
    extract_contiguous_action_token_ids,
    raw_token_ids_to_vlm_tensor,
)
from praxis.policies.core.vlm.vocab import ActionTokenVocabulary


def _vocab() -> ActionTokenVocabulary:
    return ActionTokenVocabulary(
        tokens=("<a0>", "<a1>", "<a2>"),
        token_ids=(10, 11, 12),
        mask_token="<mask>",
        mask_token_id=13,
    )


def test_build_allowed_decode_token_ids_appends_optional_eos():
    assert build_allowed_decode_token_ids(vocab=_vocab(), eos_token_id=99) == [
        10,
        11,
        12,
        99,
    ]


def test_raw_token_ids_to_vlm_tensor_maps_mask_and_action_ids():
    out = raw_token_ids_to_vlm_tensor(
        [[0, 7, 2]],
        device=torch.device("cpu"),
        mask_sentinel=7,
        vocab=_vocab(),
    )
    expected = torch.tensor([[10, 13, 12]], dtype=torch.long)
    assert out.equal(expected)


def test_build_prediction_inputs_copies_previous_block_prefix():
    prev = torch.tensor([[4, 5, 6]], dtype=torch.long)
    out = build_prediction_inputs(
        batch_size=1,
        block_size=2,
        device=torch.device("cpu"),
        prev_block_ids=prev,
        mask_sentinel=9,
    )
    expected = torch.tensor([[4, 5]], dtype=torch.long)
    assert out.equal(expected)


def test_extract_contiguous_action_token_ids_stops_at_first_gap():
    generated = torch.tensor([[4, 10, 11, 12, 2, 10]])
    out = extract_contiguous_action_token_ids(
        generated,
        vocab=_vocab(),
        prompt_length=1,
    )
    assert out == [[0, 1, 2]]
