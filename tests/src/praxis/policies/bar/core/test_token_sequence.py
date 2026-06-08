# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for token-sequence helpers."""

from __future__ import annotations

import torch

from praxis.policies.bar.core.token_sequence import (
    detokenize_fixed_action_ids,
    pad_token_id_lists_to_horizon,
    pad_variable_token_id_lists,
    resolve_max_code_seq_len,
    resolve_pad_horizon,
    resolve_variable_action_token_budget,
    resolve_variable_decode_budget,
    resolve_variable_pad_token_id,
    tokenize_fixed_actions,
)

from ._test_utils import DummyFixedTokenizer, DummyVariableTokenizer


def test_pad_token_id_lists_to_horizon_pads_and_truncates():
    out = pad_token_id_lists_to_horizon(
        [[1, 2, 3], [4]],
        device=torch.device("cpu"),
        horizon=2,
    )
    expected = torch.tensor([[1, 2], [4, 0]], dtype=torch.long)
    assert out.equal(expected)


def test_pad_variable_token_id_lists_uses_pad_value():
    out = pad_variable_token_id_lists(
        [[1, 2], [3]],
        device=torch.device("cpu"),
        pad_value=9,
    )
    expected = torch.tensor([[1, 2], [3, 9]], dtype=torch.long)
    assert out.equal(expected)


def test_resolve_max_code_seq_len_reads_codec_metadata():
    tok = DummyFixedTokenizer()
    assert resolve_max_code_seq_len(action_tokenizer=tok, owner_name="Dummy") == 4


def test_resolve_pad_horizon_defaults_to_token_horizon():
    assert resolve_pad_horizon(None, token_horizon=4) == 4


def test_tokenize_fixed_actions_returns_id_lists():
    tok = DummyFixedTokenizer()
    out = tokenize_fixed_actions(tok, torch.zeros(1, 2, 2))
    assert out == [[1, 2, 3, 4]]


def test_detokenize_fixed_action_ids_pads_to_horizon():
    tok = DummyFixedTokenizer()
    out = detokenize_fixed_action_ids(
        action_tokenizer=tok,
        batch_token_ids=[[1, 2], [3]],
        device=torch.device("cpu"),
        horizon=3,
    )
    expected = torch.tensor([[1.0, 2.0, 0.0], [3.0, 0.0, 0.0]])
    assert out.equal(expected)
    assert tok.last_detokenize_tokens.equal(expected.long())


def test_resolve_variable_decode_budget_adds_eos_headroom():
    tok = DummyVariableTokenizer()
    assert (
        resolve_variable_decode_budget(None, action_tokenizer=tok, owner_name="Dummy")
        == 5
    )


def test_resolve_variable_action_token_budget_excludes_eos_headroom():
    tok = DummyVariableTokenizer()
    assert (
        resolve_variable_action_token_budget(
            None,
            action_tokenizer=tok,
            owner_name="Dummy",
        )
        == 4
    )


def test_resolve_variable_pad_token_id_prefers_tokenizer_metadata():
    tok = DummyVariableTokenizer()
    assert (
        resolve_variable_pad_token_id(action_tokenizer=tok, tokenizer_vocab_size=8) == 9
    )
