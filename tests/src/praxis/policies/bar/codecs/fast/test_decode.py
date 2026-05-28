"""Tests for FAST variable-length token decode helpers."""

from __future__ import annotations

import logging

import torch

from praxis.policies.bar.codecs.fast.decode import (
    detokenize_fast_action_id_lists,
)

from ...core._test_utils import DummyVariableTokenizer


def test_detokenize_fast_action_id_lists_pads_before_decode():
    tok = DummyVariableTokenizer()
    out = detokenize_fast_action_id_lists(
        action_tokenizer=tok,
        batch_token_ids=[[1, 2], [3]],
        device=torch.device("cpu"),
        pad_token_id=9,
        decode_strict=None,
        decode_failure_mode="warn_zero",
        action_horizon=2,
        action_dim=2,
        policy_name="DummyFASTPolicy",
        logger=logging.getLogger(__name__),
    )
    expected = torch.tensor([[1.0, 2.0], [3.0, 9.0]])
    assert out.equal(expected)
    assert tok.last_detokenize_tokens.equal(expected.long())
