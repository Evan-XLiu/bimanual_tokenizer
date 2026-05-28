"""Tests for action-token decode metadata helpers."""

from __future__ import annotations

import torch

from praxis.policies.bar.core.decode_info import (
    TokenRegime,
    build_decode_info,
    build_fixed_length_decode_info,
)
from praxis.policies.bar.core.sampling import sample_next_token_ids


def test_build_decode_info_serializes_token_regime():
    info = build_decode_info(
        [[1, 2]],
        token_regime=TokenRegime.FIXED_LENGTH,
        decode_steps=2,
        block_schedule=[1, 1],
        decode_keep_k=2,
    )
    assert info["pred_action_token_ids"] == [[1, 2]]
    assert info["token_regime"] == "fixed_length"
    assert info["decode_steps"] == 2
    assert info["block_schedule"] == [1, 1]
    assert info["decode_keep_k"] == 2


def test_build_fixed_length_decode_info_reports_keep_k_and_steps():
    info = build_fixed_length_decode_info(
        pred_action_token_ids=[[1, 2, 3]],
        block_schedule=(1, 1, 2, 4),
    )
    assert info["decode_keep_k"] == 3
    assert info["decode_steps"] == 3
    assert info["block_schedule"] == [1, 1, 2, 4]


def test_sample_next_token_ids_respects_allowed_token_ids_in_greedy_mode():
    logits = torch.tensor([[0.1, 9.0, 8.0]], dtype=torch.float32)
    next_ids = sample_next_token_ids(
        logits, temperature=0.0, top_k=None, allowed_token_ids=[2]
    )
    assert next_ids.tolist() == [[2]]
