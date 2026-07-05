# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for OAT Pali policy specifics."""

from __future__ import annotations

import pytest
import torch

from praxis.policies.bar.backbones.pali.pali_base import PaliPolicyBase
from praxis.policies.bar.variants import OATPaliPolicy

from ._bounded_test_utils import (
    _DummyFixedTokenizer,
    _DummyProcessor,
    _DummyVLM,
    _make_bounded_policy,
    _make_io,
    _patch_bounded_vlm_inputs,
    _patch_pali,
)


def test_oat_pali_decode_keep_k_applies_to_generation_steps(monkeypatch):
    io = _make_io()
    tok = _DummyFixedTokenizer(fixed_len=6)
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_pali(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    _patch_bounded_vlm_inputs(monkeypatch)

    policy = OATPaliPolicy.from_io(io, decode_keep_k=3)
    monkeypatch.setattr(
        "praxis.policies.bar.core.vlm.decoding.sample_next_token_ids",
        lambda logits, *, temperature, top_k, allowed_token_ids=None: torch.full(
            (logits.shape[0], 1),
            fill_value=int(allowed_token_ids[0]),
            dtype=torch.long,
            device=logits.device,
        ),
    )

    _, info = policy.predict_action_chunk({"observation.state": torch.randn(1, 1, 4)})

    assert policy._resolve_decode_keep_k() == 3
    assert info["pred_action_token_ids"] == [[0, 0, 0]]
    assert info["block_schedule"] == [1, 1, 1, 1, 1, 1]
    assert info["decode_steps"] == 3
    assert info["decode_keep_k"] == 3
    assert len(vlm.prepare_calls) == 3


def test_oat_pali_init_and_max_decode_tokens(monkeypatch):
    policy = _make_bounded_policy(monkeypatch, OATPaliPolicy)
    assert policy.config.name == "bar_oat_pali"
    assert policy.max_code_seq_len == 4
    assert policy.max_decode_tokens == 4
    assert policy.config.decode_keep_k is None
    assert isinstance(policy, PaliPolicyBase)


def test_oat_pali_supervise_eos_is_false(monkeypatch):
    policy = _make_bounded_policy(monkeypatch, OATPaliPolicy)
    assert policy._supervise_eos is False


def test_oat_pali_decode_keep_k_calltime_override(monkeypatch):
    io = _make_io()
    tok = _DummyFixedTokenizer(fixed_len=4)
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_pali(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    _patch_bounded_vlm_inputs(monkeypatch)

    policy = OATPaliPolicy.from_io(io, decode_keep_k=None)
    monkeypatch.setattr(
        "praxis.policies.bar.core.vlm.decoding.sample_next_token_ids",
        lambda logits, *, temperature, top_k, allowed_token_ids=None: torch.full(
            (logits.shape[0], 1),
            fill_value=int(allowed_token_ids[0]),
            dtype=torch.long,
            device=logits.device,
        ),
    )
    _, info = policy.predict_action_chunk(
        {"observation.state": torch.randn(1, 1, 4)}, decode_keep_k=1
    )
    assert info["block_schedule"] == [1, 1, 1, 1]
    assert info["decode_steps"] == 1
    assert info["decode_keep_k"] == 1


def test_oat_pali_decode_keep_k_validation(monkeypatch):
    with pytest.raises(ValueError, match="decode_keep_k must be >= 1"):
        _make_bounded_policy(monkeypatch, OATPaliPolicy, decode_keep_k=0)

    with pytest.raises(ValueError, match="decode_keep_k must be <= token_horizon"):
        _make_bounded_policy(monkeypatch, OATPaliPolicy, decode_keep_k=10)

    policy3 = _make_bounded_policy(monkeypatch, OATPaliPolicy, decode_keep_k=None)
    with pytest.raises(TypeError, match="decode_keep_k must be an int or None"):
        _ = policy3.predict_action_chunk(
            {"observation.state": torch.randn(1, 1, 4)},
            decode_keep_k="2",
        )


def test_decode_keep_k_requires_prefix_detokenize_support(monkeypatch):
    class _NoPrefixTokenizer(_DummyFixedTokenizer):
        @property
        def supports_prefix_decode(self) -> bool:
            return False

    io = _make_io()
    tok = _NoPrefixTokenizer(fixed_len=4)
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_pali(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    _patch_bounded_vlm_inputs(monkeypatch)

    policy = OATPaliPolicy.from_io(io, decode_keep_k=None)
    with pytest.raises(ValueError, match="supports_prefix_decode=True"):
        policy.predict_action_chunk(
            {"observation.state": torch.randn(1, 1, 4)},
            decode_keep_k=1,
        )
