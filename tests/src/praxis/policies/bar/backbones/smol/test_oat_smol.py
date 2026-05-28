"""Tests for OAT Smol policy specifics."""

from __future__ import annotations

import pytest
import torch

from praxis.policies.bar.variants import OATSmolPolicy

from ._bounded_test_utils import (
    _DummyFixedTokenizer,
    _DummyProcessor,
    _DummyVLM,
    _make_bounded_policy,
    _make_io,
    _patch_bounded_vlm_inputs,
    _patch_smol,
)


def test_oat_smol_decode_keep_k_applies_to_generation_steps(monkeypatch):
    io = _make_io()
    tok = _DummyFixedTokenizer(fixed_len=6)
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    _patch_bounded_vlm_inputs(monkeypatch)

    policy = OATSmolPolicy.from_io(io, decode_keep_k=3)
    _actions, info = policy.predict_action_chunk(
        {"observation.state": torch.randn(1, 1, 4)}
    )
    assert info["decode_keep_k"] == 3
    assert info["decode_steps"] == 3
    assert policy._resolve_decode_keep_k() == 3


def test_oat_smol_init_and_max_decode_tokens(monkeypatch):
    policy = _make_bounded_policy(monkeypatch, OATSmolPolicy)
    assert policy.config.name == "bar_oat_smol"
    assert policy.max_code_seq_len == 4
    assert policy.max_decode_tokens == 4
    assert policy.config.decode_keep_k is None


def test_oat_smol_supervise_eos_is_false(monkeypatch):
    policy = _make_bounded_policy(monkeypatch, OATSmolPolicy)
    assert policy._supervise_eos is False


def test_oat_smol_decode_keep_k_calltime_override(monkeypatch):
    io = _make_io()
    tok = _DummyFixedTokenizer(fixed_len=4)
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    _patch_bounded_vlm_inputs(monkeypatch)

    policy = OATSmolPolicy.from_io(io, decode_keep_k=None)
    _actions, info = policy.predict_action_chunk(
        {"observation.state": torch.randn(1, 1, 4)}, decode_keep_k=1
    )
    assert info["decode_keep_k"] == 1
    assert info["decode_steps"] == 1


def test_oat_smol_decode_keep_k_validation(monkeypatch):
    with pytest.raises(ValueError, match="decode_keep_k must be >= 1"):
        _make_bounded_policy(monkeypatch, OATSmolPolicy, decode_keep_k=0)

    with pytest.raises(ValueError, match="decode_keep_k must be <= token_horizon"):
        _make_bounded_policy(monkeypatch, OATSmolPolicy, decode_keep_k=10)

    policy3 = _make_bounded_policy(monkeypatch, OATSmolPolicy, decode_keep_k=None)
    with pytest.raises(TypeError, match="decode_keep_k must be an int or None"):
        _ = policy3.predict_action_chunk(
            {"observation.state": torch.randn(1, 1, 4)},
            decode_keep_k="2",
        )
