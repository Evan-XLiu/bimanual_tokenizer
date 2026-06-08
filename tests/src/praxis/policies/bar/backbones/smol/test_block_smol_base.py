# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared Smol block-policy base behavior."""

from __future__ import annotations

import pytest
import torch

from praxis.policies.bar.variants import BinSmolPolicy, OATSmolPolicy, QueSTSmolPolicy

from ._bounded_test_utils import (
    _DummyFixedTokenizer,
    _DummyProcessor,
    _DummyVLM,
    _make_io,
    _patch_bounded_vlm_inputs,
    _patch_smol,
    _stats_for_io,
)


@pytest.mark.parametrize("policy_cls", [BinSmolPolicy, OATSmolPolicy, QueSTSmolPolicy])
def test_bounded_smol_decode_uses_action_only_sampling(monkeypatch, policy_cls):
    io = _make_io()
    tok = _DummyFixedTokenizer(fixed_len=4)
    processor = _DummyProcessor()
    processor.tokenizer.eos_token_id = None
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    _patch_bounded_vlm_inputs(monkeypatch)

    policy = policy_cls.from_io(io)
    captured: dict[str, object] = {}

    def spy_sample(logits, *, temperature, top_k, allowed_token_ids=None):
        captured["temperature"] = temperature
        captured["top_k"] = top_k
        captured["allowed"] = (
            allowed_token_ids.detach().cpu().tolist()
            if allowed_token_ids is not None
            else None
        )
        token = int(allowed_token_ids[0])
        return torch.full(
            (logits.shape[0], 1), token, dtype=torch.long, device=logits.device
        )

    monkeypatch.setattr(
        "praxis.policies.bar.core.vlm.decoding.sample_next_token_ids",
        spy_sample,
    )
    _ = policy.predict_action_chunk({"observation.state": torch.randn(1, 1, 4)})
    assert captured["temperature"] is None
    assert captured["top_k"] is None
    allowed = set(captured["allowed"])
    assert 99 not in allowed
    assert policy._require_action_token_vocabulary().token_min in allowed
    assert policy._require_action_token_vocabulary().token_max in allowed


@pytest.mark.parametrize("policy_cls", [BinSmolPolicy, OATSmolPolicy, QueSTSmolPolicy])
def test_bounded_smol_forward_returns_loss(monkeypatch, policy_cls):
    io = _make_io()
    tok = _DummyFixedTokenizer(fixed_len=4)
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    policy = policy_cls.from_io(io)
    policy.set_normalization_stats(_stats_for_io(io))
    batch = {
        "observation.state": torch.randn(2, 1, 4),
        "action": torch.randn(2, 2, 2),
        "action_is_pad": torch.ones(2, 2, dtype=torch.bool),
    }
    loss, info = policy(batch)
    assert loss.shape == ()
    assert isinstance(info, dict)


@pytest.mark.parametrize("policy_cls", [BinSmolPolicy, OATSmolPolicy, QueSTSmolPolicy])
def test_bounded_smol_predict_action_chunk_shape(monkeypatch, policy_cls):
    io = _make_io()
    tok = _DummyFixedTokenizer(fixed_len=4)
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    _patch_bounded_vlm_inputs(monkeypatch)
    policy = policy_cls.from_io(io)
    policy.set_normalization_stats(_stats_for_io(io))
    vlm._generate_action_ids = [
        policy._require_action_token_vocabulary().token_min + i for i in range(4)
    ]
    actions, _ = policy.predict_action_chunk(
        {"observation.state": torch.randn(1, 1, 4)}
    )
    assert actions.shape == (1, 2, 2)


def test_bounded_smol_custom_decode_uses_4d_prefill_and_4d_decode_masks(monkeypatch):
    io = _make_io()
    tok = _DummyFixedTokenizer(fixed_len=4)
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    _patch_bounded_vlm_inputs(monkeypatch)
    policy = BinSmolPolicy.from_io(io)
    policy.set_normalization_stats(_stats_for_io(io))
    vlm._generate_action_ids = [
        policy._require_action_token_vocabulary().token_min + i for i in range(4)
    ]

    _ = policy.predict_action_chunk({"observation.state": torch.randn(1, 1, 4)})
    assert vlm._last_prefill_attention_rank == 4
    assert vlm._last_decode_attention_rank == 4


def test_bounded_smol_decode_sampling_overrides(monkeypatch):
    io = _make_io()
    tok = _DummyFixedTokenizer(fixed_len=4)
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    _patch_bounded_vlm_inputs(monkeypatch)
    policy = BinSmolPolicy.from_io(io)
    policy.set_normalization_stats(_stats_for_io(io))
    captured: dict[str, int | float | None] = {}

    def spy_sample(logits, *, temperature, top_k, allowed_token_ids=None):
        del allowed_token_ids
        captured["decode_temperature"] = temperature
        captured["decode_top_k"] = top_k
        return torch.zeros((logits.shape[0], 1), dtype=torch.long, device=logits.device)

    monkeypatch.setattr(
        "praxis.policies.bar.core.vlm.decoding.sample_next_token_ids",
        spy_sample,
    )
    _ = policy.predict_action_chunk(
        {"observation.state": torch.randn(1, 1, 4)},
        decode_temperature=0.6,
        decode_top_k=4,
    )
    assert captured["decode_temperature"] == 0.6
    assert captured["decode_top_k"] == 4


def test_bounded_smol_decode_allowlist_excludes_eos(monkeypatch):
    io = _make_io()
    tok = _DummyFixedTokenizer(fixed_len=4)
    processor = _DummyProcessor()
    processor.tokenizer.eos_token_id = 99
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    policy = BinSmolPolicy.from_io(io)
    captured: dict[str, list[int]] = {}

    def spy_sample(logits, *, temperature, top_k, allowed_token_ids=None):
        del temperature, top_k
        assert allowed_token_ids is not None
        allowed = allowed_token_ids.detach().cpu().tolist()
        captured["allowed"] = allowed
        token = int(allowed[0])
        return torch.full(
            (logits.shape[0], 1),
            fill_value=token,
            dtype=torch.long,
            device=logits.device,
        )

    monkeypatch.setattr(
        "praxis.policies.bar.core.vlm.decoding.sample_next_token_ids",
        spy_sample,
    )
    _ = policy.predict_action_chunk({"observation.state": torch.randn(1, 1, 4)})
    allowed = set(captured["allowed"])
    assert 99 not in allowed
    assert policy._require_action_token_vocabulary().token_min in allowed
    assert policy._require_action_token_vocabulary().token_max in allowed


@pytest.mark.parametrize("policy_cls", [BinSmolPolicy, OATSmolPolicy, QueSTSmolPolicy])
def test_bounded_smol_forward_all_ignore_labels_skips_batch(monkeypatch, policy_cls):
    io = _make_io()
    tok = _DummyFixedTokenizer(fixed_len=4)
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    policy = policy_cls.from_io(io)
    batch = {
        "observation.state": torch.randn(2, 1, 4),
        "action": torch.randn(2, 2, 2),
        "action_is_pad": torch.ones(2, 2, dtype=torch.bool),
    }
    loss, info = policy(batch)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    assert info.get("skipped_all_ignore_batch") == 1.0


@pytest.mark.parametrize("policy_cls", [BinSmolPolicy, OATSmolPolicy, QueSTSmolPolicy])
def test_bounded_smol_frozen_tokenizer_excluded_from_state_dict(
    monkeypatch, policy_cls
):
    io = _make_io()
    tok = _DummyFixedTokenizer(fixed_len=4)
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    policy = policy_cls.from_io(io)
    sd = policy.state_dict()
    assert not any(k.startswith("action_tokenizer.") for k in sd)


@pytest.mark.parametrize("policy_cls", [BinSmolPolicy, OATSmolPolicy, QueSTSmolPolicy])
def test_bounded_smol_initializes_action_tokenizer_stats_once(monkeypatch, policy_cls):
    io = _make_io()
    tok = _DummyFixedTokenizer(fixed_len=4)
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    policy = policy_cls.from_io(io)

    policy.set_normalization_stats(_stats_for_io(io))
    assert policy.action_tokenizer.stats_set_calls == 0
    policy.set_normalization_stats(_stats_for_io(io))
    assert policy.action_tokenizer.stats_set_calls == 0


@pytest.mark.parametrize("policy_cls", [BinSmolPolicy, OATSmolPolicy, QueSTSmolPolicy])
def test_bounded_smol_does_not_overwrite_existing_action_tokenizer_stats(
    monkeypatch, policy_cls
):
    io = _make_io()
    tok = _DummyFixedTokenizer(fixed_len=4)
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    policy = policy_cls.from_io(io)
    policy.action_tokenizer.normalize_action = object()
    policy.action_tokenizer.unnormalize_action = object()

    policy.set_normalization_stats(_stats_for_io(io))
    assert policy.action_tokenizer.stats_set_calls == 0


@pytest.mark.parametrize("policy_cls", [BinSmolPolicy, QueSTSmolPolicy])
def test_bounded_smol_default_generation_steps_use_max_code_seq_len(
    monkeypatch, policy_cls
):
    io = _make_io()
    tok = _DummyFixedTokenizer(fixed_len=4)
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    _patch_bounded_vlm_inputs(monkeypatch)

    policy = policy_cls.from_io(io, max_new_tokens=8)
    _actions, info = policy.predict_action_chunk(
        {"observation.state": torch.randn(1, 1, 4)}
    )
    assert policy.max_decode_tokens == 8
    assert policy.max_code_seq_len == 4
    assert info["decode_keep_k"] == 4
    assert info["decode_steps"] == 4
