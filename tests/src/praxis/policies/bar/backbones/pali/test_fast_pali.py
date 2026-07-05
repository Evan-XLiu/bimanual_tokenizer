# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for FAST Pali policy specifics."""

from __future__ import annotations

import pytest
import torch

import praxis.policies.factory as policy_factory
from praxis.policies.bar.variants import FASTPaliPolicy, FASTPaliPolicyConfig

from ._test_utils import (
    _DummyFASTTokenizer,
    _DummyProcessor,
    _DummyVLM,
    _make_io,
    _patch_pali,
    _stats_for_io,
)


def test_factory_registry_contains_pali_entries():
    names = set(policy_factory.available_policies())
    assert {"bar_fast_pali", "bar_oat_pali", "bar_bin_pali", "bar_quest_pali"}.issubset(
        names
    )


def test_fast_pali_config_defaults():
    cfg = FASTPaliPolicyConfig()
    assert cfg.name == "bar_fast_pali"
    assert cfg.decode_failure_mode == "raise"


def test_fast_pali_max_decode_tokens_inferred_from_tokenizer(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_pali(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)

    policy = FASTPaliPolicy.from_io(io)
    assert policy.max_decode_tokens == policy.action_tokenizer.max_code_seq_len + 1
    assert policy.max_action_tokens == policy.action_tokenizer.max_code_seq_len


def test_fast_pali_supervise_eos_true(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_pali(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)

    policy = FASTPaliPolicy.from_io(io)
    assert policy._supervise_eos is True


def test_fast_pali_decode_strict_override_is_forwarded(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_pali(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)

    policy = FASTPaliPolicy.from_io(io, decode_strict=True)
    captured: dict[str, object] = {}

    def spy_detokenize(tokens, **kwargs):
        captured["kwargs"] = dict(kwargs)
        bsz = int(tokens.shape[0])
        return torch.zeros((bsz, 2, 2), dtype=torch.float32, device=tokens.device)

    monkeypatch.setattr(policy.action_tokenizer, "detokenize", spy_detokenize)
    _ = policy._decode_generated_action_tokens([[1, 2]], device=torch.device("cpu"))
    assert captured["kwargs"].get("strict_decode") is True


def test_fast_pali_decode_failure_warn_zero_returns_fallback(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_pali(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)

    policy = FASTPaliPolicy.from_io(
        io, decode_strict=True, decode_failure_mode="warn_zero"
    )

    def raise_detokenize(_tokens, **_kwargs):
        raise ValueError("mock strict decode mismatch")

    monkeypatch.setattr(policy.action_tokenizer, "detokenize", raise_detokenize)
    out = policy._decode_generated_action_tokens(
        [[1, 2, 3]], device=torch.device("cpu")
    )
    assert out.shape == (1, policy.config.action_horizon, policy.config.action_dim)
    assert torch.all(out == 0)


def test_fast_pali_default_decode_failure_mode_raises(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_pali(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)

    policy = FASTPaliPolicy.from_io(io, decode_strict=True)

    def raise_detokenize(_tokens, **_kwargs):
        raise ValueError("mock strict decode mismatch")

    monkeypatch.setattr(policy.action_tokenizer, "detokenize", raise_detokenize)
    with pytest.raises(ValueError, match="mock strict decode mismatch"):
        _ = policy._decode_generated_action_tokens(
            [[1, 2, 3]], device=torch.device("cpu")
        )


def test_fast_pali_no_eos_decode_does_not_emit_headroom_action(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_pali(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)

    policy = FASTPaliPolicy.from_io(io)
    policy.set_normalization_stats(_stats_for_io(_make_io()))
    captured: dict[str, torch.Tensor] = {}
    sample_calls = 0

    def action_only_sample(logits, *, temperature, top_k, allowed_token_ids=None):
        nonlocal sample_calls
        del temperature, top_k, allowed_token_ids
        sample_calls += 1
        return torch.full(
            (logits.shape[0], 1),
            fill_value=policy._require_action_token_vocabulary().token_min,
            dtype=torch.long,
            device=logits.device,
        )

    def spy_detokenize(tokens, **kwargs):
        del kwargs
        captured["tokens"] = tokens.detach().clone()
        bsz = int(tokens.shape[0])
        return torch.zeros((bsz, 2, 2), dtype=torch.float32, device=tokens.device)

    monkeypatch.setattr(
        "praxis.policies.bar.core.vlm.decoding.sample_next_token_ids",
        action_only_sample,
    )
    monkeypatch.setattr(policy.action_tokenizer, "detokenize", spy_detokenize)

    _actions, info = policy.predict_action_chunk(
        {"observation.state": torch.randn(1, 1, 4)}
    )

    assert info["stopped_on_eos"] is False
    assert len(info["pred_action_token_ids"][0]) == policy.max_action_tokens
    assert captured["tokens"].shape == (1, policy.max_action_tokens)
    assert sample_calls == policy.max_action_tokens + 1


def test_fast_pali_full_action_budget_can_stop_on_eos_headroom(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_pali(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)

    policy = FASTPaliPolicy.from_io(io)
    policy.set_normalization_stats(_stats_for_io(_make_io()))
    assert policy._eos_token_id is not None
    eos_token_id = int(policy._eos_token_id)
    captured: dict[str, torch.Tensor] = {}
    sample_calls = 0

    def sample_actions_then_eos(logits, *, temperature, top_k, allowed_token_ids=None):
        nonlocal sample_calls
        del temperature, top_k, allowed_token_ids
        sample_calls += 1
        token_id = (
            eos_token_id
            if sample_calls == policy.max_action_tokens + 1
            else policy._require_action_token_vocabulary().token_min
        )
        return torch.full(
            (logits.shape[0], 1),
            fill_value=int(token_id),
            dtype=torch.long,
            device=logits.device,
        )

    def spy_detokenize(tokens, **kwargs):
        del kwargs
        captured["tokens"] = tokens.detach().clone()
        bsz = int(tokens.shape[0])
        return torch.zeros((bsz, 2, 2), dtype=torch.float32, device=tokens.device)

    monkeypatch.setattr(
        "praxis.policies.bar.core.vlm.decoding.sample_next_token_ids",
        sample_actions_then_eos,
    )
    monkeypatch.setattr(policy.action_tokenizer, "detokenize", spy_detokenize)

    _actions, info = policy.predict_action_chunk(
        {"observation.state": torch.randn(1, 1, 4)}
    )

    assert info["stopped_on_eos"] is True
    assert len(info["pred_action_token_ids"][0]) == policy.max_action_tokens
    assert captured["tokens"].shape == (1, policy.max_action_tokens)
    assert sample_calls == policy.max_action_tokens + 1


def test_fast_pali_decode_sampling_validation(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_pali(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)

    policy = FASTPaliPolicy.from_io(io)
    with pytest.raises(ValueError, match="decode_temperature"):
        _ = policy.predict_action_chunk(
            {"observation.state": torch.zeros(1, 4)}, decode_temperature=-0.1
        )
    with pytest.raises(ValueError, match="decode_top_k"):
        _ = policy.predict_action_chunk(
            {"observation.state": torch.zeros(1, 4)}, decode_top_k=0
        )


def test_fast_pali_custom_decode_passes_topk_to_sampler(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_pali(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)

    policy = FASTPaliPolicy.from_io(io)
    policy.set_normalization_stats(_stats_for_io(_make_io()))
    captured: dict[str, int | float | None] = {}

    def spy_sample(logits, *, temperature, top_k, allowed_token_ids=None):
        del allowed_token_ids
        captured["temperature"] = temperature
        captured["top_k"] = top_k
        return torch.full(
            (logits.shape[0], 1),
            fill_value=policy._require_action_token_vocabulary().token_min,
            dtype=torch.long,
            device=logits.device,
        )

    monkeypatch.setattr(
        "praxis.policies.bar.core.vlm.decoding.sample_next_token_ids",
        spy_sample,
    )
    _ = policy.predict_action_chunk(
        {"observation.state": torch.randn(1, 1, 4)},
        decode_top_k=4,
    )

    assert captured["top_k"] == 4
    assert captured["temperature"] is None


def test_fast_pali_custom_decode_passes_temperature_without_topk(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_pali(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)

    policy = FASTPaliPolicy.from_io(io)
    policy.set_normalization_stats(_stats_for_io(_make_io()))
    captured: dict[str, int | float | None] = {}

    def spy_sample(logits, *, temperature, top_k, allowed_token_ids=None):
        del allowed_token_ids
        captured["temperature"] = temperature
        captured["top_k"] = top_k
        return torch.full(
            (logits.shape[0], 1),
            fill_value=policy._require_action_token_vocabulary().token_min,
            dtype=torch.long,
            device=logits.device,
        )

    monkeypatch.setattr(
        "praxis.policies.bar.core.vlm.decoding.sample_next_token_ids",
        spy_sample,
    )
    _ = policy.predict_action_chunk(
        {"observation.state": torch.randn(1, 1, 4)},
        decode_temperature=0.7,
    )

    assert captured["temperature"] == 0.7
    assert captured["top_k"] is None


def test_fast_pali_custom_decode_allowlist_includes_action_tokens_and_eos(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_pali(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)

    policy = FASTPaliPolicy.from_io(io)
    policy.set_normalization_stats(_stats_for_io(_make_io()))
    captured: dict[str, list[int]] = {}

    def spy_sample(logits, *, temperature, top_k, allowed_token_ids=None):
        del temperature, top_k
        assert allowed_token_ids is not None
        allowed = torch.as_tensor(allowed_token_ids).detach().cpu().tolist()
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
    assert int(policy._eos_token_id) in allowed
    assert policy._require_action_token_vocabulary().token_min in allowed
    assert policy._require_action_token_vocabulary().token_max in allowed
