"""Tests for FAST Smol policy specifics."""

from __future__ import annotations

import pytest
import torch

import praxis.policies.factory as policy_factory
from praxis.policies.bar.variants import FASTSmolPolicy, FASTSmolPolicyConfig

from ._test_utils import (
    _DummyFASTTokenizer,
    _DummyProcessor,
    _DummyVLM,
    _make_io,
    _patch_smol,
)


def test_factory_registry_contains_smol_entries():
    names = set(policy_factory.available_policies())
    assert {"bar_fast_smol", "bar_oat_smol", "bar_bin_smol", "bar_quest_smol"}.issubset(
        names
    )


def test_fast_smol_config_defaults():
    cfg = FASTSmolPolicyConfig()
    assert cfg.name == "bar_fast_smol"
    assert cfg.action_dim is None
    assert cfg.resize_imgs_with_padding == (512, 512)


def test_fast_smol_from_io_rejects_explicit_policy_action_dim_mismatch():
    io = _make_io()
    with pytest.raises(ValueError, match="action_dim mismatch"):
        _ = FASTSmolPolicy.from_io(io, action_dim=999)


def test_fast_smol_pretrained_tokenizer_action_dim_mismatch_raises(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    tok.action_dim = 999
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    with pytest.raises(ValueError, match="Action tokenizer action_dim mismatch"):
        _ = FASTSmolPolicy.from_io(
            io,
            action_tokenizer={
                "kind": "tokens",
                "name": "fast",
                "pretrained_path": "/tmp/ckpt",
            },
        )


def test_fast_smol_max_decode_tokens_inferred_from_tokenizer(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    policy = FASTSmolPolicy.from_io(io)
    assert policy.max_decode_tokens == policy.action_tokenizer.max_code_seq_len + 1


def test_fast_smol_requires_eos_token_id(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    processor.tokenizer.eos_token_id = None
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)

    with pytest.raises(ValueError, match="eos_token_id"):
        _ = FASTSmolPolicy.from_io(io)


def test_fast_smol_missing_max_code_seq_len_requires_max_new_tokens(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    tok.max_token_length = None
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)

    with pytest.raises(ValueError, match="max_new_tokens"):
        _ = FASTSmolPolicy.from_io(io, max_new_tokens=None)


def test_fast_smol_predict_action_chunk_shape(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    policy = FASTSmolPolicy.from_io(io)
    policy.set_normalization_stats(
        {
            "observation.state": {
                "min": -torch.ones(4),
                "max": torch.ones(4),
                "mean": torch.zeros(4),
                "std": torch.ones(4),
            },
            "action": {
                "min": torch.full((2,), -1.0),
                "max": torch.full((2,), 1.0),
                "mean": torch.zeros(2),
                "std": torch.ones(2),
            },
        }
    )
    vlm._generate_action_ids = [
        policy._require_action_token_vocabulary().token_min + i for i in [1, 2, 3]
    ]
    actions, _ = policy.predict_action_chunk(
        {"observation.state": torch.randn(1, 1, 4)}
    )
    assert actions.shape == (1, 2, 2)


def test_fast_smol_decode_strict_override_is_forwarded(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    policy = FASTSmolPolicy.from_io(io, decode_strict=True)
    captured: dict[str, object] = {}

    def spy_detokenize(tokens, **kwargs):
        captured["kwargs"] = dict(kwargs)
        bsz = int(tokens.shape[0])
        return torch.zeros((bsz, 2, 2), dtype=torch.float32, device=tokens.device)

    monkeypatch.setattr(policy.action_tokenizer, "detokenize", spy_detokenize)
    _ = policy._decode_generated_action_tokens([[1, 2]], device=torch.device("cpu"))
    assert captured["kwargs"].get("strict_decode") is True


def test_fast_smol_decode_failure_warn_zero_returns_fallback(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    policy = FASTSmolPolicy.from_io(
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


def test_fast_smol_decode_uses_4d_prefill_and_4d_decode_masks(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    policy = FASTSmolPolicy.from_io(io)
    policy.max_action_tokens = 3
    policy.max_decode_tokens = 3
    vlm._generate_action_ids = [
        policy._require_action_token_vocabulary().token_min + i for i in [1, 2, 3]
    ]

    _ = policy.predict_action_chunk({"observation.state": torch.zeros(1, 4)})
    assert vlm._last_prefill_attention_rank == 4
    assert vlm._last_decode_attention_rank == 4


def test_fast_smol_custom_decode_sampling_overrides(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    policy = FASTSmolPolicy.from_io(io)
    policy.config.decode_temperature = 0.7
    policy.config.decode_top_k = 8
    captured: dict[str, int | float | None] = {}

    def spy_sample(logits, *, temperature, top_k, allowed_token_ids=None):
        del logits, allowed_token_ids
        captured["decode_temperature"] = temperature
        captured["decode_top_k"] = top_k
        return torch.full(
            (1, 1),
            fill_value=policy._require_action_token_vocabulary().token_min,
            dtype=torch.long,
            device=policy.vlm.device,
        )

    monkeypatch.setattr(
        "praxis.policies.bar.core.vlm.decoding.sample_next_token_ids",
        spy_sample,
    )
    _ = policy.predict_action_chunk(
        {"observation.state": torch.zeros(1, 4)},
        decode_temperature=0.6,
        decode_top_k=4,
    )
    assert captured["decode_temperature"] == 0.6
    assert captured["decode_top_k"] == 4


def test_fast_smol_custom_decode_allowlist_includes_action_tokens_and_eos(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    processor.tokenizer.eos_token_id = 7
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    policy = FASTSmolPolicy.from_io(io)
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
    _ = policy.predict_action_chunk({"observation.state": torch.zeros(1, 4)})
    allowed = set(captured["allowed"])
    assert 7 in allowed
    assert policy._require_action_token_vocabulary().token_min in allowed
    assert policy._require_action_token_vocabulary().token_max in allowed


def test_fast_smol_decode_sampling_validation(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    policy = FASTSmolPolicy.from_io(io)
    with pytest.raises(ValueError, match="decode_temperature"):
        _ = policy.predict_action_chunk(
            {"observation.state": torch.zeros(1, 4)},
            decode_temperature=-0.1,
        )
    with pytest.raises(ValueError, match="decode_top_k"):
        _ = policy.predict_action_chunk(
            {"observation.state": torch.zeros(1, 4)},
            decode_top_k=0,
        )


def test_fast_smol_supervise_eos_is_true(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    processor.tokenizer.eos_token_id = 7
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    policy = FASTSmolPolicy.from_io(io)
    assert policy._supervise_eos is True
    assert policy._eos_token_id == 7


def test_fast_smol_eos_token_id_prefers_tokenizer_config(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    processor.tokenizer.eos_token_id = 7
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    policy = FASTSmolPolicy.from_io(io)
    assert policy._eos_token_id == 7
