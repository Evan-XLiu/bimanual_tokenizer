# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for FAST Qwen policy specifics."""

from __future__ import annotations

import pytest
import torch

from praxis.policies.bar.variants import FASTQwenPolicy

from ._test_utils import (
    IM_END_TOKEN_ID,
    DummyFASTTokenizer,
    DummyProcessor,
    DummyVLM,
    make_io,
    make_policy,
    patch_fast_qwen_dependencies,
    stats_for_io,
)


class TestFASTQwenPolicy:
    def test_max_decode_tokens_inferred_from_tokenizer(self, monkeypatch):
        policy = make_policy(monkeypatch)
        assert policy.max_decode_tokens == policy.action_tokenizer.max_code_seq_len + 1

    def test_missing_max_code_seq_len_requires_max_new_tokens(self, monkeypatch):
        io = make_io()
        tokenizer = DummyFASTTokenizer()
        delattr(tokenizer, "max_token_length")
        patch_fast_qwen_dependencies(
            monkeypatch,
            tokenizer=tokenizer,
            vlm=DummyVLM(vocab_size=200, hidden_dim=16),
            processor=DummyProcessor(),
        )

        with pytest.raises(ValueError, match="max_new_tokens"):
            FASTQwenPolicy.from_io(
                io,
                action_tokenizer={
                    "kind": "tokens",
                    "name": "fast",
                    "pretrained_path": None,
                },
                freeze_action_tokenizer=True,
            )

    def test_predict_action_chunk_shape(self, monkeypatch):
        policy = make_policy(monkeypatch)
        io = make_io()
        policy.set_normalization_stats(stats_for_io(io))
        policy.eval()
        policy.vlm._generate_action_ids = [
            policy._require_action_token_vocabulary().token_min + i for i in [1, 2, 3]
        ]

        obs = {"observation.state": torch.randn(1, 1, 4)}
        actions, _ = policy.predict_action_chunk(obs)
        assert actions.shape == (1, 2, 2)

    def test_decode_strict_override_is_forwarded(self, monkeypatch):
        policy = make_policy(monkeypatch)
        policy.decode_strict = True
        captured: dict[str, object] = {}

        def spy_detokenize(tokens, **kwargs):
            captured["kwargs"] = dict(kwargs)
            batch_size = int(tokens.shape[0])
            return torch.zeros(
                (batch_size, 2, 2), dtype=torch.float32, device=tokens.device
            )

        monkeypatch.setattr(policy.action_tokenizer, "detokenize", spy_detokenize)
        _ = policy._decode_generated_action_tokens([[1, 2]], device=torch.device("cpu"))
        assert captured["kwargs"].get("strict_decode") is True

    def test_decode_failure_warn_zero_returns_fallback(self, monkeypatch):
        policy = make_policy(monkeypatch)
        policy.decode_strict = True
        policy.decode_failure_mode = "warn_zero"

        def raise_detokenize(_tokens, **_kwargs):
            raise ValueError("mock strict decode mismatch")

        monkeypatch.setattr(policy.action_tokenizer, "detokenize", raise_detokenize)
        out = policy._decode_generated_action_tokens(
            [[1, 2, 3]], device=torch.device("cpu")
        )
        assert out.shape == (1, policy.config.action_horizon, policy.config.action_dim)
        assert torch.all(out == 0)

    def test_pad_variable_tokens_respects_requested_device(self, monkeypatch):
        policy = make_policy(monkeypatch)
        assert policy._resolve_block_schedule() == (1,) * int(policy.max_decode_tokens)

    def test_supervise_eos_is_true(self, monkeypatch):
        policy = make_policy(monkeypatch)
        assert policy._supervise_eos is True
        assert policy._eos_token_id == IM_END_TOKEN_ID

    def test_eos_token_id_prefers_tokenizer_config(self, monkeypatch):
        io = make_io()
        processor = DummyProcessor()
        processor.tokenizer.eos_token_id = 7
        patch_fast_qwen_dependencies(
            monkeypatch,
            tokenizer=DummyFASTTokenizer(),
            vlm=DummyVLM(vocab_size=200, hidden_dim=16),
            processor=processor,
        )

        policy = FASTQwenPolicy.from_io(
            io,
            n_obs_steps=1,
            action_horizon=2,
            n_action_steps=1,
            action_tokenizer={
                "kind": "tokens",
                "name": "fast",
                "pretrained_path": None,
            },
            freeze_action_tokenizer=True,
        )
        assert policy._eos_token_id == 7

    def test_missing_eos_token_id_raises_for_fast(self, monkeypatch):
        io = make_io()
        processor = DummyProcessor()
        processor.tokenizer.eos_token_id = None
        patch_fast_qwen_dependencies(
            monkeypatch,
            tokenizer=DummyFASTTokenizer(),
            vlm=DummyVLM(vocab_size=200, hidden_dim=16),
            processor=processor,
        )

        with pytest.raises(ValueError, match="requires tokenizer.eos_token_id"):
            FASTQwenPolicy.from_io(
                io,
                n_obs_steps=1,
                action_horizon=2,
                n_action_steps=1,
                action_tokenizer={
                    "kind": "tokens",
                    "name": "fast",
                    "pretrained_path": None,
                },
                freeze_action_tokenizer=True,
            )

    def test_fast_decode_uses_4d_prefill_and_4d_decode_masks(self, monkeypatch):
        policy = make_policy(monkeypatch)
        io = make_io()
        policy.set_normalization_stats(stats_for_io(io))
        policy.max_action_tokens = 3
        policy.max_decode_tokens = 3
        policy.vlm._generate_action_ids = [
            policy._require_action_token_vocabulary().token_min + i for i in [1, 2, 3]
        ]

        _ = policy.predict_action_chunk({"observation.state": torch.zeros(1, 4)})
        assert isinstance(policy.vlm, DummyVLM)
        assert policy.vlm._last_prefill_attention_rank == 4
        assert policy.vlm._last_decode_attention_rank == 4

    def test_fast_custom_decode_sampling_overrides(self, monkeypatch):
        policy = make_policy(monkeypatch)
        io = make_io()
        policy.set_normalization_stats(stats_for_io(io))
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

    def test_fast_custom_decode_allowlist_includes_action_tokens_and_eos(
        self, monkeypatch
    ):
        policy = make_policy(monkeypatch)
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
        assert IM_END_TOKEN_ID in allowed
        assert policy._require_action_token_vocabulary().token_min in allowed
        assert policy._require_action_token_vocabulary().token_max in allowed

    def test_fast_decode_sampling_validation(self, monkeypatch):
        policy = make_policy(monkeypatch)
        io = make_io()
        policy.set_normalization_stats(stats_for_io(io))
        with pytest.raises(ValueError, match="decode_temperature"):
            policy.predict_action_chunk(
                {"observation.state": torch.zeros(1, 4)},
                decode_temperature=-0.1,
            )
        with pytest.raises(ValueError, match="decode_top_k"):
            policy.predict_action_chunk(
                {"observation.state": torch.zeros(1, 4)},
                decode_top_k=0,
            )
