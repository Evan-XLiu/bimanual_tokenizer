"""Tests for shared bounded-token Qwen blockwise base behavior."""

from __future__ import annotations

import pytest
import torch

from praxis.policies.bar.variants import BinQwenPolicy, OATQwenPolicy, QueSTQwenPolicy

from ._bounded_test_utils import (
    make_bounded_policy,
    make_io,
    mock_build_vlm_inputs_all_ignore,
    stats_for_io,
)


class TestBoundedQwenBase:
    @pytest.mark.parametrize(
        "policy_cls", [BinQwenPolicy, QueSTQwenPolicy, OATQwenPolicy]
    )
    def test_forward_returns_loss(self, monkeypatch, policy_cls):
        policy = make_bounded_policy(monkeypatch, policy_cls)
        io = make_io()
        policy.set_normalization_stats(stats_for_io(io))

        batch = {
            "observation.state": torch.randn(2, 1, 4),
            "action": torch.randn(2, 2, 2),
            "action_is_pad": torch.ones(2, 2, dtype=torch.bool),
        }
        loss, info = policy(batch)
        assert loss.shape == ()
        assert isinstance(info, dict)

    @pytest.mark.parametrize(
        "policy_cls", [BinQwenPolicy, QueSTQwenPolicy, OATQwenPolicy]
    )
    def test_forward_all_ignore_labels_skips_batch(self, monkeypatch, policy_cls):
        policy = make_bounded_policy(monkeypatch, policy_cls)
        io = make_io()
        policy.set_normalization_stats(stats_for_io(io))
        monkeypatch.setattr(
            "praxis.policies.bar.backbones.qwen.qwen_base.QwenPolicyBase._build_vlm_inputs",
            mock_build_vlm_inputs_all_ignore,
        )
        batch = {
            "observation.state": torch.randn(2, 1, 4),
            "action": torch.randn(2, 2, 2),
            "action_is_pad": torch.ones(2, 2, dtype=torch.bool),
        }
        loss, info = policy(batch)
        assert torch.isfinite(loss)
        assert loss.item() == 0.0
        assert info.get("skipped_all_ignore_batch") == 1.0

    @pytest.mark.parametrize(
        "policy_cls", [BinQwenPolicy, QueSTQwenPolicy, OATQwenPolicy]
    )
    def test_predict_action_chunk_shape(self, monkeypatch, policy_cls):
        policy = make_bounded_policy(monkeypatch, policy_cls)
        io = make_io()
        policy.set_normalization_stats(stats_for_io(io))
        policy.eval()
        action_ids = [
            policy._require_action_token_vocabulary().token_min + i for i in range(4)
        ]
        policy.vlm._generate_action_ids = action_ids

        obs = {"observation.state": torch.randn(1, 1, 4)}
        actions, _ = policy.predict_action_chunk(obs)
        assert actions.shape == (1, 2, 2)

    def test_custom_decode_uses_4d_prefill_and_4d_decode_masks(self, monkeypatch):
        policy = make_bounded_policy(monkeypatch, BinQwenPolicy)
        io = make_io()
        policy.set_normalization_stats(stats_for_io(io))
        policy.vlm._generate_action_ids = [
            policy._require_action_token_vocabulary().token_min + i for i in range(4)
        ]

        _ = policy.predict_action_chunk({"observation.state": torch.randn(1, 1, 4)})
        assert policy.vlm._last_prefill_attention_rank == 4
        assert policy.vlm._last_decode_attention_rank == 4

    def test_bounded_decode_sampling_overrides(self, monkeypatch):
        policy = make_bounded_policy(monkeypatch, BinQwenPolicy)
        io = make_io()
        policy.set_normalization_stats(stats_for_io(io))
        captured: dict[str, int | float | None] = {}

        def spy_sample(logits, *, temperature, top_k, allowed_token_ids=None):
            del allowed_token_ids
            captured["decode_temperature"] = temperature
            captured["decode_top_k"] = top_k
            return torch.zeros(
                (logits.shape[0], 1), dtype=torch.long, device=logits.device
            )

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

    def test_bounded_decode_allowlist_excludes_eos(self, monkeypatch):
        policy = make_bounded_policy(monkeypatch, BinQwenPolicy)
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

    @pytest.mark.parametrize(
        "policy_cls", [BinQwenPolicy, QueSTQwenPolicy, OATQwenPolicy]
    )
    def test_frozen_tokenizer_excluded_from_state_dict(self, monkeypatch, policy_cls):
        policy = make_bounded_policy(monkeypatch, policy_cls)
        state_dict = policy.state_dict()
        tokenizer_keys = [
            key for key in state_dict if key.startswith("action_tokenizer.")
        ]
        assert len(tokenizer_keys) == 0

    @pytest.mark.parametrize("policy_cls", [BinQwenPolicy, QueSTQwenPolicy])
    def test_default_generation_steps_use_max_code_seq_len(
        self, monkeypatch, policy_cls
    ):
        policy = make_bounded_policy(monkeypatch, policy_cls, max_new_tokens=8)
        io = make_io()
        policy.set_normalization_stats(stats_for_io(io))
        policy.eval()
        _actions, info = policy.predict_action_chunk(
            {"observation.state": torch.randn(1, 1, 4)}
        )
        assert policy.max_decode_tokens == 8
        assert policy.max_code_seq_len == 4
        assert info["decode_keep_k"] == 4
        assert info["decode_steps"] == 4

    @pytest.mark.parametrize(
        "policy_cls", [BinQwenPolicy, QueSTQwenPolicy, OATQwenPolicy]
    )
    def test_policy_initializes_action_tokenizer_stats_once(
        self, monkeypatch, policy_cls
    ):
        policy = make_bounded_policy(monkeypatch, policy_cls)
        io = make_io()
        assert policy.action_tokenizer.stats_set_calls == 0

        policy.set_normalization_stats(stats_for_io(io))
        assert policy.action_tokenizer.stats_set_calls == 0

        policy.set_normalization_stats(stats_for_io(io))
        assert policy.action_tokenizer.stats_set_calls == 0

    @pytest.mark.parametrize(
        "policy_cls", [BinQwenPolicy, QueSTQwenPolicy, OATQwenPolicy]
    )
    def test_policy_does_not_overwrite_existing_action_tokenizer_stats(
        self, monkeypatch, policy_cls
    ):
        policy = make_bounded_policy(monkeypatch, policy_cls)
        io = make_io()
        policy.action_tokenizer.normalize_action = object()
        policy.action_tokenizer.unnormalize_action = object()

        policy.set_normalization_stats(stats_for_io(io))
        assert policy.action_tokenizer.stats_set_calls == 0
