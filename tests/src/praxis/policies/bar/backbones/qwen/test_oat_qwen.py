"""Tests for OAT Qwen policy specifics."""

from __future__ import annotations

import pytest
import torch

from praxis.policies.bar.variants import OATQwenPolicy

from ._bounded_test_utils import make_bounded_policy, make_io, stats_for_io


class TestOATQwenPolicy:
    def test_oat_qwen_init_and_max_decode_tokens(self, monkeypatch):
        policy = make_bounded_policy(monkeypatch, OATQwenPolicy)
        assert policy.config.name == "bar_oat_qwen"
        assert policy.max_code_seq_len == 4
        assert policy.max_decode_tokens == 4
        assert policy.config.decode_keep_k is None

    def test_oat_qwen_decode_keep_k_limits_generation(self, monkeypatch):
        policy = make_bounded_policy(monkeypatch, OATQwenPolicy, decode_keep_k=2)
        io = make_io()
        policy.set_normalization_stats(stats_for_io(io))
        policy.eval()
        obs = {"observation.state": torch.randn(1, 1, 4)}
        _actions, info = policy.predict_action_chunk(obs)
        assert info["decode_keep_k"] == 2
        assert info["decode_steps"] == 2
        assert len(info["pred_action_token_ids"][0]) == 2

    def test_oat_qwen_decode_keep_k_calltime_override(self, monkeypatch):
        policy = make_bounded_policy(monkeypatch, OATQwenPolicy, decode_keep_k=None)
        io = make_io()
        policy.set_normalization_stats(stats_for_io(io))
        policy.eval()
        obs = {"observation.state": torch.randn(1, 1, 4)}
        _actions, info = policy.predict_action_chunk(obs, decode_keep_k=1)
        assert info["decode_keep_k"] == 1
        assert info["decode_steps"] == 1
        assert len(info["pred_action_token_ids"][0]) == 1

    def test_oat_qwen_decode_keep_k_validation(self, monkeypatch):
        with pytest.raises(ValueError, match="decode_keep_k must be >= 1"):
            make_bounded_policy(monkeypatch, OATQwenPolicy, decode_keep_k=0)

        with pytest.raises(ValueError, match="decode_keep_k must be <= token_horizon"):
            make_bounded_policy(monkeypatch, OATQwenPolicy, decode_keep_k=10)

        policy3 = make_bounded_policy(monkeypatch, OATQwenPolicy, decode_keep_k=None)
        with pytest.raises(TypeError, match="decode_keep_k must be an int or None"):
            policy3.predict_action_chunk(
                {"observation.state": torch.randn(1, 1, 4)},
                decode_keep_k="2",
            )

    def test_oat_qwen_supervise_eos_is_false(self, monkeypatch):
        policy = make_bounded_policy(monkeypatch, OATQwenPolicy)
        assert policy._supervise_eos is False
