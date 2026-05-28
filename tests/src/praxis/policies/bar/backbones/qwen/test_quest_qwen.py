"""Tests for QueST Qwen policy specifics."""

from __future__ import annotations

from praxis.policies.bar.variants import QueSTQwenPolicy

from ._bounded_test_utils import make_bounded_policy


class TestQueSTQwenPolicy:
    def test_quest_qwen_init_and_max_decode_tokens(self, monkeypatch):
        policy = make_bounded_policy(monkeypatch, QueSTQwenPolicy)
        assert policy.config.name == "bar_quest_qwen"
        assert policy.max_code_seq_len == 4
        assert policy.max_decode_tokens == 4

    def test_quest_qwen_supervise_eos_is_false(self, monkeypatch):
        policy = make_bounded_policy(monkeypatch, QueSTQwenPolicy)
        assert policy._supervise_eos is False
