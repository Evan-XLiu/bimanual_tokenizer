"""Tests for QueST Pali policy specifics."""

from __future__ import annotations

from praxis.policies.bar.variants import QueSTPaliPolicy

from ._bounded_test_utils import _make_bounded_policy


def test_quest_pali_init_and_max_decode_tokens(monkeypatch):
    policy = _make_bounded_policy(monkeypatch, QueSTPaliPolicy)
    assert policy.config.name == "bar_quest_pali"
    assert policy.max_code_seq_len == 4
    assert policy.max_decode_tokens == 4


def test_quest_pali_supervise_eos_is_false(monkeypatch):
    policy = _make_bounded_policy(monkeypatch, QueSTPaliPolicy)
    assert policy._supervise_eos is False
