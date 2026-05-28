"""Tests for Bin Pali policy specifics."""

from __future__ import annotations

from praxis.policies.bar.variants import BinPaliPolicy

from ._bounded_test_utils import _make_bounded_policy


def test_bin_pali_init_and_max_decode_tokens(monkeypatch):
    policy = _make_bounded_policy(monkeypatch, BinPaliPolicy)
    assert policy.config.name == "bar_bin_pali"
    assert policy.max_code_seq_len == 4
    assert policy.max_decode_tokens == 4


def test_bin_pali_supervise_eos_is_false(monkeypatch):
    policy = _make_bounded_policy(monkeypatch, BinPaliPolicy)
    assert policy._supervise_eos is False
