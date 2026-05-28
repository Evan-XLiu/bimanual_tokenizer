"""Tests for Bin Smol policy specifics."""

from __future__ import annotations

from praxis.policies.bar.variants import BinSmolPolicy

from ._bounded_test_utils import _make_bounded_policy


def test_bin_smol_init_and_max_decode_tokens(monkeypatch):
    policy = _make_bounded_policy(monkeypatch, BinSmolPolicy)
    assert policy.config.name == "bar_bin_smol"
    assert policy.max_code_seq_len == 4
    assert policy.max_decode_tokens == 4


def test_bin_smol_supervise_eos_is_false(monkeypatch):
    policy = _make_bounded_policy(monkeypatch, BinSmolPolicy)
    assert policy._supervise_eos is False
