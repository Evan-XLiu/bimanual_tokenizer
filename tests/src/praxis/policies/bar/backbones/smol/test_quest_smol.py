# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for QueST Smol policy specifics."""

from __future__ import annotations

from praxis.policies.bar.variants import QueSTSmolPolicy

from ._bounded_test_utils import _make_bounded_policy


def test_quest_smol_init_and_max_decode_tokens(monkeypatch):
    policy = _make_bounded_policy(monkeypatch, QueSTSmolPolicy)
    assert policy.config.name == "bar_quest_smol"
    assert policy.max_code_seq_len == 4
    assert policy.max_decode_tokens == 4


def test_quest_smol_supervise_eos_is_false(monkeypatch):
    policy = _make_bounded_policy(monkeypatch, QueSTSmolPolicy)
    assert policy._supervise_eos is False
