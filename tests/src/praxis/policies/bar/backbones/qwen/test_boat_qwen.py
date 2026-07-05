# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for BOAT Qwen fixed-budget policy behavior."""

from __future__ import annotations

import torch

from praxis.policies.bar.variants import BOATQwenPolicy

from ._bounded_test_utils import make_bounded_policy, make_io, stats_for_io


def test_boat_qwen_init_and_fixed_budget_defaults(monkeypatch):
    policy = make_bounded_policy(monkeypatch, BOATQwenPolicy)

    assert policy.config.name == "bar_boat_qwen"
    assert policy.config.action_tokenizer["name"] == "boat"
    assert policy.config.decode_stop_mode == "fixed_budget"
    assert policy.config.decode_keep_k is None
    assert policy._supervise_eos is False
    assert not hasattr(policy, "stop_head")


def test_boat_qwen_decode_keep_k_limits_generation(monkeypatch):
    policy = make_bounded_policy(monkeypatch, BOATQwenPolicy, decode_keep_k=2)
    io = make_io()
    policy.set_normalization_stats(stats_for_io(io))
    policy.eval()

    _actions, info = policy.predict_action_chunk(
        {"observation.state": torch.randn(1, 1, 4)}
    )

    assert info["decode_keep_k"] == 2
    assert info["decode_steps"] == 2
    assert len(info["pred_action_token_ids"][0]) == 2
    assert not any(str(key).startswith("pred_stop") for key in info)


def test_boat_qwen_decode_keep_k_calltime_override(monkeypatch):
    policy = make_bounded_policy(monkeypatch, BOATQwenPolicy, decode_keep_k=None)
    io = make_io()
    policy.set_normalization_stats(stats_for_io(io))
    policy.eval()

    _actions, info = policy.predict_action_chunk(
        {"observation.state": torch.randn(1, 1, 4)},
        decode_keep_k=1,
    )

    assert info["decode_keep_k"] == 1
    assert info["decode_steps"] == 1
