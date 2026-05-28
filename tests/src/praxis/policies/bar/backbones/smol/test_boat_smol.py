"""Tests for BOAT Smol fixed-budget policy behavior."""

from __future__ import annotations

import torch

from praxis.policies.bar.variants import BOATSmolPolicy

from ._bounded_test_utils import _make_bounded_policy, _make_io, _stats_for_io


def test_boat_smol_init_and_fixed_budget_defaults(monkeypatch):
    policy = _make_bounded_policy(monkeypatch, BOATSmolPolicy)

    assert policy.config.name == "bar_boat_smol"
    assert policy.config.action_tokenizer["name"] == "boat"
    assert policy.config.decode_stop_mode == "fixed_budget"
    assert policy.config.decode_keep_k is None
    assert policy._supervise_eos is False
    assert not hasattr(policy, "stop_head")


def test_boat_smol_decode_keep_k_limits_generation(monkeypatch):
    policy = _make_bounded_policy(monkeypatch, BOATSmolPolicy, decode_keep_k=2)
    io = _make_io()
    policy.set_normalization_stats(_stats_for_io(io))
    policy.eval()

    _actions, info = policy.predict_action_chunk(
        {"observation.state": torch.randn(1, 1, 4)}
    )

    assert info["decode_keep_k"] == 2
    assert info["decode_steps"] == 2
    assert len(info["pred_action_token_ids"][0]) == 2
    assert not any(str(key).startswith("pred_stop") for key in info)


def test_boat_smol_decode_keep_k_calltime_override(monkeypatch):
    policy = _make_bounded_policy(monkeypatch, BOATSmolPolicy, decode_keep_k=None)
    io = _make_io()
    policy.set_normalization_stats(_stats_for_io(io))
    policy.eval()

    _actions, info = policy.predict_action_chunk(
        {"observation.state": torch.randn(1, 1, 4)},
        decode_keep_k=1,
    )

    assert info["decode_keep_k"] == 1
    assert info["decode_steps"] == 1
