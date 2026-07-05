# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for action-token policy construction helpers."""

from __future__ import annotations

from types import SimpleNamespace

from praxis.policies.core.action_tokens import (
    ActionTokenPolicyBase,
    action_tokenizer_config_factory,
)


class DummyRuntimePolicy(ActionTokenPolicyBase):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(n_action_steps=2)
        self.last_tokenizer_io = None
        self.post_set_io_called = 0

        class _Tokenizer:
            def __init__(self, owner) -> None:
                self.owner = owner

            def set_io(self, io) -> None:
                self.owner.last_tokenizer_io = io

        self.action_tokenizer = _Tokenizer(self)

    def _validate_action_tokenizer_action_dim(self, io):
        self.validated_io = io

    def _post_set_io(self):
        self.post_set_io_called += 1

    def forward(self, batch):
        del batch
        raise NotImplementedError

    def predict_action(self, obs, **kwargs):
        del obs, kwargs
        raise NotImplementedError


def test_action_token_policy_set_io_sets_tokenizer_io_and_queue():
    policy = DummyRuntimePolicy()
    io = SimpleNamespace(
        obs_features=[],
        action_feature=SimpleNamespace(key="action"),
        n_action_steps=3,
        fps=10,
    )
    policy.set_io(io)
    assert policy.last_tokenizer_io.action_feature.key == "action"
    assert policy._action_queue.maxlen == policy.config.n_action_steps
    assert policy.post_set_io_called == 1


def test_action_tokenizer_config_factory_returns_fresh_token_config():
    factory = action_tokenizer_config_factory("oat")
    first = factory()
    second = factory()

    assert first == {"kind": "tokens", "name": "oat", "pretrained_path": None}
    assert second == first
    assert second is not first
