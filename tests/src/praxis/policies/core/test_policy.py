# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared policy base behavior."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch.nn as nn
from safetensors.torch import load_file

from praxis.core.checkpoint_artifacts import resolve_pretrained_weight_file
from praxis.core.types import FeatureSpec, FeatureType, PolicyIO
from praxis.policies.core import BasePolicy
from praxis.policies.core.vlm.checkpoint import VlmCheckpointFilterMixin


class DummyPolicy(BasePolicy):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(n_obs_steps=1, n_action_steps=2)
        self.io = PolicyIO(
            obs_features=[
                FeatureSpec(
                    key="observation.state",
                    type=FeatureType.STATE,
                    shape=(2,),
                )
            ],
            action_feature=None,
            n_obs_steps=1,
            n_action_steps=2,
            fps=10,
        )
        self.normalize = None
        self._action_queue = deque(maxlen=2)

    def forward(self, batch):
        del batch
        raise NotImplementedError

    def predict_action(self, obs, **kwargs):
        del obs, kwargs
        raise NotImplementedError


class DynamicIoModulePolicy(BasePolicy):
    """Test policy that creates a trainable module only after PolicyIO is known."""

    def __init__(self, dynamic_dim: int = 3) -> None:
        super().__init__()
        self.dynamic_dim = int(dynamic_dim)
        self.config = SimpleNamespace(dynamic_dim=self.dynamic_dim)
        self.dynamic = None

    def set_io(self, io: PolicyIO) -> None:
        super().set_io(io)
        self.dynamic = nn.Linear(self.dynamic_dim, 1, bias=False)

    def forward(self, batch):
        del batch
        raise NotImplementedError

    def predict_action(self, obs, **kwargs):
        del obs, kwargs
        raise NotImplementedError


class ShardedIoModulePolicy(BasePolicy):
    """Policy with enough independent weights to force HF safetensor shards."""

    def __init__(self, dynamic_dim: int = 16) -> None:
        super().__init__()
        self.dynamic_dim = int(dynamic_dim)
        self.config = SimpleNamespace(dynamic_dim=self.dynamic_dim)
        self.left = None
        self.right = None

    def set_io(self, io: PolicyIO) -> None:
        super().set_io(io)
        self.left = nn.Linear(self.dynamic_dim, self.dynamic_dim, bias=False)
        self.right = nn.Linear(self.dynamic_dim, self.dynamic_dim, bias=False)

    def forward(self, batch):
        del batch
        raise NotImplementedError

    def predict_action(self, obs, **kwargs):
        del obs, kwargs
        raise NotImplementedError


class VlmOnlyFilterPolicy(VlmCheckpointFilterMixin, BasePolicy):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(use_lora=False)

    def forward(self, batch):
        del batch
        raise NotImplementedError

    def predict_action(self, obs, **kwargs):
        del obs, kwargs
        raise NotImplementedError


def _policy_io(dynamic_dim: int = 3) -> PolicyIO:
    return PolicyIO(
        obs_features=[
            FeatureSpec(
                key="observation.state",
                type=FeatureType.STATE,
                shape=(dynamic_dim,),
            )
        ],
        action_feature=FeatureSpec(
            key="action",
            type=FeatureType.ACTION,
            shape=(2,),
        ),
        n_obs_steps=1,
        n_action_steps=2,
        fps=10,
    )


def test_predict_action_from_chunk_queue_reuses_chunk_predictions():
    policy = DummyPolicy()
    obs = {"observation.state": torch.randn(2)}
    inference_calls = 0

    def predict_action_chunk(stacked_obs, **chunk_kwargs):
        nonlocal inference_calls
        del stacked_obs, chunk_kwargs
        inference_calls += 1
        return torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), {}

    action0 = policy._predict_action_from_chunk_queue(
        obs,
        predict_action_chunk=predict_action_chunk,
    )
    action1 = policy._predict_action_from_chunk_queue(
        obs,
        predict_action_chunk=predict_action_chunk,
    )

    assert action0.shape == (2,)
    assert action1.shape == (2,)
    assert inference_calls == 1
    assert len(policy._action_queue) == 0


def test_rollout_profiler_observes_chunk_queue_without_changing_actions(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("ROLLOUT_PROFILE", "1")
    policy = DummyPolicy()
    obs = {"observation.state": torch.randn(2)}
    inference_calls = 0

    def predict_action_chunk(stacked_obs, **chunk_kwargs):
        nonlocal inference_calls
        del stacked_obs, chunk_kwargs
        inference_calls += 1
        return torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), {}

    action0 = policy._predict_action_from_chunk_queue(
        obs,
        predict_action_chunk=predict_action_chunk,
    )
    event0 = dict(policy._rollout_profile_last_step or {})
    action1 = policy._predict_action_from_chunk_queue(
        obs,
        predict_action_chunk=predict_action_chunk,
    )
    event1 = dict(policy._rollout_profile_last_step or {})

    torch.testing.assert_close(action0, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(action1, torch.tensor([3.0, 4.0]))
    assert inference_calls == 1
    assert event0["new_inference"] is True
    assert event0["chunk_remaining_before"] == 0
    assert event0["predicted_chunk"] == 2
    assert event0["executed_chunk_length"] == 2
    assert event1["new_inference"] is False
    assert event1["chunk_remaining_before"] == 1
    assert event1["predicted_chunk"] is None


def test_predict_action_from_chunk_queue_batches_observation_by_policy_io_shape():
    policy = DummyPolicy()
    policy.set_io(
        PolicyIO(
            obs_features=[
                FeatureSpec(
                    key="observation.matrix",
                    type=FeatureType.STATE,
                    shape=(2, 2),
                )
            ],
            action_feature=FeatureSpec(
                key="action",
                type=FeatureType.ACTION,
                shape=(2,),
            ),
            n_obs_steps=1,
            n_action_steps=2,
        )
    )
    policy.reset()
    obs = {"observation.matrix": torch.randn(2, 2)}

    def predict_action_chunk(stacked_obs, **chunk_kwargs):
        del chunk_kwargs
        assert stacked_obs["observation.matrix"].shape == (1, 1, 2, 2)
        return torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), {}

    action = policy._predict_action_from_chunk_queue(
        obs,
        predict_action_chunk=predict_action_chunk,
    )

    assert action.shape == (2,)


def test_predict_action_from_chunk_queue_rejects_bare_tensor_chunk_result():
    policy = DummyPolicy()
    obs = {"observation.state": torch.randn(2)}

    def predict_action_chunk(stacked_obs, **chunk_kwargs):
        del stacked_obs, chunk_kwargs
        return torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    with pytest.raises(TypeError, match=r"predict_action_chunk must return"):
        policy._predict_action_from_chunk_queue(
            obs,
            predict_action_chunk=cast(Any, predict_action_chunk),
        )


def test_predict_action_from_chunk_queue_records_new_chunk_info():
    policy = DummyPolicy()
    obs = {"observation.state": torch.randn(2)}

    def predict_action_chunk(stacked_obs, **chunk_kwargs):
        del stacked_obs, chunk_kwargs
        actions = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        return actions, {"decode_steps": 2, "token_regime": "fixed_length"}

    _ = policy._predict_action_from_chunk_queue(
        obs,
        predict_action_chunk=predict_action_chunk,
    )

    assert policy.consume_inference_info() == {
        "decode_steps": 2,
        "token_regime": "fixed_length",
    }
    assert policy.consume_inference_info() is None


def test_from_pretrained_loads_weights_for_modules_created_by_set_io(tmp_path):
    io = _policy_io(dynamic_dim=3)
    policy = DynamicIoModulePolicy(dynamic_dim=3)
    policy.set_io(io)
    assert policy.dynamic is not None
    with torch.no_grad():
        policy.dynamic.weight.copy_(torch.tensor([[1.0, 2.0, 3.0]]))

    policy.save_pretrained(tmp_path, config={"dynamic_dim": 3})

    loaded = DynamicIoModulePolicy.from_pretrained(tmp_path)

    assert loaded.dynamic is not None
    assert torch.equal(loaded.dynamic.weight, torch.tensor([[1.0, 2.0, 3.0]]))


def test_save_pretrained_uses_explicit_state_dict(tmp_path):
    io = _policy_io(dynamic_dim=3)
    policy = DynamicIoModulePolicy(dynamic_dim=3)
    policy.set_io(io)
    assert policy.dynamic is not None
    with torch.no_grad():
        policy.dynamic.weight.zero_()

    policy.save_pretrained(
        tmp_path,
        config={"dynamic_dim": 3},
        state_dict={"dynamic.weight": torch.tensor([[4.0, 5.0, 6.0]])},
    )

    saved = load_file(tmp_path / "model.safetensors")
    assert torch.equal(saved["dynamic.weight"], torch.tensor([[4.0, 5.0, 6.0]]))
    assert torch.equal(policy.dynamic.weight, torch.zeros((1, 3)))


def test_save_pretrained_supports_sharded_safetensors(tmp_path):
    io = _policy_io(dynamic_dim=16)
    policy = ShardedIoModulePolicy(dynamic_dim=16)
    policy.set_io(io)
    assert policy.left is not None
    assert policy.right is not None
    with torch.no_grad():
        policy.left.weight.copy_(torch.arange(256, dtype=torch.float32).view(16, 16))
        policy.right.weight.copy_(
            torch.arange(256, 512, dtype=torch.float32).view(16, 16)
        )

    policy.save_pretrained(
        tmp_path,
        config={"dynamic_dim": 16},
        max_shard_size=512,
    )

    assert (tmp_path / "model.safetensors.index.json").exists()
    assert len(list(tmp_path.glob("model-*.safetensors"))) >= 2

    loaded = ShardedIoModulePolicy.from_pretrained(tmp_path)

    assert loaded.left is not None
    assert loaded.right is not None
    assert torch.equal(loaded.left.weight, policy.left.weight)
    assert torch.equal(loaded.right.weight, policy.right.weight)


def test_vlm_checkpoint_filter_supports_base_policy_parent() -> None:
    policy = VlmOnlyFilterPolicy()
    state = {"vlm.weight": torch.tensor([1.0]), "head.weight": torch.tensor([2.0])}

    assert policy.filter_checkpoint_export_state_dict(state) == state
    assert policy.filter_checkpoint_incompatible_keys(
        missing=("vlm.weight",),
        unexpected={"head.weight"},
    ) == (["vlm.weight"], ["head.weight"])


def test_save_pretrained_pushes_after_sidecars_are_written(tmp_path, monkeypatch):
    io = _policy_io(dynamic_dim=3)
    policy = DynamicIoModulePolicy(dynamic_dim=3)
    policy.set_io(io)
    calls = []

    class _FakeHfApi:
        def __init__(self, *, token=None):
            calls.append(("init", token))

        def create_repo(self, *, repo_id, private=None, exist_ok=True):
            calls.append(("create_repo", repo_id, private, exist_ok))
            return SimpleNamespace(repo_id=repo_id)

        def upload_folder(self, *, folder_path, **kwargs):
            assert Path(folder_path) == tmp_path
            assert (tmp_path / "policy_io.json").exists()
            assert (tmp_path / "norm_stats.pt").exists()
            calls.append(("upload_folder", kwargs))
            return "https://huggingface.co/org/repo/commit/abc"

    monkeypatch.setattr("praxis.core.checkpoint_artifacts.HfApi", _FakeHfApi)

    result = policy.save_pretrained(
        tmp_path,
        config={"dynamic_dim": 3},
        push_to_hub=True,
        repo_id="org/repo",
        commit_message="publish",
    )

    assert result == "https://huggingface.co/org/repo/commit/abc"
    assert calls == [
        ("init", None),
        ("create_repo", "org/repo", None, True),
        (
            "upload_folder",
            {
                "repo_id": "org/repo",
                "repo_type": "model",
                "commit_message": "publish",
                "revision": None,
                "create_pr": None,
                "allow_patterns": None,
                "ignore_patterns": None,
                "delete_patterns": None,
            },
        ),
    ]


def test_save_pretrained_rejects_unknown_push_kwargs(tmp_path):
    io = _policy_io(dynamic_dim=3)
    policy = DynamicIoModulePolicy(dynamic_dim=3)
    policy.set_io(io)

    try:
        policy.save_pretrained(
            tmp_path,
            push_to_hub=True,
            repo_id="org/repo",
            unknown_option=True,
        )
    except TypeError as exc:
        assert "unknown_option" in str(exc)
    else:
        raise AssertionError("save_pretrained must reject unknown push kwargs")


def test_resolve_pretrained_weight_file_rejects_incomplete_local_shards(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "left.weight": "model-00001-of-00002.safetensors",
                    "right.weight": "model-00002-of-00002.safetensors",
                }
            }
        )
        + "\n"
    )
    (tmp_path / "model-00001-of-00002.safetensors").write_text("left\n")

    try:
        resolve_pretrained_weight_file(tmp_path)
    except FileNotFoundError as exc:
        assert "incomplete" in str(exc)
    else:
        raise AssertionError("incomplete local sharded safetensors should fail early")


def test_resolve_pretrained_weight_file_rejects_stale_incomplete_index(tmp_path):
    (tmp_path / "model.safetensors").write_text("single-file\n")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "left.weight": "model-00001-of-00002.safetensors",
                    "right.weight": "model-00002-of-00002.safetensors",
                }
            }
        )
        + "\n"
    )
    (tmp_path / "model-00001-of-00002.safetensors").write_text("left\n")

    try:
        resolve_pretrained_weight_file(tmp_path)
    except FileNotFoundError as exc:
        assert "incomplete" in str(exc)
    else:
        raise AssertionError("stale incomplete safetensors index should fail early")
