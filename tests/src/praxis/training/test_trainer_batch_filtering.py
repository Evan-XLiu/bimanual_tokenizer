# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for trainer batch tensor-key filtering and device transfer."""

from __future__ import annotations

import pytest
import torch

from praxis.core.types import FeatureSpec, FeatureType, NormMode, PolicyIO
from praxis.training.codecs_trainer import CodecsTrainer
from praxis.training.core.trainer import (
    DefaultTrainer,
    resolve_per_rank_batch_size,
    resolve_sampler_frame_drops,
    resolve_val_batch_size,
)
from praxis.training.policy_trainer import PolicyTrainer


class _SelectiveTrainer(DefaultTrainer):
    def _required_tensor_keys(self, phase: str, batch: dict) -> set[str]:
        del phase, batch
        return {"action"}


class _ActionKeyTrainer(DefaultTrainer):
    action_key = "act"


def _policy_io(
    action_key: str = "action",
    *,
    n_obs_steps: int = 1,
    n_action_steps: int = 16,
) -> PolicyIO:
    return PolicyIO(
        obs_features=[
            FeatureSpec(
                key="observation.state",
                type=FeatureType.STATE,
                shape=(7,),
                norm_mode=NormMode.MIN_MAX,
            ),
            FeatureSpec(
                key="observation.image",
                type=FeatureType.IMAGE,
                shape=(3, 96, 96),
                norm_mode=NormMode.MEAN_STD,
            ),
        ],
        action_feature=FeatureSpec(
            key=action_key,
            type=FeatureType.ACTION,
            shape=(7,),
            norm_mode=NormMode.MIN_MAX,
        ),
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
    )


def test_default_required_tensor_keys_selects_all_tensors() -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    batch = {
        "action": torch.randn(2, 3, 4),
        "task": "stack blocks",
        "episode_index": torch.tensor([1, 2]),
    }
    assert trainer._required_tensor_keys("train", batch) == {"action", "episode_index"}


def test_move_batch_tensors_to_device_moves_only_required_keys() -> None:
    trainer = _SelectiveTrainer.__new__(_SelectiveTrainer)
    trainer.device = torch.device("meta")
    batch = {
        "action": torch.randn(2, 3, 4),
        "episode_index": torch.tensor([1, 2]),
        "task": "stack blocks",
    }
    moved = trainer._move_batch_tensors_to_device(batch, phase="train")
    assert moved["action"].device.type == "meta"
    assert moved["episode_index"].device.type == "cpu"
    assert moved["task"] == "stack blocks"


def test_batch_size_from_batch_uses_overridable_action_key() -> None:
    trainer = _ActionKeyTrainer.__new__(_ActionKeyTrainer)
    batch = {"act": torch.randn(5, 16, 7)}
    assert trainer._batch_size_from_batch(batch) == 5


def test_move_batch_tensors_to_device_raises_when_required_key_missing() -> None:
    trainer = _SelectiveTrainer.__new__(_SelectiveTrainer)
    trainer.device = torch.device("cpu")
    batch = {"episode_index": torch.tensor([1, 2])}
    try:
        trainer._move_batch_tensors_to_device(batch, phase="train")
    except KeyError as exc:
        assert "action" in str(exc)
    else:
        raise AssertionError("Expected KeyError when required tensor key is missing.")


def test_move_batch_tensors_to_device_raises_when_required_key_not_tensor() -> None:
    trainer = _SelectiveTrainer.__new__(_SelectiveTrainer)
    trainer.device = torch.device("cpu")
    batch = {"action": [1, 2, 3]}
    try:
        trainer._move_batch_tensors_to_device(batch, phase="train")
    except TypeError as exc:
        assert "must be a tensor" in str(exc)
    else:
        raise AssertionError("Expected TypeError when required key is not a tensor.")


def test_codec_required_tensor_keys_are_action_only() -> None:
    trainer = CodecsTrainer.__new__(CodecsTrainer)
    trainer.action_key = "action"
    batch = {
        "action": torch.randn(2, 32, 7),
        "action_is_pad": torch.zeros(2, 32, dtype=torch.bool),
        "episode_index": torch.tensor([1, 2]),
    }
    assert trainer._required_tensor_keys("train", batch) == {"action", "action_is_pad"}


def test_policy_required_tensor_keys_include_obs_action_and_pad_masks() -> None:
    trainer = PolicyTrainer.__new__(PolicyTrainer)
    io = _policy_io(action_key="act")
    io.language_key = "task"
    trainer.io = io
    assert io.action_feature is not None
    trainer.action_key = io.action_feature.key
    batch = {
        "observation.state": torch.randn(2, 1, 7),
        "observation.image": torch.randn(2, 1, 3, 96, 96),
        "observation.state_is_pad": torch.zeros(2, 1, dtype=torch.bool),
        "act": torch.randn(2, 16, 7),
        "act_is_pad": torch.zeros(2, 16, dtype=torch.bool),
        "episode_index": torch.tensor([1, 2]),
        "task": "stack blocks",
    }
    required = trainer._required_tensor_keys("train", batch)
    assert required == {
        "observation.state",
        "observation.image",
        "observation.state_is_pad",
        "act",
        "act_is_pad",
    }


def test_policy_required_tensor_keys_include_tensor_language() -> None:
    trainer = PolicyTrainer.__new__(PolicyTrainer)
    io = _policy_io()
    io.language_key = "task"
    trainer.io = io
    assert io.action_feature is not None
    trainer.action_key = io.action_feature.key
    batch = {
        "observation.state": torch.randn(2, 1, 7),
        "observation.image": torch.randn(2, 1, 3, 96, 96),
        "action": torch.randn(2, 16, 7),
        "task": torch.ones(2, 8, dtype=torch.long),
    }
    required = trainer._required_tensor_keys("train", batch)
    assert "task" in required


def test_resolve_sampler_frame_drops_uses_io_minimums() -> None:
    io = _policy_io(n_obs_steps=3, n_action_steps=32)
    drop_n_first_frames, drop_n_last_frames = resolve_sampler_frame_drops(io)
    assert drop_n_first_frames == 2
    assert drop_n_last_frames == 31


def test_resolve_sampler_frame_drops_respects_larger_configured_values() -> None:
    io = _policy_io(n_obs_steps=2, n_action_steps=16)
    drop_n_first_frames, drop_n_last_frames = resolve_sampler_frame_drops(
        io,
        configured_drop_n_first_frames=5,
        configured_drop_n_last_frames=20,
    )
    assert drop_n_first_frames == 5
    assert drop_n_last_frames == 20


def test_resolve_sampler_frame_drops_rejects_negative_values() -> None:
    io = _policy_io(n_obs_steps=1, n_action_steps=1)
    try:
        resolve_sampler_frame_drops(io, configured_drop_n_first_frames=-1)
    except ValueError as exc:
        assert "drop_n_first_frames" in str(exc)
    else:
        raise AssertionError("Expected ValueError for negative drop_n_first_frames.")


def test_resolve_val_batch_size_falls_back_to_train_batch_size() -> None:
    assert resolve_val_batch_size(64, None) == 64


def test_resolve_val_batch_size_uses_override_when_provided() -> None:
    assert resolve_val_batch_size(64, 16) == 16


def test_resolve_val_batch_size_rejects_non_positive_values() -> None:
    try:
        resolve_val_batch_size(64, 0)
    except ValueError as exc:
        assert "Validation batch size must be > 0" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-positive val batch size.")


def test_resolve_per_rank_batch_size_uses_configured_per_rank_batch() -> None:
    assert (
        resolve_per_rank_batch_size(
            configured_batch_size=16,
            global_batch_size=None,
            world_size=2,
            gradient_accumulation_steps=4,
        )
        == 16
    )


def test_resolve_per_rank_batch_size_from_global_batch() -> None:
    assert (
        resolve_per_rank_batch_size(
            configured_batch_size=999,
            global_batch_size=16,
            world_size=2,
            gradient_accumulation_steps=1,
        )
        == 8
    )


def test_resolve_per_rank_batch_size_accounts_for_gradient_accumulation() -> None:
    assert (
        resolve_per_rank_batch_size(
            configured_batch_size=999,
            global_batch_size=32,
            world_size=2,
            gradient_accumulation_steps=4,
        )
        == 4
    )


def test_resolve_per_rank_batch_size_rejects_non_divisible_global_batch() -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        resolve_per_rank_batch_size(
            configured_batch_size=999,
            global_batch_size=15,
            world_size=2,
            gradient_accumulation_steps=1,
        )
