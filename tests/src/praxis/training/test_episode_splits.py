# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from praxis.training.core.dataset_filtering import (
    filter_episode_pool,
    load_action_stats_payload,
    load_excluded_episode_ids,
)
from praxis.training.core.episode_splits import split_train_val_episodes
from praxis.training.core.trainer import DefaultTrainer


def test_split_train_val_episodes_samples_exact_count_without_tail_bias() -> None:
    episodes = list(range(100))

    train_episodes, val_episodes = split_train_val_episodes(
        episodes,
        val_ratio=0.1,
        split_seed=123,
    )

    assert len(train_episodes) == 90
    assert len(val_episodes) == 10
    assert val_episodes != episodes[-10:]
    assert sorted(train_episodes + val_episodes) == episodes


def test_split_train_val_episodes_is_deterministic_by_seed() -> None:
    episodes = list(range(100))

    _, val_a = split_train_val_episodes(episodes, val_ratio=0.2, split_seed=5)
    _, val_b = split_train_val_episodes(episodes, val_ratio=0.2, split_seed=5)
    _, val_c = split_train_val_episodes(episodes, val_ratio=0.2, split_seed=6)

    assert val_a == val_b
    assert val_a != val_c


def test_split_train_val_episodes_keeps_at_least_one_training_episode() -> None:
    train_episodes, val_episodes = split_train_val_episodes(
        [10, 11, 12],
        val_ratio=1.0,
        split_seed=0,
    )

    assert len(train_episodes) == 1
    assert len(val_episodes) == 2
    assert sorted(train_episodes + val_episodes) == [10, 11, 12]


def test_split_train_val_episodes_disables_validation_for_zero_ratio() -> None:
    assert split_train_val_episodes([0, 1, 2], val_ratio=0.0) == ([0, 1, 2], [])


def test_split_train_val_episodes_rejects_nonfinite_ratio() -> None:
    with pytest.raises(ValueError, match="val_ratio must be finite"):
        split_train_val_episodes([0, 1, 2], val_ratio=float("nan"))


def test_dataset_filtering_loads_and_applies_exclusions(tmp_path: Path) -> None:
    exclusion_file = tmp_path / "excluded.json"
    exclusion_file.write_text(json.dumps([1, 3]))

    excluded = load_excluded_episode_ids(exclusion_file)

    assert excluded == {1, 3}
    assert filter_episode_pool([0, 1, 2, 3, 4], excluded) == [0, 2, 4]


@pytest.mark.parametrize("payload", [[1, 1], [True], [-1], [1.0], {"episode": 1}])
def test_dataset_filtering_rejects_invalid_exclusion_lists(
    tmp_path: Path, payload: object
) -> None:
    exclusion_file = tmp_path / "excluded.json"
    exclusion_file.write_text(json.dumps(payload))

    with pytest.raises(ValueError):
        load_excluded_episode_ids(exclusion_file)


def test_dataset_filtering_rejects_unknown_or_all_exclusions() -> None:
    with pytest.raises(ValueError, match="outside the configured dataset"):
        filter_episode_pool([0, 1], {2})
    with pytest.raises(ValueError, match="removed every configured episode"):
        filter_episode_pool([0, 1], {0, 1})


def test_load_action_stats_payload_requires_action_block(tmp_path: Path) -> None:
    stats_file = tmp_path / "stats.json"
    stats_file.write_text(json.dumps({"action": {"min": [0.0], "max": [1.0]}}))

    assert load_action_stats_payload(stats_file) == {
        "action": {"min": [0.0], "max": [1.0]}
    }

    stats_file.write_text(json.dumps({"observation.state": {}}))
    with pytest.raises(ValueError, match="action.*statistics block"):
        load_action_stats_payload(stats_file)


def test_default_trainer_split_uses_trainer_split_seed() -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    trainer.cfg = SimpleNamespace(seed=999)
    trainer.tc = SimpleNamespace(val_ratio=0.25, split_seed=7)

    assert trainer._split_episodes(list(range(20))) == split_train_val_episodes(
        list(range(20)),
        val_ratio=0.25,
        split_seed=7,
    )


def test_default_trainer_split_falls_back_to_global_seed() -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    trainer.cfg = SimpleNamespace(seed=17)
    trainer.tc = SimpleNamespace(val_ratio=0.25)

    assert trainer._split_episodes(list(range(20))) == split_train_val_episodes(
        list(range(20)),
        val_ratio=0.25,
        split_seed=17,
    )
