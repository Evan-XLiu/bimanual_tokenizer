from __future__ import annotations

from types import SimpleNamespace

import pytest

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
