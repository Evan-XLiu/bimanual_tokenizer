# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Episode-level train/validation splitting helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def split_train_val_episodes(
    all_episodes: Sequence[int],
    *,
    val_ratio: float,
    split_seed: int | None = 0,
) -> tuple[list[int], list[int]]:
    """Return a deterministic random episode split.

    The validation membership is sampled with an exact-count boolean mask. The
    returned episode lists preserve the input order so downstream LeRobot loading
    keeps stable episode traversal within each split.
    """

    episodes = [int(ep) for ep in all_episodes]
    if not math.isfinite(float(val_ratio)):
        raise ValueError(f"val_ratio must be finite, got {val_ratio}.")
    if val_ratio <= 0 or len(episodes) <= 1:
        return episodes, []

    n_episodes = len(episodes)
    n_val = max(1, round(n_episodes * float(val_ratio)))
    n_val = min(n_val, n_episodes - 1)

    rng = np.random.default_rng(0 if split_seed is None else int(split_seed))
    val_positions = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask = np.zeros(n_episodes, dtype=bool)
    val_mask[val_positions] = True

    train_episodes = [episodes[i] for i in np.flatnonzero(~val_mask)]
    val_episodes = [episodes[i] for i in np.flatnonzero(val_mask)]
    return train_episodes, val_episodes


def split_validation_episodes(
    all_episodes: Sequence[int],
    *,
    val_ratio: float,
    split_seed: int | None = 0,
) -> list[int]:
    """Return only the validation side of `split_train_val_episodes`."""

    _, val_episodes = split_train_val_episodes(
        all_episodes,
        val_ratio=val_ratio,
        split_seed=split_seed,
    )
    return val_episodes
