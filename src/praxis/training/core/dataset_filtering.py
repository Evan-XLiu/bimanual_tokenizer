# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Shared dataset filtering inputs for codec and policy training."""

from __future__ import annotations

import json
from collections.abc import Sequence
from numbers import Integral
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> tuple[Path, Any]:
    resolved = Path(str(path)).expanduser()
    try:
        return resolved, json.loads(resolved.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset filter file does not exist: {resolved}"
        ) from None
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Dataset filter file is not valid JSON: {resolved}: {exc}"
        ) from exc


def load_excluded_episode_ids(path: str | Path | None) -> set[int]:
    """Load a strict JSON list of non-negative, unique episode ids."""
    if path is None:
        return set()

    resolved, payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"{resolved} must contain a JSON list of episode ids.")

    episodes: list[int] = []
    for value in payload:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(
                f"{resolved} episode ids must be non-negative integers, got {value!r}."
            )
        episode = int(value)
        if episode < 0:
            raise ValueError(
                f"{resolved} episode ids must be non-negative integers, got {episode}."
            )
        episodes.append(episode)
    if len(set(episodes)) != len(episodes):
        raise ValueError(f"{resolved} must not contain duplicate episode ids.")
    return set(episodes)


def filter_episode_pool(
    configured_episodes: Sequence[int],
    excluded_episodes: set[int],
) -> list[int]:
    """Validate exclusions and remove them before train/validation splitting."""
    episodes = [int(episode) for episode in configured_episodes]
    unknown = excluded_episodes.difference(episodes)
    if unknown:
        raise ValueError(
            "Excluded episode ids are outside the configured dataset: "
            f"{sorted(unknown)}"
        )
    retained = [episode for episode in episodes if episode not in excluded_episodes]
    if not retained:
        raise ValueError("Episode filtering removed every configured episode.")
    return retained


def load_action_stats_payload(path: str | Path | None) -> dict[str, Any] | None:
    """Load a filtered action-statistics override."""
    if path is None:
        return None

    resolved, payload = _read_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("action"), dict):
        raise ValueError(f"{resolved} must contain an 'action' statistics block.")
    return payload
