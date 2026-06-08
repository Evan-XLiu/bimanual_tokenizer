# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Server/runtime helpers for Praxis policy serving."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from praxis_remote.serialization import ObservationValue


def collate_observations_to_torch_batch(
    observations: Sequence[Mapping[str, ObservationValue]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Collate per-example observations into the batch dict expected by policies."""
    if len(observations) == 0:
        raise ValueError("PredictRequest must contain at least one observation.")

    first_keys = list(observations[0].keys())
    expected_keys = set(first_keys)
    for index, observation in enumerate(observations[1:], start=1):
        actual_keys = set(observation.keys())
        if actual_keys != expected_keys:
            raise ValueError(
                "All batched observations must share the same keys. "
                f"Expected {sorted(expected_keys)}, got {sorted(actual_keys)} for "
                f"observation index {index}."
            )

    batch: dict[str, Any] = {}
    for key in first_keys:
        first_value = observations[0][key]
        if isinstance(first_value, str):
            values = [observation[key] for observation in observations]
            if not all(isinstance(value, str) for value in values):
                raise TypeError(
                    f"Observation key {key!r} mixes string and tensor values across the batch."
                )
            batch[key] = [str(value) for value in values]
            continue

        arrays = [np.asarray(observation[key]) for observation in observations]
        if any(isinstance(observation[key], str) for observation in observations):
            raise TypeError(
                f"Observation key {key!r} mixes tensor and string values across the batch."
            )
        try:
            stacked = np.stack(arrays, axis=0)
        except ValueError as exc:
            raise ValueError(
                f"Observation key {key!r} has incompatible tensor shapes for batching."
            ) from exc
        batch[key] = torch.from_numpy(stacked).to(device)

    return batch
