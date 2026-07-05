# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from praxis.codecs.latents.oat.schedule import (
    create_register_schedule_mask,
    pow2_register_schedule,
    register_schedule_prefix_boundaries,
    unit_register_schedule,
)


def test_unit_schedule_mask_is_causal() -> None:
    schedule = unit_register_schedule(4)
    assert schedule == (1, 1, 1, 1)

    mask = create_register_schedule_mask(schedule, "cpu")

    assert torch.equal(mask, torch.tril(torch.ones((4, 4), dtype=torch.bool)))


def test_pow2_schedule_and_boundaries() -> None:
    schedule = pow2_register_schedule(16)

    assert schedule == (1, 1, 2, 4, 8)
    assert register_schedule_prefix_boundaries(schedule) == (1, 2, 4, 8, 16)


def test_pow2_schedule_requires_power_of_two_horizon() -> None:
    with pytest.raises(ValueError, match="power of 2"):
        pow2_register_schedule(6)


def test_pow2_schedule_mask_blocks_same_level_off_diagonal() -> None:
    mask = create_register_schedule_mask(pow2_register_schedule(8), "cpu")

    assert mask.shape == (8, 8)
    assert bool(mask[2, 2])
    assert bool(mask[7, 7])

    assert not bool(mask[2, 3])
    assert not bool(mask[3, 2])

    assert bool(mask[4, 2])
    assert not bool(mask[2, 4])
    assert not bool(mask[4, 7])
    assert not bool(mask[7, 4])
