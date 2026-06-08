# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Small DTensor predicates shared by distributed training helpers."""

from __future__ import annotations

from typing import Any, cast

import torch


def is_dtensor(tensor: torch.Tensor) -> bool:
    """Return whether ``tensor`` looks like a torch DTensor."""
    return all(
        hasattr(tensor, name) for name in ("to_local", "device_mesh", "placements")
    )


def dtensor_has_sharded_placement(tensor: torch.Tensor) -> bool:
    """Return whether a DTensor has any sharded placement."""
    placements = getattr(tensor, "placements", ())
    return any(type(placement).__name__ == "Shard" for placement in placements)


def dtensor_to_local(tensor: torch.Tensor) -> torch.Tensor:
    """Return the local shard for DTensors, otherwise return the tensor itself."""
    if is_dtensor(tensor):
        return cast(torch.Tensor, cast(Any, tensor).to_local())
    return tensor
