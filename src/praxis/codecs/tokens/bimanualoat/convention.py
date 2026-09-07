# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Action convention helpers for paired bimanual actions."""

from __future__ import annotations

import torch
from torch import Tensor


def apply_bimanual_convention(actions: Tensor, right_arm_sign: Tensor) -> Tensor:
    """Map between physical and canonical ``[left, right]`` conventions.

    The sign-only mapping is self-inverse, so this function serves both
    canonicalization and decanonicalization.
    """

    arm_dim = int(right_arm_sign.numel())
    if actions.shape[-1] != 2 * arm_dim:
        raise ValueError(
            f"expected bimanual action dimension {2 * arm_dim}, "
            f"got {actions.shape[-1]}."
        )
    left, right = actions.split(arm_dim, dim=-1)
    sign = right_arm_sign.to(device=actions.device, dtype=actions.dtype)
    return torch.cat((left, right * sign), dim=-1)
