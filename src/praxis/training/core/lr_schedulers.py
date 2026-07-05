# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Learning rate schedulers."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_warmup_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Cosine annealing with linear warmup.

    LR ramps linearly from 0 → base_lr over warmup_steps,
    then decays via cosine to min_lr_ratio * base_lr at total_steps,
    then stays at min_lr_ratio * base_lr for all subsequent steps.
    """

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if step >= total_steps:
            return min_lr_ratio
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return LambdaLR(optimizer, lr_lambda)


def get_constant_warmup_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
) -> LambdaLR:
    """Constant LR with linear warmup.

    LR ramps linearly from 0 → base_lr over warmup_steps, then stays constant.
    """

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0

    return LambdaLR(optimizer, lr_lambda)
