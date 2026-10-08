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


def get_group_cosine_warmup_scheduler(
    optimizer: Optimizer,
    *,
    warmup_steps: list[int],
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Cosine schedule with an independent warmup for each optimizer group."""
    if len(warmup_steps) != len(optimizer.param_groups):
        raise ValueError("warmup_steps must match optimizer parameter groups.")

    def lr_lambda_for_group(group_index: int):
        group_warmup = max(0, int(warmup_steps[group_index]))

        def lr_lambda(step: int) -> float:
            if step < group_warmup:
                return step / max(1, group_warmup)
            if step >= total_steps:
                return min_lr_ratio
            progress = (step - group_warmup) / max(1, total_steps - group_warmup)
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )

        return lr_lambda

    return LambdaLR(
        optimizer,
        [lr_lambda_for_group(i) for i in range(len(optimizer.param_groups))],
    )


def get_group_cosine_warmup_scheduler_with_floor(
    optimizer: Optimizer,
    *,
    warmup_steps: list[int],
    min_lr_by_group: list[float] | None = None,
    total_steps: int,
    min_lr_by_family: dict[str, float] | None = None,
) -> LambdaLR:
    """Group cosine warmup with absolute final LR floors."""
    if len(warmup_steps) != len(optimizer.param_groups):
        raise ValueError("warmup_steps must match optimizer parameter groups.")
    if min_lr_by_group is not None and len(min_lr_by_group) != len(
        optimizer.param_groups
    ):
        raise ValueError("min_lr_by_group must match optimizer parameter groups.")
    min_lr_by_family = min_lr_by_family or {}

    def make_lambda(index: int):
        group = optimizer.param_groups[index]
        warmup = max(0, int(warmup_steps[index]))
        if min_lr_by_group is not None:
            floor = float(min_lr_by_group[index])
        else:
            floor = float(min_lr_by_family.get(str(group.get("family", "")), 0.0))
        base = float(group["lr"])
        ratio = min(1.0, max(0.0, floor / base)) if base > 0 else 0.0

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            if step >= total_steps:
                return ratio
            progress = (step - warmup) / max(1, total_steps - warmup)
            return ratio + (1.0 - ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

        return lr_lambda

    return LambdaLR(optimizer, [make_lambda(i) for i in range(len(optimizer.param_groups))])


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
