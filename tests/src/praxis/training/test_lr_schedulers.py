# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for praxis.training.core.lr_schedulers."""

from __future__ import annotations

import pytest
import torch

from praxis.training.core.lr_schedulers import (
    get_constant_warmup_scheduler,
    get_cosine_warmup_scheduler,
)


def _make_optimizer(lr: float = 1e-3) -> torch.optim.Optimizer:
    model = torch.nn.Linear(4, 4)
    return torch.optim.SGD(model.parameters(), lr=lr)


def _step_optimizer_and_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    n_steps: int,
) -> None:
    for _ in range(n_steps):
        optimizer.step()
        scheduler.step()


class TestCosineWarmupScheduler:
    def test_starts_at_zero(self):
        opt = _make_optimizer(lr=1.0)
        get_cosine_warmup_scheduler(opt, warmup_steps=100, total_steps=1000)
        assert opt.param_groups[0]["lr"] == pytest.approx(0.0)

    def test_warmup_reaches_base_lr(self):
        opt = _make_optimizer(lr=1.0)
        sched = get_cosine_warmup_scheduler(opt, warmup_steps=100, total_steps=1000)
        _step_optimizer_and_scheduler(opt, sched, n_steps=100)
        assert opt.param_groups[0]["lr"] == pytest.approx(1.0)

    def test_decays_after_warmup(self):
        opt = _make_optimizer(lr=1.0)
        sched = get_cosine_warmup_scheduler(opt, warmup_steps=10, total_steps=100)
        _step_optimizer_and_scheduler(opt, sched, n_steps=10)
        lr_at_warmup_end = opt.param_groups[0]["lr"]
        _step_optimizer_and_scheduler(opt, sched, n_steps=40)
        lr_mid = opt.param_groups[0]["lr"]
        assert lr_mid < lr_at_warmup_end

    def test_clamps_at_min_lr_after_total_steps(self):
        opt = _make_optimizer(lr=1.0)
        min_lr_ratio = 0.1
        sched = get_cosine_warmup_scheduler(
            opt, warmup_steps=10, total_steps=100, min_lr_ratio=min_lr_ratio
        )
        _step_optimizer_and_scheduler(opt, sched, n_steps=100)
        lr_at_total = opt.param_groups[0]["lr"]
        assert lr_at_total == pytest.approx(min_lr_ratio, abs=1e-6)

        # Step beyond total_steps — should stay clamped
        _step_optimizer_and_scheduler(opt, sched, n_steps=50)
        lr_beyond = opt.param_groups[0]["lr"]
        assert lr_beyond == pytest.approx(min_lr_ratio, abs=1e-6)

    def test_does_not_wrap_back_up(self):
        """Regression test: cosine used to wrap back up after total_steps."""
        opt = _make_optimizer(lr=1.0)
        sched = get_cosine_warmup_scheduler(
            opt, warmup_steps=10, total_steps=100, min_lr_ratio=0.0
        )
        lrs = []
        for _ in range(200):
            opt.step()
            sched.step()
            lrs.append(opt.param_groups[0]["lr"])
        # After step 100, LR should never increase
        for i in range(100, len(lrs)):
            assert lrs[i] <= lrs[99] + 1e-8, (
                f"LR increased at step {i + 1}: {lrs[i]} > {lrs[99]}"
            )


class TestConstantWarmupScheduler:
    def test_starts_at_zero(self):
        opt = _make_optimizer(lr=1.0)
        get_constant_warmup_scheduler(opt, warmup_steps=100)
        assert opt.param_groups[0]["lr"] == pytest.approx(0.0)

    def test_warmup_reaches_base_lr(self):
        opt = _make_optimizer(lr=1.0)
        sched = get_constant_warmup_scheduler(opt, warmup_steps=100)
        _step_optimizer_and_scheduler(opt, sched, n_steps=100)
        assert opt.param_groups[0]["lr"] == pytest.approx(1.0)

    def test_stays_constant_after_warmup(self):
        opt = _make_optimizer(lr=1.0)
        sched = get_constant_warmup_scheduler(opt, warmup_steps=10)
        _step_optimizer_and_scheduler(opt, sched, n_steps=10)
        lr1 = opt.param_groups[0]["lr"]
        _step_optimizer_and_scheduler(opt, sched, n_steps=100)
        lr2 = opt.param_groups[0]["lr"]
        assert lr1 == pytest.approx(lr2)
        assert lr1 == pytest.approx(1.0)
