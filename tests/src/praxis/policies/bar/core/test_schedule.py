"""Tests for BAR decode schedules and cached step masks."""

from __future__ import annotations

import torch

from praxis.policies.bar.core.decode_schedule import (
    build_step_attention_mask,
    plan_generation_schedule,
    resolve_decode_keep_k,
)
from praxis.policies.core.vlm.schedule import resolve_block_schedule

DTYPE = torch.float32
MIN_VAL = torch.finfo(DTYPE).min


def test_plan_generation_schedule_truncates_to_cover_keep_k():
    assert plan_generation_schedule((1, 1, 2, 4), keep_k=3) == (1, 1, 2)


def test_resolve_decode_keep_k_defaults_to_horizon():
    assert resolve_decode_keep_k(None, token_horizon=4) == 4


def test_ar_none_schedule_matches_explicit_all_ones_generation_steps():
    token_horizon = 8
    implicit_schedule = resolve_block_schedule(None, token_horizon=token_horizon)
    explicit_schedule = (1,) * token_horizon

    for keep_k in (1, 3, 5, 8):
        assert plan_generation_schedule(
            implicit_schedule,
            keep_k=keep_k,
        ) == plan_generation_schedule(explicit_schedule, keep_k=keep_k)


def test_step_attention_mask_returns_only_latest_block_rows():
    attn = torch.ones(1, 6, dtype=torch.long)
    step_mask = build_step_attention_mask(attn, (1, 2), DTYPE)
    assert step_mask.shape == (1, 1, 2, 6)
    assert step_mask.eq(0.0).all()


def test_step_attention_mask_only_exposes_prefix_and_previous_blocks():
    attn = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1]])
    step_mask = build_step_attention_mask(attn, (1, 1, 2), DTYPE)[0, 0]

    assert (step_mask[:, 0] == MIN_VAL).all()
    assert step_mask[0, 1] == 0.0
    assert step_mask[1, 3] == 0.0
    assert step_mask[0, 4] == 0.0
    assert step_mask[1, 5] == 0.0
    assert step_mask[0, 7] == 0.0
    assert step_mask[1, 6] == 0.0
