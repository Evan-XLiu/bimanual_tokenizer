# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for BAR decode schedules and cached step masks."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from praxis.policies.bar.core.decode_schedule import (
    build_step_attention_mask,
    plan_generation_schedule,
    resolve_decode_keep_k,
)
from praxis.policies.bar.core.vlm.decoding import (
    build_vlm_prefill_inputs,
    run_blockwise_vlm_forward_step_result,
)
from praxis.policies.core.vlm.schedule import (
    build_attention_mask,
    overlay_bidirectional_token_mask,
    resolve_block_schedule,
)

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


def test_molmo_prefill_matches_training_image_attention_with_left_padding():
    class FakeVlm:
        def prepare_inputs_for_generation(self, **kwargs):
            return kwargs

    prompt_ids = torch.tensor([[0, 10, 11, 12, 13]])
    image_ids = torch.tensor([[0, 0, 1, 1, 0]], dtype=torch.bool)
    attention = torch.tensor([[0, 1, 1, 1, 1, 1, 1]])
    prefill = build_vlm_prefill_inputs(
        vlm=FakeVlm(),
        base_vlm_inputs={"token_type_ids": image_ids},
        prompt_input_ids=prompt_ids,
        running_attention_mask_2d=attention,
        pred_block_vlm_ids=torch.tensor([[20, 21]]),
        block_schedule=(2,),
        mask_dtype=DTYPE,
        prefix_attention="causal",
        extra_prepare_keys=("token_type_ids",),
        image_token_type_ids=image_ids,
    )
    scheduled = build_attention_mask(attention, (2,), DTYPE, prefix_attention="causal")
    training_mask = overlay_bidirectional_token_mask(
        scheduled,
        torch.tensor([[0, 0, 1, 1, 0, 0, 0]], dtype=torch.bool),
    )
    assert torch.equal(prefill["attention_mask"], training_mask)
    mask = prefill["attention_mask"][0, 0]
    assert mask[2, 3] == 0.0  # An earlier image token sees a later one.
    assert mask[1, 3] == MIN_VAL  # Text remains causal.
    assert mask[2, 5] == MIN_VAL  # Images cannot see action tokens.
    assert mask[5, 3] == 0.0  # Action tokens can see all images.
    assert (mask[:, 0] == MIN_VAL).all()  # Left padding remains masked.


def test_molmo_policy_activates_image_overlay_only_for_its_prefill():
    class FakeVlm:
        def prepare_inputs_for_generation(self, **kwargs):
            return kwargs

        def __call__(self, **kwargs):
            self.forward_mask = kwargs["attention_mask"]
            return SimpleNamespace(logits=torch.zeros(1, 1, 3), past_key_values=None)

    image_ids = torch.tensor([[0, 1, 1, 0]])
    base_inputs = {"token_type_ids": image_ids}
    for enabled in (False, True):
        vlm = FakeVlm()
        policy = SimpleNamespace(
            vlm=vlm,
            _PREFILL_BIDIRECTIONAL_IMAGE_TOKENS=enabled,
            _PREFILL_EXTRA_KEYS=("token_type_ids",),
            _require_prefix_attention=lambda: "causal",
            _autocast_context=nullcontext,
        )
        run_blockwise_vlm_forward_step_result(
            policy,
            base_vlm_inputs=base_inputs,
            running_input_ids=torch.tensor([[10, 11, 12, 13]]),
            running_attention_mask_2d=torch.ones(1, 4, dtype=torch.long),
            pred_block_vlm_ids=torch.tensor([[20]]),
            block_schedule=(1,),
            mask_dtype=DTYPE,
            past_key_values=None,
        )
        assert bool(vlm.forward_mask[0, 0, 1, 2] == 0.0) is enabled
