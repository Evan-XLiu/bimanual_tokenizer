# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared VLM action-token schedules and attention masks."""

from __future__ import annotations

import pytest
import torch

from praxis.policies.core.vlm.schedule import (
    build_attention_mask,
    build_prediction_block_input_ids,
    resolve_block_schedule,
    validate_block_schedule,
)

DTYPE = torch.float32
MIN_VAL = torch.finfo(DTYPE).min


class TestBlockScheduleHelpers:
    def test_resolve_none_falls_back_to_ar(self):
        assert resolve_block_schedule(None, token_horizon=4) == (1, 1, 1, 1)

    def test_resolve_explicit_schedule_is_preserved(self):
        assert resolve_block_schedule((1, 1, 2), token_horizon=4) == (1, 1, 2)

    def test_validate_rejects_decreasing_schedule(self):
        with pytest.raises(ValueError, match="non-decreasing"):
            validate_block_schedule((1, 2, 1))

    def test_validate_rejects_wrong_sum(self):
        with pytest.raises(ValueError, match="must match token_horizon"):
            validate_block_schedule((1, 2), token_horizon=4)


class TestPredictionBlockTeacherForcing:
    def test_build_prediction_block_input_ids(self):
        targets = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]])
        out = build_prediction_block_input_ids(targets, (1, 1, 2, 4), mask_token_id=99)
        expected = torch.tensor([[99, 10, 11, 99, 12, 13, 99, 99]])
        assert out.equal(expected)

    def test_ar_none_schedule_matches_explicit_all_ones_teacher_forcing(self):
        token_horizon = 8
        targets = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]])
        implicit_schedule = resolve_block_schedule(None, token_horizon=token_horizon)
        explicit_schedule = (1,) * token_horizon

        implicit_inputs = build_prediction_block_input_ids(
            targets,
            implicit_schedule,
            mask_token_id=99,
        )
        explicit_inputs = build_prediction_block_input_ids(
            targets,
            explicit_schedule,
            mask_token_id=99,
        )
        assert implicit_schedule == explicit_schedule
        assert implicit_inputs.equal(explicit_inputs)


class TestActionTokenAttentionMask:
    def test_attention_mask_shape(self):
        attn = torch.ones(2, 6, dtype=torch.long)
        mask = build_attention_mask(
            attn,
            (1, 2),
            DTYPE,
            prefix_attention="bidirectional",
        )
        assert mask.shape == (2, 1, 6, 6)

    def test_prefix_and_block_attention_patterns(self):
        attn = torch.ones(1, 6, dtype=torch.long)
        mask = build_attention_mask(
            attn,
            (1, 2),
            DTYPE,
            prefix_attention="bidirectional",
        )[0, 0]

        assert mask[0, 2] == 0.0
        assert mask[2, 0] == 0.0
        assert mask[2, 3] == MIN_VAL
        assert mask[1, 5] == MIN_VAL
        assert mask[3, 0] == 0.0
        assert mask[3, 3] == 0.0
        assert mask[3, 4] == MIN_VAL
        assert mask[4, 0] == 0.0
        assert mask[4, 3] == 0.0
        assert mask[4, 5] == 0.0
        assert mask[5, 4] == 0.0

    def test_prefix_attention_mode_switches_only_prefix_prefix_edges(self):
        attn = torch.ones(1, 6, dtype=torch.long)
        schedule = (1, 1, 1)
        bidir = build_attention_mask(
            attn,
            schedule,
            DTYPE,
            prefix_attention="bidirectional",
        )[0, 0]
        causal = build_attention_mask(
            attn,
            schedule,
            DTYPE,
            prefix_attention="causal",
        )[0, 0]

        assert bidir[0, 1] == 0.0
        assert bidir[1, 0] == 0.0
        assert bidir[0, 2] == 0.0
        assert bidir[2, 0] == 0.0

        assert causal[0, 1] == MIN_VAL
        assert causal[1, 2] == MIN_VAL
        assert causal[0, 2] == MIN_VAL
        assert causal[1, 0] == 0.0
        assert causal[2, 0] == 0.0
        assert causal[2, 1] == 0.0

        for row in (3, 4, 5):
            for col in (0, 1, 2):
                assert bidir[row, col] == 0.0
                assert causal[row, col] == 0.0
        for row in (3, 4, 5):
            for col in (3, 4, 5):
                assert bidir[row, col] == causal[row, col]

    def test_left_pad_columns_are_blocked(self):
        attn = torch.tensor([[0, 1, 1, 1, 1, 1]])
        mask = build_attention_mask(
            attn,
            (1, 2),
            DTYPE,
            prefix_attention="bidirectional",
        )[0, 0]
        assert (mask[:, 0] == MIN_VAL).all()

    def test_prediction_position_mask_excludes_trailing_processor_eos_from_suffix(self):
        attn = torch.ones(1, 8, dtype=torch.long)
        prediction_positions = torch.tensor(
            [[False, False, False, True, True, True, True, False]]
        )

        mask = build_attention_mask(
            attn,
            (1, 1, 1, 1),
            DTYPE,
            prefix_attention="bidirectional",
            prediction_position_mask=prediction_positions,
        )[0, 0]

        assert mask[0, 2] == 0.0
        assert mask[0, 3] == MIN_VAL
        assert mask[0, 7] == MIN_VAL
        assert mask[3, 0] == 0.0
        assert mask[3, 3] == 0.0
        assert mask[3, 4] == MIN_VAL
        assert mask[3, 7] == MIN_VAL
        assert mask[6, 5] == 0.0
        assert mask[7, 6] == 0.0

    def test_prediction_position_mask_rejects_interleaved_template_tokens(self):
        attn = torch.ones(1, 8, dtype=torch.long)
        prediction_positions = torch.tensor(
            [[False, False, False, True, True, False, True, True]]
        )

        with pytest.raises(ValueError, match="interleaved positions"):
            build_attention_mask(
                attn,
                (1, 1, 1, 1),
                DTYPE,
                prefix_attention="bidirectional",
                prediction_position_mask=prediction_positions,
            )

    def test_ar_none_schedule_matches_explicit_all_ones_attention_mask(self):
        token_horizon = 4
        attn = torch.ones(1, 7, dtype=torch.long)
        implicit_schedule = resolve_block_schedule(None, token_horizon=token_horizon)
        explicit_schedule = (1,) * token_horizon

        implicit_mask = build_attention_mask(
            attn,
            implicit_schedule,
            DTYPE,
            prefix_attention="bidirectional",
        )
        explicit_mask = build_attention_mask(
            attn,
            explicit_schedule,
            DTYPE,
            prefix_attention="bidirectional",
        )
        torch.testing.assert_close(implicit_mask, explicit_mask)
