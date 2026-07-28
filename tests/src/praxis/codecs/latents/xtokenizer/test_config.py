# SPDX-FileCopyrightText: 2026 Xiaoshen Han
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for X-Tokenizer codec configuration."""

from __future__ import annotations

import pytest

from praxis.codecs.latents.xtokenizer import XTokenizerModelConfig

REORDERED_LAYOUT = {
    "convention": "absolute",
    "dof_dims": {
        "follow_left_gripper": 1,
        "follow_left_ee_cartesian_pos": 3,
        "follow_left_ee_rotation_6D": 6,
    },
    "dof_order": [
        "follow_left_gripper",
        "follow_left_ee_cartesian_pos",
        "follow_left_ee_rotation_6D",
    ],
}

DUAL_ARM_LAYOUT = {
    "convention": "absolute",
    "dof_dims": {
        "follow_left_ee_cartesian_pos": 3,
        "follow_left_ee_rotation_6D": 6,
        "follow_left_gripper": 1,
        "follow_right_ee_cartesian_pos": 3,
        "follow_right_ee_rotation_6D": 6,
        "follow_right_gripper": 1,
    },
    "dof_order": [
        "follow_left_ee_cartesian_pos",
        "follow_left_ee_rotation_6D",
        "follow_left_gripper",
        "follow_right_ee_cartesian_pos",
        "follow_right_ee_rotation_6D",
        "follow_right_gripper",
    ],
}


def test_xtokenizer_config_defaults() -> None:
    cfg = XTokenizerModelConfig.from_dict({"action_dim": 10})

    assert cfg.name == "xtokenizer"
    assert cfg.kind == "latents"
    assert cfg.action_horizon == 32
    assert cfg.max_action_horizon == 64
    assert cfg.compression_ratio == 4
    assert cfg.latent_dim == 1024
    assert cfg.encoder_num_heads == 8
    assert cfg.encoder_depth == 12
    assert cfg.decoder_num_heads == 8
    assert cfg.decoder_depth == 4
    assert cfg.mam_num_heads == 4
    assert cfg.mam_depth == 2
    assert cfg.prediction_num_heads == 4
    assert cfg.prediction_depth == 2
    assert cfg.state_dropout_probability == pytest.approx(0.2)
    assert cfg.embodiment_dropout_probability == pytest.approx(0.2)
    assert cfg.embodiment_none_probability == pytest.approx(0.1)
    assert cfg.codebook_size == 2048
    assert cfg.num_quantizers == 4
    assert cfg.kmeans_init is True
    assert cfg.kmeans_iters == 100
    assert cfg.embodiment_vocab_size == 1024
    assert cfg.vl_feature_dim == 3584
    assert cfg.reconstruction_weight == pytest.approx(1.0)
    assert cfg.commitment_weight == pytest.approx(0.25)
    assert cfg.rotation_weight == pytest.approx(0.2)
    assert cfg.dct_weight == pytest.approx(0.5)
    assert cfg.smoothness_weight == pytest.approx(0.3)
    assert cfg.mam_weight == pytest.approx(0.1)
    assert cfg.alignment_weight == pytest.approx(0.5)
    assert cfg.prediction_weight == pytest.approx(0.2)
    assert cfg.infonce_scale_init == pytest.approx(10.0)
    assert cfg.mam_warmup_epochs == 10
    assert cfg.action_layout["dof_order"] == [
        "follow_left_ee_cartesian_pos",
        "follow_left_ee_rotation_6D",
        "follow_left_gripper",
    ]


def test_xtokenizer_config_resolves_explicit_named_reordered_layout() -> None:
    cfg = XTokenizerModelConfig.from_dict(
        {
            "action_dim": 10,
            "action_layout": REORDERED_LAYOUT,
        }
    )

    assert cfg.action_layout == REORDERED_LAYOUT
    assert cfg.position_slice == slice(1, 4)
    assert cfg.rotation_slice == slice(4, 10)
    assert cfg.scalar_channel_indices == (0, 1, 2, 3)


def test_xtokenizer_config_supports_all_named_dual_arm_pose_groups() -> None:
    cfg = XTokenizerModelConfig.from_dict(
        {"action_dim": 20, "action_layout": DUAL_ARM_LAYOUT}
    )

    assert cfg.position_slices == (slice(0, 3), slice(10, 13))
    assert cfg.rotation_slices == (slice(3, 9), slice(13, 19))
    assert cfg.scalar_channel_indices == (0, 1, 2, 9, 10, 11, 12, 19)


def test_xtokenizer_26d_default_matches_the_paper_channel_table() -> None:
    cfg = XTokenizerModelConfig.from_dict({"action_dim": 26})

    assert cfg.action_layout["dof_dims"] == {
        "follow_left_ee_cartesian_pos": 3,
        "follow_left_ee_rotation_6D": 6,
        "follow_right_ee_cartesian_pos": 3,
        "follow_right_ee_rotation_6D": 6,
        "follow_left_gripper": 1,
        "follow_right_gripper": 1,
        "base_velocity": 3,
        "lift_height": 1,
        "head_pitch_yaw": 2,
    }
    assert cfg.position_slices == (slice(0, 3), slice(9, 12))
    assert cfg.rotation_slices == (slice(3, 9), slice(12, 18))


def test_xtokenizer_config_rejects_layout_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="action_layout.*action_dim"):
        XTokenizerModelConfig.from_dict(
            {
                "action_dim": 11,
                "action_layout": REORDERED_LAYOUT,
            }
        )


@pytest.mark.parametrize(
    "field",
    [
        "action_dim",
        "action_horizon",
        "compression_ratio",
        "hidden_dim",
        "latent_dim",
        "encoder_num_heads",
        "encoder_depth",
        "decoder_num_heads",
        "decoder_depth",
        "mam_num_heads",
        "mam_depth",
        "prediction_num_heads",
        "prediction_depth",
        "codebook_size",
        "num_quantizers",
        "kmeans_iters",
        "embodiment_vocab_size",
        "vl_feature_dim",
    ],
)
def test_xtokenizer_config_rejects_non_positive_dimensions(field: str) -> None:
    values = {"action_dim": 10, field: 0}

    with pytest.raises(ValueError, match=field):
        XTokenizerModelConfig.from_dict(values)


def test_xtokenizer_config_allows_deferred_action_dimension() -> None:
    cfg = XTokenizerModelConfig.from_dict({})

    assert cfg.action_dim is None


def test_xtokenizer_config_requires_divisible_action_horizon() -> None:
    with pytest.raises(ValueError, match="divisible"):
        XTokenizerModelConfig.from_dict(
            {"action_dim": 10, "action_horizon": 30, "compression_ratio": 4}
        )


@pytest.mark.parametrize(
    "field",
    [
        "reconstruction_weight",
        "commitment_weight",
        "rotation_weight",
        "dct_weight",
        "smoothness_weight",
        "mam_weight",
        "alignment_weight",
        "prediction_weight",
    ],
)
def test_xtokenizer_config_rejects_negative_objective_weights(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        XTokenizerModelConfig.from_dict({"action_dim": 10, field: -0.1})


@pytest.mark.parametrize(
    "field",
    [
        "reconstruction_weight",
        "commitment_weight",
        "rotation_weight",
        "dct_weight",
        "smoothness_weight",
        "mam_weight",
        "alignment_weight",
        "prediction_weight",
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_xtokenizer_config_rejects_non_finite_objective_weights(
    field: str, value: float
) -> None:
    with pytest.raises(ValueError, match=field):
        XTokenizerModelConfig.from_dict({"action_dim": 10, field: value})


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_xtokenizer_config_rejects_invalid_mam_mask_probability(value: float) -> None:
    with pytest.raises(ValueError, match="mam_mask_probability"):
        XTokenizerModelConfig.from_dict(
            {"action_dim": 10, "mam_mask_probability": value}
        )


@pytest.mark.parametrize(
    "value",
    [0.0, -1.0, 100.1, float("nan"), float("inf"), float("-inf")],
)
def test_xtokenizer_config_rejects_invalid_infonce_scale_init(value: float) -> None:
    with pytest.raises(ValueError, match="infonce_scale_init"):
        XTokenizerModelConfig.from_dict({"action_dim": 10, "infonce_scale_init": value})


@pytest.mark.parametrize("field", ["mam_warmup_epochs", "mam_warmup_steps"])
def test_xtokenizer_config_rejects_negative_warmup_controls(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        XTokenizerModelConfig.from_dict({"action_dim": 10, field: -1})


def test_xtokenizer_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unknown_key"):
        XTokenizerModelConfig.from_dict({"action_dim": 10, "unknown_key": True})
