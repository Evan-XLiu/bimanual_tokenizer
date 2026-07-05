# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.tokenizer.fast.token_stats import (
    _infer_action_key,
    _load_train_split_config,
    _reconstruction_error_sums,
)


def test_reconstruction_error_sums_counts_action_dimensions_without_pad() -> None:
    actions = torch.zeros(2, 3, 4)
    recons = torch.ones_like(actions)

    sqerr_sum, abserr_sum, n_valid = _reconstruction_error_sums(
        recons=recons,
        actions=actions,
        action_is_pad=None,
    )

    assert sqerr_sum == 24.0
    assert abserr_sum == 24.0
    assert n_valid == 24.0


def test_reconstruction_error_sums_counts_action_dimensions_with_pad() -> None:
    actions = torch.zeros(2, 3, 4)
    recons = torch.ones_like(actions)
    action_is_pad = torch.tensor(
        [
            [False, False, False],
            [False, True, False],
        ]
    )

    sqerr_sum, abserr_sum, n_valid = _reconstruction_error_sums(
        recons=recons,
        actions=actions,
        action_is_pad=action_is_pad,
    )

    assert sqerr_sum == 20.0
    assert abserr_sum == 20.0
    assert n_valid == 20.0


def test_load_train_split_config_resolves_seed_interpolation(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "train_config.yaml").write_text(
        "\n".join(
            [
                "seed: 42",
                "dataset:",
                "  repo_id: tester/dataset",
                "trainer:",
                "  val_ratio: 0.25",
                "  split_seed: ${seed}",
            ]
        ),
        encoding="utf-8",
    )

    _episodes, _val_ratio, split_seed, *_rest = _load_train_split_config(
        checkpoint_dir=checkpoint_dir
    )

    assert split_seed == 42


def test_infer_action_key_prefers_canonical_action_feature() -> None:
    assert (
        _infer_action_key(
            {
                "observation.state": {"shape": [3]},
                "action": {"shape": [7]},
                "arm_action": {"shape": [7]},
            }
        )
        == "action"
    )


def test_infer_action_key_accepts_single_action_named_candidate() -> None:
    assert (
        _infer_action_key(
            {
                "index": {"shape": [1]},
                "observation.image": {"shape": [3, 224, 224]},
                "next.action": {"shape": [7]},
                "robot_action": {"shape": [7]},
                "robot_action_is_pad": {"shape": [1]},
            }
        )
        == "robot_action"
    )


def test_infer_action_key_rejects_ambiguous_candidates() -> None:
    with pytest.raises(ValueError, match="cannot infer action key"):
        _infer_action_key(
            {
                "left": {"shape": [7]},
                "right": {"shape": [7]},
            }
        )
