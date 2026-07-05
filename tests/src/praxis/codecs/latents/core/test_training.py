# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared latent-codec training helpers."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from praxis.codecs.latents.core import masked_autoencoder_loss, split_decay_param_groups


class _TinyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.scale = nn.Parameter(torch.ones(()))
        self.frozen = nn.Parameter(torch.ones(2, 2), requires_grad=False)


def test_masked_autoencoder_loss_matches_plain_mean_without_padding() -> None:
    raw_loss = torch.tensor(
        [
            [[1.0, 3.0], [5.0, 7.0]],
            [[2.0, 4.0], [6.0, 8.0]],
        ]
    )

    assert masked_autoencoder_loss(raw_loss, None) == raw_loss.mean()
    assert (
        masked_autoencoder_loss(raw_loss, torch.zeros((2, 2), dtype=torch.bool))
        == raw_loss.mean()
    )


def test_masked_autoencoder_loss_drops_samples_with_any_padding() -> None:
    raw_loss = torch.tensor(
        [
            [[1.0, 3.0], [5.0, 7.0]],
            [[100.0, 100.0], [100.0, 100.0]],
        ]
    )
    pad_mask = torch.tensor([[False, False], [False, True]])

    loss = masked_autoencoder_loss(raw_loss, pad_mask)

    assert loss == torch.tensor(4.0)


def test_masked_autoencoder_loss_returns_zero_when_all_samples_padded() -> None:
    raw_loss = torch.ones((2, 3, 4))
    pad_mask = torch.ones((2, 3), dtype=torch.bool)

    assert masked_autoencoder_loss(raw_loss, pad_mask) == torch.tensor(0.0)


def test_masked_autoencoder_loss_validates_batch_size() -> None:
    with pytest.raises(ValueError, match="batch size must match"):
        masked_autoencoder_loss(
            torch.ones((2, 3, 4)),
            torch.zeros((3, 3), dtype=torch.bool),
        )


def test_split_decay_param_groups_separates_matrix_and_non_matrix_params() -> None:
    module = _TinyModule()

    groups = split_decay_param_groups(
        module,
        decay_weight_decay=None,
        no_decay_weight_decay=0.0,
    )

    assert len(groups) == 2
    decay_ids = {id(param) for param in groups[0]["params"]}
    no_decay_ids = {id(param) for param in groups[1]["params"]}
    assert id(module.linear.weight) in decay_ids
    assert id(module.linear.bias) in no_decay_ids
    assert id(module.scale) in no_decay_ids
    assert id(module.frozen) not in decay_ids | no_decay_ids
    assert "weight_decay" not in groups[0]
    assert groups[1]["weight_decay"] == 0.0


def test_split_decay_param_groups_can_force_decay_group_weight_decay() -> None:
    module = _TinyModule()

    groups = split_decay_param_groups(module, decay_weight_decay=0.0)

    assert groups[0]["weight_decay"] == 0.0
    assert groups[1]["weight_decay"] == 0.0
