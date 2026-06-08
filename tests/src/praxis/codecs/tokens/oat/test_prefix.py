# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for OAT-family token prefix helpers."""

from __future__ import annotations

import pytest
import torch

from praxis.codecs.tokens.oat.prefix import prefix_token_ids_to_tensor


def test_prefix_token_ids_to_tensor_handles_tensor_prefix() -> None:
    tokens = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)

    out, lengths = prefix_token_ids_to_tensor(
        tokens,
        token_horizon=4,
        device=torch.device("cpu"),
    )

    assert lengths == [2, 2]
    assert out.equal(torch.tensor([[1, 2, 0, 0], [3, 4, 0, 0]], dtype=torch.long))


def test_prefix_token_ids_to_tensor_handles_variable_lists() -> None:
    out, lengths = prefix_token_ids_to_tensor(
        [[1], [2, 3]],
        token_horizon=3,
        device=torch.device("cpu"),
    )

    assert lengths == [1, 2]
    assert out.equal(torch.tensor([[1, 0, 0], [2, 3, 0]], dtype=torch.long))


def test_prefix_token_ids_to_tensor_rejects_empty_lists() -> None:
    with pytest.raises(ValueError, match="empty token sequence"):
        prefix_token_ids_to_tensor([], token_horizon=3, device=None)
