# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared decode sampling helper."""

from __future__ import annotations

import pytest
import torch

from praxis.policies.bar.core.sampling import sample_next_token_ids


def test_sample_next_token_ids_temperature_zero_is_greedy():
    logits = torch.tensor([[1.0, 3.0, 2.0], [0.5, -1.0, 0.0]])
    out = sample_next_token_ids(logits, temperature=0.0, top_k=None)
    assert out.shape == (2, 1)
    assert out.tolist() == [[1], [0]]


def test_sample_next_token_ids_top_k_restricts_candidates():
    logits = torch.tensor([[10.0, 9.0, 1.0, -1.0]])
    torch.manual_seed(0)
    samples = [
        int(sample_next_token_ids(logits, temperature=1.0, top_k=2).item())
        for _ in range(64)
    ]
    assert set(samples).issubset({0, 1})


def test_sample_next_token_ids_uses_explicit_generator():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    first = sample_next_token_ids(
        logits,
        temperature=1.0,
        top_k=None,
        generator=torch.Generator().manual_seed(42),
    )
    torch.manual_seed(999)
    second = sample_next_token_ids(
        logits,
        temperature=1.0,
        top_k=None,
        generator=torch.Generator().manual_seed(42),
    )
    assert torch.equal(first, second)


def test_sample_next_token_ids_invalid_args_raise():
    logits = torch.randn(2, 5)
    with pytest.raises(ValueError):
        _ = sample_next_token_ids(logits, temperature=-0.1, top_k=None)
    with pytest.raises(ValueError):
        _ = sample_next_token_ids(logits, temperature=1.0, top_k=0)


def test_sample_next_token_ids_allowed_token_ids_restricts_greedy():
    logits = torch.tensor([[10.0, 9.0, 1.0, -1.0]])
    out = sample_next_token_ids(
        logits,
        temperature=0.0,
        top_k=None,
        allowed_token_ids=[2, 3],
    )
    assert out.tolist() == [[2]]


def test_sample_next_token_ids_allowed_token_ids_out_of_range_raises():
    logits = torch.randn(1, 4)
    with pytest.raises(ValueError, match="out-of-range"):
        _ = sample_next_token_ids(
            logits,
            temperature=0.0,
            top_k=None,
            allowed_token_ids=[0, 7],
        )
