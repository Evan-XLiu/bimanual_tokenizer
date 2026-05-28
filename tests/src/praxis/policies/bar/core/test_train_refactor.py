"""Tests for refactored generic action-token training helpers."""

from __future__ import annotations

import pytest
import torch

from praxis.policies.core.vlm.loss import (
    collect_block_loss_chunks,
    reduce_block_loss_chunks,
    resolve_token_lengths,
)
from praxis.policies.core.vlm.supervision import build_eos_supervision_targets


def test_resolve_token_lengths_without_pad() -> None:
    out = resolve_token_lengths([[1, 2, 3], [9], []])
    assert out == [3, 1, 0]


def test_resolve_token_lengths_with_trailing_pad() -> None:
    out = resolve_token_lengths([[1, 2, 0, 0], [5, 6], [0, 0]], pad_token_id=0)
    assert out == [2, 2, 0]


def test_build_eos_supervision_targets_masks_suffix_after_eos() -> None:
    # horizon=6, second sample has full length so no EOS slot in-horizon.
    target = torch.tensor([[10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25]])
    labels = build_eos_supervision_targets(
        target_vlm_token_ids=target,
        lengths=[3, 6],
        eos_token_id=99,
        ignore_index=-100,
    )
    expected = torch.tensor([[10, 11, 12, 99, -100, -100], [20, 21, 22, 23, 24, 25]])
    assert torch.equal(labels, expected)


def test_collect_block_loss_chunks_requires_dense_when_partial_disallowed() -> None:
    ce = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    supervised = torch.tensor([[True, True, False, False]])
    with pytest.raises(RuntimeError, match="Unexpected supervised-token layout"):
        collect_block_loss_chunks(
            ce_per_token=ce,
            supervised_mask=supervised,
            block_schedule=(2, 2),
            allow_partial_blocks=False,
        )


def test_collect_block_loss_chunks_allows_partial_blocks() -> None:
    ce = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    supervised = torch.tensor([[True, True, False, False]])
    chunks = collect_block_loss_chunks(
        ce_per_token=ce,
        supervised_mask=supervised,
        block_schedule=(2, 2),
        allow_partial_blocks=True,
    )
    assert len(chunks) == 2

    first_ce, first_w, first_size = chunks[0]
    assert first_size == 2
    assert torch.equal(first_ce, torch.tensor([[1.0, 2.0]]))
    assert torch.equal(first_w, torch.tensor([[1.0, 1.0]]))

    second_ce, second_w, second_size = chunks[1]
    assert second_size == 2
    assert torch.equal(second_ce, torch.tensor([[0.0, 0.0]]))
    assert torch.equal(second_w, torch.tensor([[0.0, 0.0]]))


def test_reduce_block_loss_chunks_token_mean_with_partial_weights() -> None:
    chunks = [
        (
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([[1.0, 1.0]]),
            2,
        ),
        (
            torch.tensor([[8.0, 0.0]]),
            torch.tensor([[1.0, 0.0]]),
            2,
        ),
    ]
    loss = reduce_block_loss_chunks(chunks, weighting_mode="token_mean")
    assert loss is not None
    # weighted supervised mean = (1 + 2 + 8) / 3
    assert torch.allclose(loss, torch.tensor((1.0 + 2.0 + 8.0) / 3.0))


def test_reduce_block_loss_chunks_block_mean_keeps_nominal_block_size_norm() -> None:
    chunks = [
        (
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([[1.0, 1.0]]),
            2,
        ),
        (
            torch.tensor([[8.0, 0.0]]),
            torch.tensor([[1.0, 0.0]]),
            2,
        ),
    ]
    loss = reduce_block_loss_chunks(chunks, weighting_mode="block_mean")
    assert loss is not None
    # block1 row term=(1+2)/2=1.5, block2 row term=(8+0)/2=4.0, mean=2.75
    assert torch.allclose(loss, torch.tensor(2.75))
