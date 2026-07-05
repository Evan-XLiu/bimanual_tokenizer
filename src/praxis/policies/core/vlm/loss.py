# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Generic loss helpers for refactored action-token policies."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor

BLOCK_CE_WEIGHTING_MODES = frozenset({"token_mean", "block_mean"})
CE_VOCAB_SCOPES = frozenset({"full", "action"})


def validate_block_ce_weighting(value: object) -> str:
    """Validate the configured blockwise CE reduction mode."""
    if not isinstance(value, str):
        raise TypeError(
            f"block_ce_weighting must be a string, got {type(value).__name__}."
        )
    if value not in BLOCK_CE_WEIGHTING_MODES:
        expected = ", ".join(sorted(BLOCK_CE_WEIGHTING_MODES))
        raise ValueError(
            f"Unsupported block_ce_weighting={value!r}. Expected one of: {expected}."
        )
    return value


def validate_ce_vocab_scope(value: object) -> str:
    """Validate which vocabulary rows are exposed to action-token CE."""
    if not isinstance(value, str):
        raise TypeError(f"ce_vocab_scope must be a string, got {type(value).__name__}.")
    if value not in CE_VOCAB_SCOPES:
        expected = ", ".join(sorted(CE_VOCAB_SCOPES))
        raise ValueError(
            f"Unsupported ce_vocab_scope={value!r}. Expected one of: {expected}."
        )
    return value


def resolve_token_lengths(
    token_ids_batch: Iterable[Iterable[int]],
    *,
    pad_token_id: int | None = None,
) -> list[int]:
    """Resolve per-sample token lengths from a Python token-id batch.

    When ``pad_token_id`` is provided, contiguous trailing pad tokens are removed
    from each sample for length estimation.
    """
    lengths: list[int] = []
    for sample_ids in token_ids_batch:
        ids = [int(x) for x in sample_ids]
        if pad_token_id is None:
            lengths.append(len(ids))
            continue
        end = len(ids)
        while end > 0 and ids[end - 1] == int(pad_token_id):
            end -= 1
        lengths.append(end)
    return lengths


def collect_block_loss_chunks(
    *,
    ce_per_token: Tensor,
    supervised_mask: Tensor,
    block_schedule: tuple[int, ...],
    allow_partial_blocks: bool = False,
) -> list[tuple[Tensor, Tensor, int]]:
    """Collect per-block CE chunks over supervised positions.

    Returns tuples ``(block_ce, block_weights, block_size)`` where:
    - ``block_ce`` has shape ``(active_rows, block_size)``,
    - ``block_weights`` is a float mask with the same shape (1=supervised).

    When ``allow_partial_blocks=False`` this enforces dense supervision in each
    block for all active rows, which is the fixed-horizon action-token case.

    When ``allow_partial_blocks=True`` partial supervision is allowed (required
    for EOS-terminated adaptive supervision where suffix positions are ignored).
    """
    if ce_per_token.ndim != 2 or supervised_mask.ndim != 2:
        raise ValueError(
            "ce_per_token and supervised_mask must both be rank-2; got "
            f"{tuple(ce_per_token.shape)} and {tuple(supervised_mask.shape)}."
        )
    if ce_per_token.shape != supervised_mask.shape:
        raise ValueError(
            "ce_per_token and supervised_mask shapes must match, got "
            f"{tuple(ce_per_token.shape)} and {tuple(supervised_mask.shape)}."
        )

    active_rows = supervised_mask.any(dim=1)
    if not bool(active_rows.any()):
        return []

    active_ce = ce_per_token[active_rows]
    active_mask = supervised_mask[active_rows]
    active_batch_size = int(active_ce.shape[0])
    supervised_index = active_mask.long().cumsum(dim=1) - 1

    block_chunks: list[tuple[Tensor, Tensor, int]] = []
    offset = 0
    for raw_block_size in block_schedule:
        block_size = int(raw_block_size)
        block_mask = (
            active_mask
            & supervised_index.ge(offset)
            & supervised_index.lt(offset + block_size)
        )
        actual_count = int(block_mask.sum().item())
        expected_count = active_batch_size * block_size

        if not allow_partial_blocks and actual_count != expected_count:
            raise RuntimeError(
                "Unexpected supervised-token layout while grouping block losses: "
                f"block_size={block_size}, expected={expected_count}, actual={actual_count}."
            )

        block_ce = torch.zeros(
            (active_batch_size, block_size),
            dtype=active_ce.dtype,
            device=active_ce.device,
        )
        block_w = torch.zeros(
            (active_batch_size, block_size),
            dtype=torch.float32,
            device=active_ce.device,
        )

        row_ids, _col_ids = torch.nonzero(block_mask, as_tuple=True)
        if row_ids.numel() > 0:
            # Dense ordering by supervised index within each row.
            pos_in_block = supervised_index[block_mask] - int(offset)
            block_ce[row_ids, pos_in_block] = active_ce[block_mask]
            block_w[row_ids, pos_in_block] = 1.0

        block_chunks.append((block_ce, block_w, block_size))
        offset += block_size

    return block_chunks


def reduce_block_loss_chunks(
    block_chunks: list[tuple[Tensor, Tensor, int]],
    *,
    weighting_mode: str,
) -> Tensor | None:
    """Reduce block loss chunks under token_mean or block_mean semantics.

    Supports partial blocks through per-position weights.
    """
    non_empty = [
        (ce, w, int(block_size))
        for ce, w, block_size in block_chunks
        if ce.numel() > 0 and float(w.sum().item()) > 0.0
    ]
    if not non_empty:
        return None

    weighting_mode = validate_block_ce_weighting(weighting_mode)

    if weighting_mode == "token_mean":
        numerator = torch.stack([(ce * w).sum() for ce, w, _ in non_empty]).sum()
        denominator = (
            torch.stack([w.sum() for _, w, _ in non_empty]).sum().clamp(min=1e-8)
        )
        return numerator / denominator

    if weighting_mode == "block_mean":
        block_terms: list[Tensor] = []
        for ce, w, block_size in non_empty:
            row_weight = w.sum(dim=1)
            valid_rows = row_weight > 0
            if not bool(valid_rows.any()):
                continue
            # Each block contributes equally after nominal block-size normalization.
            row_term = ((ce * w).sum(dim=1) / float(block_size))[valid_rows]
            block_terms.append(row_term)
        if not block_terms:
            return None
        return torch.cat(block_terms, dim=0).mean()

    raise AssertionError(f"unreachable block CE weighting mode: {weighting_mode!r}")
