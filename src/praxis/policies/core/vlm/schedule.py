# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Shared scheduled action-token teacher-forcing and attention-mask utilities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch

PrefixAttentionMode = Literal["bidirectional", "causal"]


def overlay_bidirectional_token_mask(
    attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor | None,
) -> torch.Tensor:
    """Let marked prefix tokens attend to one another in a prepared 4D mask."""
    if token_type_ids is None:
        return attention_mask
    if attention_mask.ndim != 4 or attention_mask.shape[1] != 1:
        raise ValueError("Expected attention mask with shape (B, 1, L, L).")
    if token_type_ids.ndim != 2 or attention_mask.shape[0] != token_type_ids.shape[0]:
        raise ValueError("Token type ids must match the attention mask batch.")
    seq_len = token_type_ids.shape[1]
    if attention_mask.shape[-2:] != (seq_len, seq_len):
        raise ValueError("Token type ids must match the attention mask sequence.")
    marked = token_type_ids.to(device=attention_mask.device, dtype=torch.bool)
    token_pairs = marked[:, None, :, None] & marked[:, None, None, :]
    if attention_mask.dtype == torch.bool:
        return attention_mask | token_pairs
    return attention_mask.masked_fill(token_pairs, 0)


def _schedule_tensor(
    block_schedule: Sequence[int],
    *,
    device: torch.device | str,
    token_horizon: int | None = None,
) -> torch.Tensor:
    schedule = validate_block_schedule(block_schedule, token_horizon=token_horizon)
    return torch.as_tensor(schedule, dtype=torch.long, device=device)


def _build_suffix_block_ids(
    *,
    seq_len: int,
    schedule_t: torch.Tensor,
) -> torch.Tensor:
    """Return per-position block ids, using ``-1`` for prefix positions."""
    suffix_len = int(schedule_t.sum().item())
    if suffix_len > int(seq_len):
        raise ValueError(
            f"VLM suffix length {suffix_len} exceeds sequence length {seq_len}."
        )

    block_ids = torch.full(
        (int(seq_len),), -1, dtype=torch.long, device=schedule_t.device
    )
    suffix_ids = torch.repeat_interleave(
        torch.arange(
            int(schedule_t.numel()), device=schedule_t.device, dtype=torch.long
        ),
        schedule_t,
    )
    block_ids[int(seq_len) - suffix_len :] = suffix_ids
    return block_ids


def _build_position_block_ids(
    *,
    attention_mask_2d: torch.Tensor,
    prediction_position_mask: torch.Tensor,
    schedule_t: torch.Tensor,
) -> torch.Tensor:
    """Return per-sample block ids from actual prediction-token positions."""
    if prediction_position_mask.shape != attention_mask_2d.shape:
        raise ValueError(
            "prediction_position_mask shape must match attention_mask_2d, got "
            f"{tuple(prediction_position_mask.shape)} and {tuple(attention_mask_2d.shape)}."
        )
    valid_mask = attention_mask_2d.bool()
    prediction_mask = prediction_position_mask.to(
        device=attention_mask_2d.device
    ).bool()
    prediction_mask &= valid_mask

    suffix_len = int(schedule_t.sum().item())
    prediction_counts = prediction_mask.sum(dim=1)
    expected = torch.full_like(prediction_counts, suffix_len)
    if bool((prediction_counts != expected).any()):
        raise ValueError(
            "prediction_position_mask must contain exactly one scheduled suffix per row; "
            f"got counts={prediction_counts.tolist()} expected={expected.tolist()}."
        )

    batch_size, seq_len = attention_mask_2d.shape
    block_ids = torch.full(
        (int(batch_size), int(seq_len)),
        -1,
        dtype=torch.long,
        device=attention_mask_2d.device,
    )
    suffix_ids = torch.repeat_interleave(
        torch.arange(
            int(schedule_t.numel()),
            device=attention_mask_2d.device,
            dtype=torch.long,
        ),
        schedule_t,
    )
    positions = torch.arange(seq_len, device=attention_mask_2d.device)
    tail_block_id = int(schedule_t.numel())
    for row in range(int(batch_size)):
        pred_positions = prediction_mask[row].nonzero(as_tuple=False).flatten()
        first_pred = int(pred_positions[0].item())
        last_pred = int(pred_positions[-1].item())
        interleaved_non_prediction = (
            valid_mask[row]
            & ~prediction_mask[row]
            & positions.gt(first_pred)
            & positions.lt(last_pred)
        )
        if bool(interleaved_non_prediction.any()):
            bad = positions[interleaved_non_prediction].tolist()
            raise ValueError(
                "prediction positions must be contiguous except for trailing "
                f"template/EOS tokens; row {row} has interleaved positions {bad}."
            )
        block_ids[row, pred_positions] = suffix_ids
        trailing = valid_mask[row] & ~prediction_mask[row] & positions.gt(last_pred)
        block_ids[row, trailing] = tail_block_id
    return block_ids


def validate_block_schedule(
    block_schedule: Sequence[int],
    *,
    token_horizon: int | None = None,
) -> tuple[int, ...]:
    """Validate a schedule and normalize it to a tuple."""
    schedule = tuple(int(size) for size in block_schedule)
    if not schedule:
        raise ValueError("block_schedule must be non-empty.")
    if any(size <= 0 for size in schedule):
        raise ValueError(
            f"block_schedule must contain only positive ints, got {schedule}."
        )
    if any(left > right for left, right in zip(schedule, schedule[1:], strict=False)):
        raise ValueError(f"block_schedule must be non-decreasing, got {schedule}.")
    if token_horizon is not None and sum(schedule) != int(token_horizon):
        raise ValueError(
            f"block_schedule sum {sum(schedule)} must match token_horizon={token_horizon}."
        )
    return schedule


def resolve_block_schedule(
    block_schedule: Sequence[int] | None,
    *,
    token_horizon: int,
) -> tuple[int, ...]:
    """Resolve optional tokenizer metadata to a concrete schedule."""
    horizon = int(token_horizon)
    if horizon <= 0:
        raise ValueError(f"token_horizon must be positive, got {token_horizon}.")
    if block_schedule is None:
        return (1,) * horizon
    return validate_block_schedule(block_schedule, token_horizon=horizon)


def build_prediction_block_input_ids(
    target_token_ids: torch.Tensor,
    block_schedule: Sequence[int],
    *,
    mask_token_id: int,
) -> torch.Tensor:
    """Construct block-shift teacher-forcing inputs from target token ids."""
    if target_token_ids.ndim != 2:
        raise ValueError(
            "target_token_ids must be rank-2 with shape (B, token_horizon), "
            f"got {tuple(target_token_ids.shape)}."
        )
    schedule_t = _schedule_tensor(
        block_schedule,
        device=target_token_ids.device,
        token_horizon=target_token_ids.shape[1],
    )
    out = torch.full_like(target_token_ids, int(mask_token_id))

    block_ids = torch.repeat_interleave(
        torch.arange(
            int(schedule_t.numel()), device=target_token_ids.device, dtype=torch.long
        ),
        schedule_t,
    )
    block_starts = torch.cumsum(schedule_t, dim=0) - schedule_t
    prev_sizes = torch.cat([schedule_t.new_zeros(1), schedule_t[:-1]])
    prev_starts = torch.cat([schedule_t.new_zeros(1), block_starts[:-1]])
    copy_lengths = torch.minimum(prev_sizes, schedule_t)

    positions = torch.arange(target_token_ids.shape[1], device=target_token_ids.device)
    intra_block_offset = positions - block_starts[block_ids]
    copy_mask = intra_block_offset < copy_lengths[block_ids]
    source_positions = prev_starts[block_ids] + intra_block_offset

    if bool(copy_mask.any()):
        out[:, copy_mask] = target_token_ids[:, source_positions[copy_mask]]

    return out


def build_attention_mask(
    attention_mask_2d: torch.Tensor,
    block_schedule: Sequence[int],
    dtype: torch.dtype,
    *,
    prefix_attention: PrefixAttentionMode,
    prediction_position_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a 4D schedule-aware attention mask for left-padded prefix + fixed suffix inputs."""
    if attention_mask_2d.ndim != 2:
        raise ValueError(
            f"attention_mask_2d must be rank-2 with shape (B, L), got {tuple(attention_mask_2d.shape)}."
        )

    batch_size, seq_len = attention_mask_2d.shape
    schedule_t = _schedule_tensor(block_schedule, device=attention_mask_2d.device)

    device = attention_mask_2d.device
    min_val = torch.finfo(dtype).min
    zero = torch.zeros((), dtype=dtype, device=device)

    if prediction_position_mask is None:
        block_ids = (
            _build_suffix_block_ids(
                seq_len=seq_len,
                schedule_t=schedule_t,
            )
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
    else:
        block_ids = _build_position_block_ids(
            attention_mask_2d=attention_mask_2d,
            prediction_position_mask=prediction_position_mask,
            schedule_t=schedule_t,
        )

    query_ids = block_ids.unsqueeze(2)
    key_ids = block_ids.unsqueeze(1)
    allow = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.bool, device=device)

    is_prefix_query = query_ids.eq(-1)
    is_prefix_key = key_ids.eq(-1)
    is_suffix_query = query_ids.ge(0)
    is_suffix_key = key_ids.ge(0)

    if prefix_attention == "bidirectional":
        allow |= is_prefix_query & is_prefix_key
    elif prefix_attention == "causal":
        positions = torch.arange(seq_len, device=device)
        prefix_causal = positions.view(1, seq_len, 1) >= positions.view(1, 1, seq_len)
        allow |= is_prefix_query & is_prefix_key & prefix_causal
    else:
        raise ValueError(
            f"Unsupported prefix_attention={prefix_attention!r}; expected 'bidirectional' or 'causal'."
        )
    allow |= is_suffix_query & is_prefix_key
    allow |= is_suffix_query & is_suffix_key & key_ids.le(query_ids)

    mask = torch.where(
        allow,
        zero,
        torch.full((batch_size, seq_len, seq_len), min_val, dtype=dtype, device=device),
    )

    pad = attention_mask_2d == 0
    mask = mask.masked_fill(pad[:, None, :], min_val)
    return mask[:, None, :, :]
