# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""BAR decode scheduling and incremental attention-mask helpers."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from praxis.policies.core.vlm.schedule import validate_block_schedule


def resolve_decode_keep_k(
    keep_k: int | None,
    *,
    token_horizon: int,
) -> int:
    """Resolve BAR fixed-budget decode length against the tokenizer horizon."""
    horizon = int(token_horizon)
    if keep_k is None:
        return horizon
    keep_k = int(keep_k)
    if keep_k < 1:
        raise ValueError(f"decode_keep_k must be >= 1, got {keep_k}.")
    if keep_k > horizon:
        raise ValueError(
            f"decode_keep_k must be <= token_horizon ({horizon}), got {keep_k}."
        )
    return keep_k


def plan_generation_schedule(
    block_schedule: Sequence[int],
    *,
    keep_k: int,
) -> tuple[int, ...]:
    """Truncate a BAR block schedule to the minimum prefix covering keep_k tokens."""
    resolved_keep_k = int(keep_k)
    covered = 0
    steps: list[int] = []
    for block_size in block_schedule:
        steps.append(int(block_size))
        covered += int(block_size)
        if covered >= resolved_keep_k:
            return tuple(steps)
    raise RuntimeError(
        "Resolved block schedule did not cover the requested decode_keep_k. "
        f"keep_k={resolved_keep_k}, schedule={tuple(int(size) for size in block_schedule)}."
    )


def build_step_attention_mask(
    attention_mask_2d: torch.Tensor,
    block_schedule: Sequence[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build newest-block incremental mask rows for cached BAR decode."""
    if attention_mask_2d.ndim != 2:
        raise ValueError(
            f"attention_mask_2d must be rank-2 with shape (B, L), got {tuple(attention_mask_2d.shape)}."
        )

    batch_size, seq_len = attention_mask_2d.shape
    device = attention_mask_2d.device
    min_val = torch.finfo(dtype).min
    zero = torch.zeros((), dtype=dtype, device=device)
    schedule_t = _schedule_tensor(block_schedule, device=device)
    block_ids = _build_suffix_block_ids(seq_len=seq_len, schedule_t=schedule_t)
    step_size = int(schedule_t[-1].item())

    query_ids = block_ids[-step_size:].view(step_size, 1)
    key_ids = block_ids.view(1, seq_len)
    allow = torch.zeros((step_size, seq_len), dtype=torch.bool, device=device)

    is_prefix_key = key_ids.eq(-1)
    is_suffix_key = key_ids.ge(0)
    is_suffix_query = query_ids.ge(0)

    allow |= is_suffix_query & is_prefix_key
    allow |= is_suffix_query & is_suffix_key & key_ids.le(query_ids)

    mask = torch.where(
        allow.unsqueeze(0).expand(batch_size, -1, -1),
        zero,
        torch.full(
            (batch_size, step_size, seq_len), min_val, dtype=dtype, device=device
        ),
    )
    pad = attention_mask_2d == 0
    mask = mask.masked_fill(pad[:, None, :], min_val)
    return mask[:, None, :, :]


def _schedule_tensor(
    block_schedule: Sequence[int],
    *,
    device: torch.device | str,
) -> torch.Tensor:
    schedule = validate_block_schedule(block_schedule)
    return torch.as_tensor(schedule, dtype=torch.long, device=device)


def _build_suffix_block_ids(
    *,
    seq_len: int,
    schedule_t: torch.Tensor,
) -> torch.Tensor:
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
