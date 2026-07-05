# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""BAR generated-token filtering and extraction helpers."""

from __future__ import annotations

import torch
from torch import Tensor

from praxis.policies.core.vlm.vocab import ActionTokenVocabulary


def build_allowed_decode_token_ids(
    *,
    vocab: ActionTokenVocabulary,
    eos_token_id: int | None = None,
) -> list[int]:
    """Return the BAR decode allowlist for action tokens plus optional EOS."""
    ids = list(range(int(vocab.token_min), int(vocab.token_max) + 1))
    if eos_token_id is not None:
        ids.append(int(eos_token_id))
    return ids


def raw_token_ids_to_vlm_tensor(
    batch_token_ids: Tensor | list[list[int]],
    *,
    device: torch.device,
    mask_sentinel: int,
    vocab: ActionTokenVocabulary,
) -> Tensor:
    """Map BAR raw decode token ids into VLM token ids, preserving masks."""
    if isinstance(batch_token_ids, Tensor):
        token_tensor = batch_token_ids.to(device=device, dtype=torch.long)
    else:
        if not batch_token_ids:
            return torch.empty((0, 0), dtype=torch.long, device=device)
        token_tensor = torch.as_tensor(batch_token_ids, dtype=torch.long, device=device)

    if token_tensor.ndim != 2:
        raise ValueError(
            f"Prediction blocks must be rank-2 with shape (B, block_size), got {tuple(token_tensor.shape)}."
        )
    if token_tensor.numel() == 0:
        return torch.empty((0, 0), dtype=torch.long, device=device)

    return torch.where(
        token_tensor.eq(mask_sentinel),
        torch.full_like(token_tensor, int(vocab.mask_token_id)),
        token_tensor + int(vocab.token_min),
    )


def build_prediction_inputs(
    *,
    batch_size: int,
    block_size: int,
    device: torch.device,
    prev_block_ids: Tensor | None,
    mask_sentinel: int,
) -> Tensor:
    """Build BAR's next block input from previous decoded raw token ids."""
    pred_block_ids = torch.full(
        (batch_size, int(block_size)),
        fill_value=mask_sentinel,
        dtype=torch.long,
        device=device,
    )
    if prev_block_ids is not None:
        copy_size = min(int(prev_block_ids.shape[1]), int(block_size))
        pred_block_ids[:, :copy_size] = prev_block_ids[:, :copy_size]
    return pred_block_ids


def extract_contiguous_action_token_ids(
    generated_ids: Tensor,
    *,
    vocab: ActionTokenVocabulary,
    prompt_length: int = 0,
) -> list[list[int]]:
    """Extract the first contiguous generated span of BAR action-token ids."""
    if generated_ids.ndim != 2:
        raise ValueError(
            f"Expected generated_ids rank-2 tensor, got shape {tuple(generated_ids.shape)}."
        )
    prompt_length = int(prompt_length)
    if prompt_length < 0:
        raise ValueError(f"prompt_length must be >= 0, got {prompt_length}.")
    if prompt_length >= int(generated_ids.shape[1]):
        return [[] for _ in range(int(generated_ids.shape[0]))]
    if prompt_length > 0:
        generated_ids = generated_ids[:, prompt_length:]

    mask = (generated_ids >= int(vocab.token_min)) & (
        generated_ids <= int(vocab.token_max)
    )
    batch_size, seq_len = mask.shape
    device = mask.device
    has_action = mask.any(dim=1)
    if not has_action.any():
        return [[] for _ in range(batch_size)]

    positions = torch.arange(seq_len, device=device)
    first_action = mask.to(torch.uint8).argmax(dim=1)
    after_start = positions.unsqueeze(0) >= first_action.unsqueeze(1)
    gap = after_start & ~mask
    gap_or_len = torch.where(gap, positions.unsqueeze(0), seq_len)
    span_end = gap_or_len.min(dim=1).values
    span_len = torch.where(
        has_action, span_end - first_action, torch.zeros_like(span_end)
    )
    max_span = int(span_len.max().item())
    if max_span == 0:
        return [[] for _ in range(batch_size)]

    offsets = torch.arange(max_span, device=device).unsqueeze(0)
    gather_idx = (first_action.unsqueeze(1) + offsets).clamp(max=seq_len - 1)
    gathered = generated_ids.gather(1, gather_idx) - int(vocab.token_min)

    gathered_list = gathered.tolist()
    span_len_list = span_len.tolist()
    return [
        row[: int(slen)] if slen > 0 else []
        for row, slen in zip(gathered_list, span_len_list, strict=True)
    ]
