# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Shared token-sequence helpers for fixed-length and variable-length policies."""

from __future__ import annotations

import torch
from torch import Tensor

from praxis.policies.bar.core.decode_info import resolve_max_new_tokens
from praxis.policies.core.vlm.tokenization import tokenize_actions


def pad_token_id_lists_to_horizon(
    batch_ids: list[list[int]],
    *,
    device: torch.device,
    horizon: int,
    pad_value: int = 0,
) -> Tensor:
    """Pad or truncate token-id lists to a fixed horizon."""
    padded = torch.full(
        (len(batch_ids), int(horizon)),
        fill_value=int(pad_value),
        dtype=torch.long,
        device=device,
    )
    if not batch_ids:
        return padded

    flat: list[int] = []
    lengths: list[int] = []
    for ids in batch_ids:
        n = min(len(ids), int(horizon))
        lengths.append(n)
        if n > 0:
            flat.extend(ids[:n])

    if not flat:
        return padded

    vals = torch.tensor(flat, dtype=torch.long, device=device)
    len_t = torch.tensor(lengths, dtype=torch.long, device=device)
    rows = torch.arange(len(batch_ids), device=device).repeat_interleave(len_t)
    cols = torch.cat(
        [torch.arange(length, device=device) for length in lengths if length > 0]
    )
    padded[rows, cols] = vals
    return padded


def pad_variable_token_id_lists(
    batch_ids: list[list[int]],
    *,
    device: torch.device,
    pad_value: int,
    minimum_width: int = 1,
) -> Tensor:
    """Pad variable-length token-id lists to the batch max length."""
    max_len = max(max((len(ids) for ids in batch_ids), default=0), int(minimum_width))
    padded = torch.full(
        (len(batch_ids), max_len),
        fill_value=int(pad_value),
        dtype=torch.long,
        device=device,
    )
    for i, ids in enumerate(batch_ids):
        if ids:
            padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    return padded


def resolve_max_code_seq_len(*, action_tokenizer, owner_name: str) -> int:
    """Resolve the fixed code horizon from codec metadata."""
    code_seq_len = getattr(action_tokenizer, "max_code_seq_len", None)
    if code_seq_len is None:
        raise ValueError(
            f"{owner_name} requires an action codec with max_code_seq_len."
        )
    return int(code_seq_len)


def resolve_pad_horizon(
    pad_horizon: int | None,
    *,
    token_horizon: int,
) -> int:
    """Resolve optional pad horizon against the tokenizer horizon."""
    if pad_horizon is None:
        return int(token_horizon)
    resolved = int(pad_horizon)
    if resolved < 1:
        raise ValueError(f"pad_horizon must be >= 1, got {resolved}.")
    if resolved > int(token_horizon):
        raise ValueError(
            f"pad_horizon must be <= token_horizon ({token_horizon}), got {resolved}."
        )
    return resolved


def tokenize_fixed_actions(action_tokenizer, actions: Tensor) -> list[list[int]]:
    """Tokenize fixed-length actions into ID lists."""
    return tokenize_actions(action_tokenizer, actions)


def swap_btt_arm_token_blocks(batch_ids: list[list[int]]) -> list[list[int]]:
    """Convert between native [L0..7,R0..7] and VLA [R0..7,L0..7]."""
    if any(len(ids) != 16 for ids in batch_ids):
        raise ValueError("BTT arm reordering requires all 16 tokens (8 per arm).")
    return [ids[8:] + ids[:8] for ids in batch_ids]


def detokenize_fixed_action_ids(
    *,
    action_tokenizer,
    batch_token_ids: list[list[int]],
    device: torch.device,
    horizon: int,
    pad_value: int = 0,
) -> Tensor:
    """Pad/truncate fixed-length token ids to horizon, then detokenize."""
    tokens = pad_token_id_lists_to_horizon(
        batch_token_ids,
        device=device,
        horizon=horizon,
        pad_value=pad_value,
    )
    with torch.inference_mode():
        return action_tokenizer.detokenize(tokens)


def resolve_variable_action_token_budget(
    configured: int | None,
    *,
    action_tokenizer,
    owner_name: str,
) -> int:
    """Resolve the maximum number of action tokens a variable-length policy may emit."""
    return int(
        resolve_max_new_tokens(
            configured,
            codec_max_code_seq_len=getattr(action_tokenizer, "max_code_seq_len", None),
            owner_name=owner_name,
        )
    )


def resolve_variable_decode_budget(
    configured: int | None,
    *,
    action_tokenizer,
    owner_name: str,
    eos_headroom: int = 1,
) -> int:
    """Resolve variable-length decode steps, including optional EOS headroom."""
    budget = resolve_variable_action_token_budget(
        configured,
        action_tokenizer=action_tokenizer,
        owner_name=owner_name,
    )
    return int(budget) + int(eos_headroom)


def resolve_variable_pad_token_id(
    *,
    action_tokenizer,
    tokenizer_vocab_size: int,
) -> int:
    """Resolve the padding token id for a variable-length tokenizer."""
    return int(
        getattr(action_tokenizer, "_pad_token_id", int(tokenizer_vocab_size) - 1)
    )
