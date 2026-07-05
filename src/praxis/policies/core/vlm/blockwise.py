# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for scheduled VLM action-token prediction."""

from __future__ import annotations

from torch import Tensor

from praxis.policies.core.vlm.vocab import ActionTokenVocabulary


def render_block_token_string(
    token_ids: list[int],
    *,
    mask_sentinel: int,
    vocab: ActionTokenVocabulary,
) -> str:
    """Render one raw token row, preserving input-only mask positions."""
    parts: list[str] = []
    for token_id in token_ids:
        if int(token_id) == int(mask_sentinel):
            parts.append(vocab.mask_token)
        else:
            parts.append(vocab.tokens[int(token_id)])
    return "".join(parts)


def render_block_token_strings(
    token_ids: Tensor,
    *,
    mask_sentinel: int,
    vocab: ActionTokenVocabulary,
) -> list[str]:
    """Render a scheduled token batch into VLM strings."""
    if token_ids.ndim != 2:
        raise ValueError(
            f"Token-id batch must be rank-2 with shape (B, token_horizon), got {tuple(token_ids.shape)}."
        )
    return [
        render_block_token_string(
            row.tolist(),
            mask_sentinel=mask_sentinel,
            vocab=vocab,
        )
        for row in token_ids.detach().cpu()
    ]
