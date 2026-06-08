# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Shared token-sampling and EOS-stop helpers for action-token policies."""

from __future__ import annotations

import torch
from torch import Tensor


def sample_next_token_ids(
    logits: Tensor,
    *,
    temperature: float | None,
    top_k: int | None,
    allowed_token_ids: Tensor | list[int] | None = None,
) -> Tensor:
    """Sample or greedily select next-token ids from logits."""
    if logits.ndim != 2:
        raise ValueError(
            f"Expected logits with shape (B, V), got {tuple(logits.shape)}."
        )

    if temperature is None:
        temperature = 0.0
    temperature = float(temperature)
    if temperature < 0.0:
        raise ValueError(f"temperature must be >= 0, got {temperature}.")

    if top_k is not None:
        top_k = int(top_k)
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1 when set, got {top_k}.")

    scores = logits.float()
    if allowed_token_ids is not None:
        allowed = torch.as_tensor(
            allowed_token_ids, device=scores.device, dtype=torch.long
        ).flatten()
        if allowed.numel() == 0:
            raise ValueError("allowed_token_ids must not be empty when provided.")
        if torch.any((allowed < 0) | (allowed >= scores.shape[-1])):
            raise ValueError(
                f"allowed_token_ids contains out-of-range ids for vocab_size={scores.shape[-1]}."
            )
        allow_mask = torch.zeros(
            scores.shape[-1], dtype=torch.bool, device=scores.device
        )
        allow_mask[allowed] = True
        scores = scores.masked_fill(~allow_mask.unsqueeze(0), float("-inf"))
        invalid_rows = torch.isneginf(scores).all(dim=1)
        if invalid_rows.any():
            scores[invalid_rows, int(allowed[0].item())] = 0.0

    if temperature == 0.0 or top_k == 1:
        return torch.argmax(scores, dim=-1, keepdim=True)

    scores = scores.div(temperature)
    if top_k is not None:
        k = min(top_k, int(scores.shape[-1]))
        top_values, _ = torch.topk(scores, k=k, dim=-1)
        scores = scores.masked_fill(scores < top_values[:, [-1]], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.multinomial(probs, num_samples=1)
