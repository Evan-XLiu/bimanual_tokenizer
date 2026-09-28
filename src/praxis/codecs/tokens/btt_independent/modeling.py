# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Independent directional cross-attention for the BTT ablation."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from praxis.codecs.tokens.btt.modeling import causal_attention_mask


class DirectionalCausalCrossAttention(nn.Module):
    """One direction, with its own normalization and attention parameters."""

    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        query_source = self.norm(query)
        context_source = self.norm(context)
        update, _ = self.attention(
            query_source,
            context_source,
            context_source,
            attn_mask=causal_attention_mask(query.shape[1], query.device),
            need_weights=False,
        )
        return query + update


class IndependentCausalCrossAttention(nn.Module):
    """Both directions read the original inputs; no direction shares parameters."""

    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.left_from_right = DirectionalCausalCrossAttention(dim, num_heads, dropout)
        self.right_from_left = DirectionalCausalCrossAttention(dim, num_heads, dropout)

    def forward(self, left: Tensor, right: Tensor) -> tuple[Tensor, Tensor]:
        if left.shape != right.shape:
            raise ValueError("Left and right features must have identical shapes.")
        return self.left_from_right(left, right), self.right_from_left(right, left)
