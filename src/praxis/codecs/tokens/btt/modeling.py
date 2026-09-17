# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Strictly causal, weight-shared modeling blocks for the BTT tokenizer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def causal_attention_mask(length: int, device: torch.device) -> Tensor:
    """Return a boolean mask in which position ``i`` cannot access ``j > i``."""

    return torch.triu(
        torch.ones((int(length), int(length)), dtype=torch.bool, device=device),
        diagonal=1,
    )


class CausalDownsampleConv1d(nn.Module):
    """A causal Conv1D whose outputs align with the end of each stride block.

    With stride two, output ``i`` is aligned to input ``2 * i + 1``.  Explicit
    left-only padding makes the dependency causal and avoids normalization over
    the temporal dimension.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int,
    ) -> None:
        super().__init__()
        if kernel_size < stride:
            raise ValueError("kernel_size must be at least stride.")
        self.left_padding = int(kernel_size) - int(stride)
        self.conv = nn.Conv1d(
            int(in_channels),
            int(out_channels),
            kernel_size=int(kernel_size),
            stride=int(stride),
            padding=0,
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected [B, C, T], got {tuple(x.shape)}.")
        return self.conv(F.pad(x, (self.left_padding, 0)))


class CausalTemporalDownsampler(nn.Module):
    """Shared single-arm projection and 32-to-8 causal temporal encoder."""

    def __init__(
        self,
        *,
        arm_dim: int,
        model_dim: int,
        kernel_sizes: tuple[int, ...],
        strides: tuple[int, ...],
        expected_horizon: int,
    ) -> None:
        super().__init__()
        self.arm_dim = int(arm_dim)
        self.expected_horizon = int(expected_horizon)
        self.action_projection = nn.Linear(self.arm_dim, int(model_dim))
        self.stages = nn.ModuleList(
            CausalDownsampleConv1d(
                int(model_dim),
                int(model_dim),
                kernel_size=kernel,
                stride=stride,
            )
            for kernel, stride in zip(kernel_sizes, strides, strict=True)
        )
        # LayerNorm acts on features independently at each time position.  In
        # contrast to GroupNorm over [C, T], it cannot leak future statistics.
        self.norms = nn.ModuleList(nn.LayerNorm(int(model_dim)) for _ in self.stages)
        self.activation = nn.Mish()

    def forward(self, arm_actions: Tensor) -> Tensor:
        if arm_actions.ndim != 3 or arm_actions.shape[-1] != self.arm_dim:
            raise ValueError(
                f"expected single-arm actions [B, T, {self.arm_dim}], "
                f"got {tuple(arm_actions.shape)}."
            )
        x = self.action_projection(arm_actions)
        for conv, norm in zip(self.stages, self.norms, strict=True):
            x = conv(x.transpose(1, 2)).transpose(1, 2)
            x = self.activation(norm(x))
        if x.shape[1] != self.expected_horizon:
            raise RuntimeError(
                "causal temporal encoder produced an unexpected horizon: "
                f"expected {self.expected_horizon}, got {x.shape[1]}."
            )
        return x


class SharedCausalSelfAttention(nn.Module):
    """Pre-norm causal self-attention reused for either arm."""

    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(dim))
        self.attention = nn.MultiheadAttention(
            int(dim), int(num_heads), dropout=float(dropout), batch_first=True
        )

    def forward(self, x: Tensor) -> Tensor:
        normalized = self.norm(x)
        update, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_attention_mask(x.shape[1], x.device),
            need_weights=False,
        )
        return x + update


class SymmetricCausalCrossAttention(nn.Module):
    """Shared-parameter cross-attention computed in parallel from both inputs."""

    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(dim))
        self.attention = nn.MultiheadAttention(
            int(dim), int(num_heads), dropout=float(dropout), batch_first=True
        )

    def forward(self, left: Tensor, right: Tensor) -> tuple[Tensor, Tensor]:
        if left.shape != right.shape:
            raise ValueError(
                "left and right features must have identical shapes, got "
                f"{tuple(left.shape)} and {tuple(right.shape)}."
            )
        left_source = self.norm(left)
        right_source = self.norm(right)
        mask = causal_attention_mask(left.shape[1], left.device)

        # Both updates read the same original S_L/S_R.  Neither conditioned
        # result is fed into the opposite direction.
        left_update, _ = self.attention(
            left_source,
            right_source,
            right_source,
            attn_mask=mask,
            need_weights=False,
        )
        right_update, _ = self.attention(
            right_source,
            left_source,
            left_source,
            attn_mask=mask,
            need_weights=False,
        )
        return left + left_update, right + right_update
