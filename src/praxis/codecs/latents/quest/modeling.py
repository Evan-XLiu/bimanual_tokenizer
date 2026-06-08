# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""QueST-family temporal convolution modeling blocks."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class CausalConv1d(nn.Module):
    """Causal 1D convolution used by the QueST temporal encoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        stride: int,
        no_pad: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        if no_pad:
            self.padding = 0
        else:
            self.padding = int(dilation) * (self.kernel_size - 1)
        self.stride = int(stride)
        self.conv = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            padding=self.padding,
            dilation=int(dilation),
            stride=self.stride,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        last_n = (2 * self.padding - self.kernel_size) // self.stride + 1
        if last_n > 0:
            return x[:, :, :-last_n]
        return x


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish."""

    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        n_groups: int = 4,
        causal: bool = True,
        no_pad: bool = False,
    ) -> None:
        super().__init__()
        if causal:
            conv = CausalConv1d(
                inp_channels,
                out_channels,
                kernel_size,
                dilation=1,
                stride=stride,
                no_pad=no_pad,
            )
        else:
            conv = nn.Conv1d(
                inp_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
                stride=stride,
            )

        self.block = nn.Sequential(
            conv,
            Rearrange("batch channels horizon -> batch channels 1 horizon"),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange("batch channels 1 horizon -> batch channels horizon"),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualTemporalBlock(nn.Module):
    """Stacked temporal convolution block used by QueST."""

    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | None = None,
        stride: Sequence[int] | None = None,
        n_groups: int = 8,
        causal: bool = True,
        residual: bool = False,
        pooling_layers: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if kernel_size is None:
            kernel_size = [5, 3]
        if stride is None:
            stride = [2, 2]
        if pooling_layers is None:
            pooling_layers = []
        self.pooling_layers = tuple(int(v) for v in pooling_layers)
        self.blocks = nn.ModuleList()
        for i in range(len(kernel_size)):
            block = Conv1dBlock(
                inp_channels if i == 0 else out_channels,
                out_channels,
                int(kernel_size[i]),
                int(stride[i]),
                n_groups=n_groups,
                causal=causal,
            )
            self.blocks.append(block)
        if residual:
            if out_channels == inp_channels and int(stride[0]) == 1:
                self.residual_conv = nn.Identity()
            else:
                residual_stride = 1
                for stride_value in stride:
                    residual_stride *= int(stride_value)
                self.residual_conv = nn.Conv1d(
                    inp_channels,
                    out_channels,
                    kernel_size=1,
                    stride=residual_stride,
                )
        if pooling_layers:
            self.pooling = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.transpose(x, 1, 2)
        out = x
        for layer_num, block in enumerate(self.blocks):
            out = block(out)
            if hasattr(self, "pooling") and layer_num in self.pooling_layers:
                out = self.pooling(out)
        if hasattr(self, "residual_conv"):
            out = out + self.residual_conv(x)
        return torch.transpose(out, 1, 2)
