# --------------------------------------------------------------------
# Copyright (C) 2024-2025 EPFL & Apple Inc.
# Licensed under the EPFL-Apple Sample Code License (Non-Commercial)
#
# Modifications by Chaoqi Liu for research purposes.
# --------------------------------------------------------------------

import torch.nn as nn
from torch import Tensor

__all__ = ["Mlp", "GatedMlp"]


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class GatedMlp(nn.Module):
    """Implements SwiGLU and other gated feed-forward layers."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.SiLU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        # If gated, multiply hidden_dim by 2/3 to account for extra matmul
        hidden_features = int(2 * (hidden_features or in_features) / 3)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.fc3 = nn.Linear(in_features, hidden_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x)) * self.fc3(x)))
