# --------------------------------------------------------------------
# Copyright (C) 2024-2025 EPFL & Apple Inc.
# Licensed under the EPFL-Apple Sample Code License (Non-Commercial)
#
# Modifications by Chaoqi Liu for research purposes.
# --------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Fp32LayerNorm", "Fp32RMSNorm"]


class Fp32LayerNorm(nn.LayerNorm):
    """
    Mixed precision friendly LayerNorm.
    From torchtune.modules.Fp32LayerNorm.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class Fp32RMSNorm(nn.Module):
    """Mixed precision friendly RMSNorm.

    RMSNorm scales by the root-mean-square of activations and optionally applies
    a learned affine weight.
    """

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        eps: float = 1e-6,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        else:
            self.register_parameter("weight", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input.float()
        dims = tuple(range(-len(self.normalized_shape), 0))
        rms = torch.sqrt(torch.mean(x.pow(2), dim=dims, keepdim=True) + self.eps)
        out = x / rms
        if self.weight is not None:
            out = out * self.weight.float()
        return out.type_as(input)
