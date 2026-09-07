# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Paired finite-scalar quantization for bimanual OAT."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from praxis.nn.quantization.fsq import FiniteScalarQuantization


class PairedFiniteScalarQuantization(nn.Module):
    """Quantize two streams and combine their indices into one token ID."""

    def __init__(
        self,
        *,
        common_levels: tuple[int, ...],
        relative_levels: tuple[int, ...],
        drop_quant_p: float = 0.0,
        corrupt_tokens_p: float = 0.0,
        min_corrupt_tokens_p: float | None = None,
        apply_corrupt_tokens_p: float = 0.2,
    ) -> None:
        super().__init__()
        kwargs = {
            "drop_quant_p": drop_quant_p,
            "corrupt_tokens_p": corrupt_tokens_p,
            "min_corrupt_tokens_p": min_corrupt_tokens_p,
            "apply_corrupt_tokens_p": apply_corrupt_tokens_p,
        }
        self.common_quantizer = FiniteScalarQuantization(
            levels=list(common_levels), **kwargs
        )
        self.relative_quantizer = FiniteScalarQuantization(
            levels=list(relative_levels), **kwargs
        )
        self.common_codebook_size = int(self.common_quantizer.codebook_size)
        self.relative_codebook_size = int(self.relative_quantizer.codebook_size)
        self.codebook_size = self.common_codebook_size * self.relative_codebook_size

    def pair_tokens(self, common_tokens: Tensor, relative_tokens: Tensor) -> Tensor:
        if common_tokens.shape != relative_tokens.shape:
            raise ValueError(
                "common and relative token tensors must have identical shapes, got "
                f"{tuple(common_tokens.shape)} and {tuple(relative_tokens.shape)}."
            )
        return common_tokens + self.common_codebook_size * relative_tokens

    def unpair_tokens(self, paired_tokens: Tensor) -> tuple[Tensor, Tensor]:
        if paired_tokens.numel() and (
            bool((paired_tokens < 0).any())
            or bool((paired_tokens >= self.codebook_size).any())
        ):
            raise ValueError(f"paired token IDs must be in [0, {self.codebook_size}).")
        return (
            paired_tokens % self.common_codebook_size,
            paired_tokens // self.common_codebook_size,
        )

    def forward(
        self, common_latents: Tensor, relative_latents: Tensor
    ) -> tuple[Tensor, Tensor]:
        common, common_tokens = self.common_quantizer(common_latents)
        relative, relative_tokens = self.relative_quantizer(relative_latents)
        quantized = torch.cat((common, relative), dim=-1)
        return quantized, self.pair_tokens(common_tokens, relative_tokens)

    def indices_to_embedding(self, paired_tokens: Tensor) -> Tensor:
        common_tokens, relative_tokens = self.unpair_tokens(paired_tokens)
        common = self.common_quantizer.indices_to_embedding(common_tokens)
        relative = self.relative_quantizer.indices_to_embedding(relative_tokens)
        return torch.cat((common, relative), dim=-1)
