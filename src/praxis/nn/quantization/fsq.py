# --------------------------------------------------------------------
# Copyright (C) 2024-2025 EPFL & Apple Inc.
# Licensed under the EPFL-Apple Sample Code License (Non-Commercial)
#
# Modifications by Chaoqi Liu for research purposes.
# --------------------------------------------------------------------

import random
from typing import cast

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor, int32

__all__ = ["FiniteScalarQuantization"]


def _validate_probability(name: str, value: float) -> float:
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {probability}.")
    return probability


def round_ste_quant_dropout(z: Tensor, drop_quant_p: float) -> Tensor:
    """Round with straight through gradients, randomly skip quantization per sample."""
    zhat = z.round()
    batch_size = z.shape[0]
    device = z.device
    # Create a mask where each sample has a probability `drop_quant_p` to not be quantized
    mask = torch.bernoulli(torch.full((batch_size,), drop_quant_p, device=device))
    # Reshape mask to broadcast over the remaining dimensions
    mask = mask.view(batch_size, *([1] * (z.ndim - 1)))
    # Apply the mask: if mask=1, keep the original value; if mask=0, use the quantized value
    output = z + ((1 - mask) * (zhat - z)).detach()
    return output


class FiniteScalarQuantization(nn.Module):
    """Minimal finite-scalar quantization implementation.

    Args:
        levels: List of quantization levels.
        drop_quant_p: During training, pass the non-rounded values with this probability for each
            sample in the batch.
        corrupt_tokens_p: During training, optionally corrupt tokens by setting a percentage of them
            to random other tokens indices.
        min_corrupt_tokens_p: Optional argument specifying a minimum percentage of tokens to be
            corrupted. The actual percentage is sampled uniformly between the min and max per sample.
        apply_corrupt_tokens_p: Probability of activating token corruption per sample. Only active if
            corrupt_tokens_p > 0.
    """

    def __init__(
        self,
        levels: list[int],
        drop_quant_p: float = 0.0,
        corrupt_tokens_p: float = 0.0,
        min_corrupt_tokens_p: float | None = None,
        apply_corrupt_tokens_p: float = 0.2,
    ):
        super().__init__()

        if not levels:
            raise ValueError("levels must be non-empty.")
        if any(level < 2 for level in levels):
            raise ValueError(f"all levels must be >= 2, got levels={levels!r}.")

        drop_quant_p = _validate_probability("drop_quant_p", drop_quant_p)
        corrupt_tokens_p = _validate_probability("corrupt_tokens_p", corrupt_tokens_p)
        min_corrupt_tokens_p = (
            corrupt_tokens_p
            if min_corrupt_tokens_p is None
            else _validate_probability("min_corrupt_tokens_p", min_corrupt_tokens_p)
        )
        apply_corrupt_tokens_p = _validate_probability(
            "apply_corrupt_tokens_p", apply_corrupt_tokens_p
        )
        if min_corrupt_tokens_p > corrupt_tokens_p:
            raise ValueError(
                "min_corrupt_tokens_p must be <= corrupt_tokens_p, got "
                f"min_corrupt_tokens_p={min_corrupt_tokens_p}, "
                f"corrupt_tokens_p={corrupt_tokens_p}."
            )

        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent=False)

        self.dim = len(levels)
        self.codebook_size = int(self._levels.prod().item())

        implicit_codebook = self.indices_to_embedding(torch.arange(self.codebook_size))
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

        self.drop_quant_p = drop_quant_p
        self.corrupt_tokens_p = corrupt_tokens_p
        self.min_corrupt_tokens_p = min_corrupt_tokens_p
        self.apply_corrupt_tokens_p = apply_corrupt_tokens_p

    def __repr__(self):
        cls_name = self.__class__.__name__
        return (
            f"{cls_name}(\n"
            f"  levels={self._levels.tolist()!r},\n"
            f"  codebook_size={self.codebook_size!r},\n"
            f"  drop_quant_p={self.drop_quant_p!r},\n"
            ")"
        )

    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        bounded = self.bound(z)
        drop_quant_p = self.drop_quant_p if self.training else 0.0
        quantized = round_ste_quant_dropout(bounded, drop_quant_p)
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_embedding(
        self,
        indices: Tensor,
    ) -> Tensor:
        """Inverse of `codes_to_indices`."""
        indices = rearrange(indices, "... -> ... 1")
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)
        return codes

    def corrupt_quant(self, quant: Tensor) -> Tensor:
        """Randomly corrupt some entries of the quantized Tensor."""
        quant_shape, quant_device = quant.shape[:-1], quant.device
        random_indices = torch.randint(
            low=0, high=self.codebook_size, size=quant_shape, device=quant_device
        )
        random_quant = self.implicit_codebook[random_indices]
        sample_corrupt_tokens_p = random.uniform(
            self.min_corrupt_tokens_p, self.corrupt_tokens_p
        )
        corruption_mask = (
            torch.rand(quant_shape, device=quant_device) < sample_corrupt_tokens_p
        )
        corruption_mask = repeat(corruption_mask, "... -> ... d", d=quant.shape[-1])
        return torch.where(corruption_mask, random_quant, quant)

    @torch.autocast(device_type="cuda", enabled=False)
    def forward_z(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.LongTensor]:
        assert z.shape[-1] == self.dim, (
            f"expected dimension of {self.dim} but found dimension of {z.shape[-1]}"
        )

        quant = self.quantize(z.float())
        if (
            self.training
            and self.corrupt_tokens_p > 0.0
            and random.random() < self.apply_corrupt_tokens_p
        ):
            # Optionally corrupt a random percentage of tokens during training
            quant = self.corrupt_quant(quant)
        tokens = self.codes_to_indices(quant)

        # Incompatible return value type (got "Tensor", expected "LongTensor")  [return-value]
        return quant, cast(torch.LongTensor, tokens.long())

    @torch.compiler.disable
    def forward(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.LongTensor]:
        if isinstance(latents, list):
            raise TypeError(
                "FiniteScalarQuantization.forward expects a tensor shaped (B, L, D), not a list."
            )
        return self.forward_z(latents)

    _levels: Tensor
    _basis: Tensor
    implicit_codebook: Tensor
