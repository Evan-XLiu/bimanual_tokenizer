"""Learned absolute position embeddings."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

__all__ = ["LearnedPosEmbed"]

ScalingMode = Literal["absolute", "interpolate"]


class LearnedPosEmbed(nn.Module):
    """Learned absolute position embeddings shaped `[1, ..., D]`."""

    def __init__(
        self,
        dim: int,
        max_shape: Sequence[int],
        *,
        scaling: ScalingMode = "absolute",
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}.")
        if not max_shape or any(size <= 0 for size in max_shape):
            raise ValueError(f"max_shape must contain positive sizes, got {max_shape}.")
        if scaling not in ("absolute", "interpolate"):
            raise ValueError(f"Unsupported scaling mode {scaling!r}.")
        self.dim = int(dim)
        self.max_shape = tuple(int(size) for size in max_shape)
        self.scaling: ScalingMode = scaling
        self.pos_embed = nn.Parameter(torch.empty(1, *self.max_shape, self.dim))
        trunc_normal_(self.pos_embed, std=float(init_std))

    @torch.compiler.disable
    def forward(
        self,
        shape: Sequence[int],
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        shape = tuple(int(size) for size in shape)
        if len(shape) != len(self.max_shape):
            raise ValueError(
                f"shape must have {len(self.max_shape)} dims, got {shape}."
            )
        if any(size <= 0 for size in shape):
            raise ValueError(f"shape must contain positive sizes, got {shape}.")

        if self.scaling == "absolute":
            if any(
                size > max_size
                for size, max_size in zip(shape, self.max_shape, strict=True)
            ):
                raise ValueError(
                    f"Requested shape {shape} exceeds max_shape {self.max_shape}."
                )
            slices = (slice(None), *[slice(0, size) for size in shape], slice(None))
            pos_embed = self.pos_embed[slices]
        elif self.scaling == "interpolate":
            channel_first = self.pos_embed.movedim(-1, 1)
            if len(shape) == 1:
                pos_embed = F.interpolate(
                    channel_first, size=shape[0], mode="linear", align_corners=True
                )
            elif len(shape) == 2:
                pos_embed = F.interpolate(
                    channel_first, size=shape, mode="bilinear", align_corners=True
                )
            else:
                raise NotImplementedError(
                    "LearnedPosEmbed interpolation supports only 1D/2D shapes."
                )
            pos_embed = pos_embed.movedim(1, -1)
        else:
            raise AssertionError(f"Unhandled scaling mode {self.scaling!r}.")

        if device is not None or dtype is not None:
            return pos_embed.to(
                device=device or pos_embed.device,
                dtype=dtype or pos_embed.dtype,
            )
        return pos_embed
