"""Rotary position embeddings for attention queries and keys."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

__all__ = ["RotaryPairing", "RotaryPosEmbed"]

RotaryPairing = Literal["interleaved", "split"]


class RotaryPosEmbed(nn.Module):
    """Apply RoPE to tensors shaped `[..., T, D]`."""

    def __init__(
        self,
        dim: int,
        *,
        rotary_dim: int | None = None,
        base: float = 10000.0,
        pairing: RotaryPairing = "interleaved",
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}.")
        rotary_dim = int(dim if rotary_dim is None else rotary_dim)
        if rotary_dim <= 0:
            raise ValueError(f"rotary_dim must be positive, got {rotary_dim}.")
        if rotary_dim > dim:
            raise ValueError(
                f"rotary_dim must be <= dim, got rotary_dim={rotary_dim}, dim={dim}."
            )
        if rotary_dim % 2 != 0:
            raise ValueError(f"rotary_dim must be even for RoPE, got {rotary_dim}.")
        if base <= 0:
            raise ValueError(f"base must be positive, got {base}.")
        if pairing not in ("interleaved", "split"):
            raise ValueError(f"Unsupported rotary pairing {pairing!r}.")
        self.dim = int(dim)
        self.rotary_dim = int(rotary_dim)
        self.base = float(base)
        self.pairing: RotaryPairing = pairing
        inv_freq = self.base ** (
            -torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)
            / float(self.rotary_dim)
        )
        self.inv_freq: torch.Tensor
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        offset: int = 0,
    ) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError(f"x must have at least 2 dims, got {tuple(x.shape)}.")
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"Expected last dim {self.dim}, got tensor shape {tuple(x.shape)}."
            )

        seq_len = int(x.shape[-2])
        if positions is None:
            positions = torch.arange(
                offset,
                offset + seq_len,
                device=x.device,
                dtype=torch.float32,
            )
        else:
            if positions.ndim not in (1, 2):
                raise ValueError(
                    "positions must be rank-1 [T] or rank-2 [B, T], "
                    f"got {tuple(positions.shape)}."
                )
            if int(positions.shape[-1]) != seq_len:
                raise ValueError(
                    f"positions length must match sequence length {seq_len}, "
                    f"got {tuple(positions.shape)}."
                )
            if positions.ndim == 2:
                if x.ndim < 3:
                    raise ValueError(
                        "rank-2 positions [B, T] require x shaped [B, ..., T, D]."
                    )
                if int(positions.shape[0]) != int(x.shape[0]):
                    raise ValueError(
                        "positions batch size must match x first dimension, got "
                        f"positions={tuple(positions.shape)} and x={tuple(x.shape)}."
                    )
            positions = positions.to(device=x.device, dtype=torch.float32)

        angles = positions[..., None] * self.inv_freq.to(device=x.device)[None, :]
        if positions.ndim == 1:
            view_shape = (1,) * (x.ndim - 2) + angles.shape
        else:
            view_shape = (angles.shape[0],) + (1,) * (x.ndim - 3) + angles.shape[1:]
        cos = angles.cos().view(view_shape)
        sin = angles.sin().view(view_shape)

        x_float = x[..., : self.rotary_dim].float()
        out = torch.empty_like(x_float)
        if self.pairing == "interleaved":
            x_even = x_float[..., 0::2]
            x_odd = x_float[..., 1::2]
            out[..., 0::2] = x_even * cos - x_odd * sin
            out[..., 1::2] = x_even * sin + x_odd * cos
        else:
            half = self.rotary_dim // 2
            x_first = x_float[..., :half]
            x_second = x_float[..., half:]
            out[..., :half] = x_first * cos - x_second * sin
            out[..., half:] = x_first * sin + x_second * cos
        out = out.to(dtype=x.dtype)
        if self.rotary_dim == self.dim:
            return out
        return torch.cat([out, x[..., self.rotary_dim :]], dim=-1)
