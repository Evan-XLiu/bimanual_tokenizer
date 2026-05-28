"""Prefix-code helpers for prefix-aware action codecs."""

from __future__ import annotations

import torch
from torch import Tensor


def pad_code_sequence(
    codes: Tensor, max_seq_len: int, *, pad_value: float = 0.0
) -> Tensor:
    """Pad a code tensor along its sequence dimension."""
    if codes.ndim < 2:
        raise ValueError(f"codes must have rank >= 2, got shape={tuple(codes.shape)}.")
    pad_len = int(max_seq_len) - int(codes.shape[1])
    if pad_len <= 0:
        return codes
    pad_shape = list(codes.shape)
    pad_shape[1] = pad_len
    pad_seq = torch.full(
        tuple(pad_shape),
        fill_value=pad_value,
        device=codes.device,
        dtype=codes.dtype,
    )
    return torch.cat([codes, pad_seq], dim=1)
