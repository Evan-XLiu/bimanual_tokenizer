# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Prefix-token helpers for OAT-family token codecs."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from praxis.codecs.core import pad_code_sequence


def prefix_token_ids_to_tensor(
    tokens: Tensor | Sequence[Sequence[int]],
    *,
    token_horizon: int,
    device: torch.device | None,
    pad_value: int = 0,
) -> tuple[Tensor, list[int]]:
    """Normalize tensor/list token prefixes to a padded rank-2 token tensor."""
    if isinstance(tokens, Tensor):
        if tokens.ndim != 2:
            raise ValueError(
                f"tokens must be rank-2 [B, L], got shape={tuple(tokens.shape)}."
            )
        if int(tokens.shape[1]) > int(token_horizon):
            raise ValueError(
                f"token length mismatch: expected <= {token_horizon}, got {int(tokens.shape[1])}."
            )
        token_lens = [int(tokens.shape[1])] * int(tokens.shape[0])
        token_tensor = tokens.long()
        if device is not None:
            token_tensor = token_tensor.to(device=device)
        return (
            pad_code_sequence(
                token_tensor,
                token_horizon,
                pad_value=pad_value,
            ),
            token_lens,
        )

    token_tensors = []
    token_lens = []
    for seq in tokens:
        seq_tensor = torch.as_tensor(list(seq), dtype=torch.long, device=device)
        if int(seq_tensor.shape[0]) > int(token_horizon):
            raise ValueError(
                f"token length mismatch: expected <= {token_horizon}, got {int(seq_tensor.shape[0])}."
            )
        seq_tensor = seq_tensor.unsqueeze(0)
        token_tensors.append(
            pad_code_sequence(
                seq_tensor,
                token_horizon,
                pad_value=pad_value,
            )
        )
        token_lens.append(int(seq_tensor.shape[1]))

    if not token_tensors:
        raise ValueError("decode received an empty token sequence list.")
    return torch.cat(token_tensors, dim=0), token_lens
