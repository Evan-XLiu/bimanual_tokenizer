# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Training helpers shared by learned action autoencoders."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch.nn as nn
from torch import Tensor


def masked_autoencoder_loss(raw_loss: Tensor, pad_mask: Tensor | None) -> Tensor:
    """Reduce elementwise autoencoder loss with sample-level pad filtering.

    Codec training samples are fixed action chunks.  If any timestep in a
    chunk is padding, the current trainer contract excludes that whole chunk
    from the primary autoencoder loss.
    """
    if pad_mask is None:
        return raw_loss.mean()
    if raw_loss.ndim < 1:
        raise ValueError(
            f"raw_loss must include a batch dimension, got shape={tuple(raw_loss.shape)}."
        )
    batch_size = int(raw_loss.shape[0])
    if int(pad_mask.shape[0]) != batch_size:
        raise ValueError(
            "pad_mask batch size must match raw_loss batch size: "
            f"{int(pad_mask.shape[0])} != {batch_size}."
        )
    if not bool(pad_mask.any()):
        return raw_loss.mean()

    per_sample_loss = raw_loss.reshape(batch_size, -1).mean(dim=1)
    valid = ~pad_mask.reshape(batch_size, -1).any(dim=1)
    return (per_sample_loss * valid.float()).sum() / valid.float().sum().clamp(min=1)


def split_decay_param_groups(
    module: nn.Module,
    *,
    decay_weight_decay: float | None = None,
    no_decay_weight_decay: float = 0.0,
) -> list[dict[str, Any]]:
    """Split trainable parameters into rank-based decay/no-decay groups.

    Matrix-like tensors (`dim >= 2`) go in the decay group.  Biases, norms, and
    scalar parameters (`dim < 2`) go in the no-decay group.
    """
    decay_params = [p for p in module.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [
        p for p in module.parameters() if p.requires_grad and p.dim() < 2
    ]
    groups: list[dict[str, Any]] = []
    groups.extend(_make_param_group(decay_params, weight_decay=decay_weight_decay))
    groups.extend(
        _make_param_group(no_decay_params, weight_decay=no_decay_weight_decay)
    )
    return groups


def _make_param_group(
    params: Iterable[nn.Parameter],
    *,
    weight_decay: float | None,
) -> list[dict[str, Any]]:
    param_list = list(params)
    if not param_list:
        return []
    group: dict[str, Any] = {"params": param_list}
    if weight_decay is not None:
        group["weight_decay"] = float(weight_decay)
    return [group]
