# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Shared projection helpers for VLM language-model logits."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def project_hidden_to_vocab_logits(
    hidden: Tensor,
    *,
    output_embeddings: nn.Module | None,
    input_embeddings: nn.Module,
    candidate_token_ids: Tensor | None = None,
) -> Tensor:
    """Project hidden states through a full or candidate-only LM head."""
    if candidate_token_ids is None:
        if output_embeddings is None:
            return F.linear(hidden, _embedding_weight(input_embeddings))
        return output_embeddings(hidden)

    candidate_token_ids = candidate_token_ids.to(device=hidden.device, dtype=torch.long)
    linear = _as_linear(output_embeddings)
    if linear is not None:
        return _linear_candidate_projection(
            hidden,
            weight=linear.weight,
            bias=linear.bias,
            candidate_token_ids=candidate_token_ids,
        )

    if output_embeddings is None:
        return _linear_candidate_projection(
            hidden,
            weight=_embedding_weight(input_embeddings),
            bias=None,
            candidate_token_ids=candidate_token_ids,
        )

    return output_embeddings(hidden).index_select(-1, candidate_token_ids)


def _linear_candidate_projection(
    hidden: Tensor,
    *,
    weight: Tensor,
    bias: Tensor | None,
    candidate_token_ids: Tensor,
) -> Tensor:
    candidate_token_ids = candidate_token_ids.to(device=weight.device, dtype=torch.long)
    selected_weight = weight.index_select(0, candidate_token_ids)
    selected_bias = None if bias is None else bias.index_select(0, candidate_token_ids)
    return F.linear(hidden, selected_weight, selected_bias)


def _embedding_weight(module: nn.Module) -> Tensor:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, Tensor):
        raise TypeError(
            f"input_embeddings must expose a Tensor weight, got {type(module).__name__}."
        )
    return weight


def _as_linear(module: nn.Module | None) -> nn.Linear | None:
    if module is None:
        return None
    if isinstance(module, nn.Linear):
        return module

    active_saved = _active_module_to_save(module)
    if active_saved is not None and active_saved is not module:
        return _as_linear(active_saved)

    original = getattr(module, "original_module", None)
    if isinstance(original, nn.Module) and original is not module:
        return _as_linear(original)
    return None


def _active_module_to_save(module: nn.Module) -> nn.Module | None:
    modules_to_save = getattr(module, "modules_to_save", None)
    if not isinstance(modules_to_save, nn.ModuleDict) or not modules_to_save:
        return None

    active_adapter = getattr(module, "active_adapter", None)
    active_name: str | None = None
    if isinstance(active_adapter, str):
        active_name = active_adapter
    elif isinstance(active_adapter, (list, tuple)) and active_adapter:
        active_name = str(active_adapter[0])

    if active_name is not None and active_name in modules_to_save:
        return cast(nn.Module, modules_to_save[active_name])
    return cast(nn.Module, next(iter(modules_to_save.values())))
