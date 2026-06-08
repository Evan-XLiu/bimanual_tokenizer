# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""LoRA helpers for VLM policies."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, cast

import torch.nn as nn
from torch import Tensor


def vlm_ties_input_output_embeddings(vlm: nn.Module) -> bool:
    """Return whether a VLM shares its input and output embedding weights."""
    get_input = getattr(vlm, "get_input_embeddings", None)
    get_output = getattr(vlm, "get_output_embeddings", None)
    if not callable(get_input) or not callable(get_output):
        return False
    input_embeddings = get_input()
    output_embeddings = get_output()
    input_weight = getattr(input_embeddings, "weight", None)
    output_weight = getattr(output_embeddings, "weight", None)
    if not isinstance(input_weight, nn.Parameter) or not isinstance(
        output_weight, nn.Parameter
    ):
        return False
    return input_weight.data_ptr() == output_weight.data_ptr()


def resolve_lora_modules_to_save(
    *,
    vlm: nn.Module,
    modules_to_save: Iterable[str] | None,
    require_vocab_modules: bool = True,
) -> tuple[str, ...]:
    """Validate LoRA modules needed to preserve trainable vocab rows when required."""
    resolved = tuple(modules_to_save or ())
    if not require_vocab_modules:
        return resolved
    module_set = set(resolved)
    if vlm_ties_input_output_embeddings(vlm):
        if "embed_tokens" not in module_set:
            raise ValueError(
                "LoRA VLM policies with tied input/output embeddings must save "
                "the embedding action-token rows. policy.lora_modules_to_save "
                "must include 'embed_tokens'."
            )
        return tuple(module for module in resolved if module != "lm_head")

    required_modules_to_save = {"embed_tokens", "lm_head"}
    missing = required_modules_to_save - module_set
    if missing:
        raise ValueError(
            "LoRA VLM policies must save action-token vocabulary rows. "
            "policy.lora_modules_to_save must include "
            f"{sorted(required_modules_to_save)}; missing {sorted(missing)}."
        )
    return resolved


def apply_lora_to_vlm(
    *,
    vlm: nn.Module,
    lora_r: int,
    lora_alpha: int,
    lora_target_modules: Iterable[str],
    lora_dropout: float,
    lora_modules_to_save: Iterable[str] | None,
    require_vocab_modules: bool = True,
) -> nn.Module:
    """Apply a causal-LM LoRA adapter to a VLM."""
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=int(lora_r),
        lora_alpha=int(lora_alpha),
        target_modules=list(lora_target_modules),
        lora_dropout=float(lora_dropout),
        task_type="CAUSAL_LM",
        modules_to_save=list(
            resolve_lora_modules_to_save(
                vlm=vlm,
                modules_to_save=lora_modules_to_save,
                require_vocab_modules=require_vocab_modules,
            )
        ),
        ensure_weight_tying=True,
    )
    return cast(nn.Module, get_peft_model(cast(Any, vlm), lora_config))


def build_lora_optim_param_groups(
    *,
    named_parameters: Iterable[tuple[str, Tensor]],
    use_lora: bool,
    is_vocab_matrix_param_name: Callable[[str], bool],
) -> list[dict]:
    """Group optimizer params, excluding weight decay on vocab matrices under LoRA."""
    if not use_lora:
        params = [param for _name, param in named_parameters if param.requires_grad]
        return [{"params": params}] if params else []

    decay_params: list[Tensor] = []
    no_decay_params: list[Tensor] = []
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        if is_vocab_matrix_param_name(name):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    out: list[dict] = []
    if decay_params:
        out.append({"params": decay_params})
    if no_decay_params:
        out.append({"params": no_decay_params, "weight_decay": 0.0})
    return out
