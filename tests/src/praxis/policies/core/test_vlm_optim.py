# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for VLM policy optimizer parameter-family grouping."""

from __future__ import annotations

import torch
import torch.nn as nn

from praxis.policies.core.vlm.optim import build_vlm_param_groups


def _param_families(groups: list[dict]) -> dict[int, str | None]:
    out: dict[int, str | None] = {}
    for group in groups:
        family = group.get("family")
        for param in group["params"]:
            out[id(param)] = None if family is None else str(family)
    return out


def test_vlm_optim_groups_parameter_families_without_lrs() -> None:
    vlm = nn.Parameter(torch.ones(1))
    vision = nn.Parameter(torch.ones(1))
    connector = nn.Parameter(torch.ones(1))
    image_connector = nn.Parameter(torch.ones(1))
    vision_connector = nn.Parameter(torch.ones(1))
    inputs_merger = nn.Parameter(torch.ones(1))
    qwen_merger = nn.Parameter(torch.ones(1))
    state_prefix = nn.Parameter(torch.ones(1))
    expert = nn.Parameter(torch.ones(1))

    groups = build_vlm_param_groups(
        named_parameters=[
            ("vlm.language_model.layers.0.self_attn.q_proj.weight", vlm),
            ("vlm.vision_tower.encoder.layers.0.weight", vision),
            ("vlm.multi_modal_projector.linear.weight", connector),
            ("vlm.image_connector.linear.weight", image_connector),
            ("vlm.vision_connector.linear.weight", vision_connector),
            ("vlm.model.inputs_merger.gate.weight", inputs_merger),
            ("vlm.visual.merger.mlp.0.weight", qwen_merger),
            ("state_prefix_projector.weight", state_prefix),
            ("action_expert.layers.0.weight", expert),
        ],
        use_lora=False,
        is_vocab_matrix_param_name=lambda _name: False,
    )

    families = _param_families(groups)
    assert families[id(vlm)] == "vlm"
    assert families[id(vision)] == "vision"
    assert families[id(connector)] == "bridge"
    assert families[id(image_connector)] == "bridge"
    assert families[id(vision_connector)] == "bridge"
    assert families[id(inputs_merger)] == "bridge"
    assert families[id(qwen_merger)] == "bridge"
    assert families[id(state_prefix)] == "bridge"
    assert families[id(expert)] == "action_expert"
    assert all("lr" not in group for group in groups)


def test_vlm_optim_keeps_lora_vocab_rows_without_weight_decay() -> None:
    vocab = nn.Parameter(torch.ones(1))
    other = nn.Parameter(torch.ones(1))

    groups = build_vlm_param_groups(
        named_parameters=[
            ("vlm.lm_head.weight", vocab),
            ("vlm.language_model.layers.0.mlp.up_proj.weight", other),
        ],
        use_lora=True,
        is_vocab_matrix_param_name=lambda name: name.endswith("lm_head.weight"),
    )

    weight_decay_by_param = {
        id(param): group.get("weight_decay")
        for group in groups
        for param in group["params"]
    }
    assert weight_decay_by_param[id(vocab)] == 0.0
    assert weight_decay_by_param[id(other)] is None
