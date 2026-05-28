"""Parameter-family grouping for VLM-backed policies."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Literal

from torch import Tensor

VlmParamFamily = Literal["vlm", "vision", "bridge", "action_expert"]


def build_vlm_param_groups(
    *,
    named_parameters: Iterable[tuple[str, Tensor]],
    use_lora: bool,
    is_vocab_matrix_param_name: Callable[[str], bool],
) -> list[dict]:
    """Group VLM policy params by train-recipe LR family and weight-decay lane."""
    grouped: dict[tuple[str | None, float | None], list[Tensor]] = {}
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        family = vlm_param_family(name)
        weight_decay = 0.0 if use_lora and is_vocab_matrix_param_name(name) else None
        grouped.setdefault((family, weight_decay), []).append(param)

    out: list[dict] = []
    for (family, weight_decay), params in grouped.items():
        group: dict[str, object] = {"params": params}
        if family is not None:
            group["family"] = family
        if weight_decay is not None:
            group["weight_decay"] = float(weight_decay)
        out.append(group)
    return out


def vlm_param_family(name: str) -> VlmParamFamily | None:
    if name.startswith("action_expert."):
        return "action_expert"
    if is_bridge_param_name(name):
        return "bridge"
    if is_vision_param_name(name):
        return "vision"
    if name.startswith("vlm."):
        return "vlm"
    return None


def is_bridge_param_name(name: str) -> bool:
    """Return whether a parameter bridges non-language inputs into the VLM lane."""
    if name.startswith("state_prefix_projector."):
        return True
    bridge_markers = (
        "multi_modal_projector",
        "image_connector",
        "vision_connector",
        "inputs_merger",
        "visual.merger",
    )
    return any(marker in name for marker in bridge_markers)


def is_vision_param_name(name: str) -> bool:
    return (
        name.startswith("vlm.vision_tower.")
        or ".vision_tower." in name
        or ".vision_model." in name
        or name.startswith("vlm.visual.")
    )
