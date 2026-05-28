"""FSDP wrap-boundary helpers for VLM policies."""

from __future__ import annotations

import torch.nn as nn


def select_vlm_block_modules(root: nn.Module) -> tuple[nn.Module, ...]:
    """Return parameterized HF-style blocks from ModuleList containers."""
    blocks: list[nn.Module] = []
    seen: set[int] = set()
    for module in root.modules():
        if not isinstance(module, nn.ModuleList):
            continue
        for child in module:
            if not isinstance(child, nn.Module):
                continue
            if type(child).forward is nn.Module.forward:
                continue
            if not any(True for _ in child.parameters(recurse=True)):
                continue
            marker = id(child)
            if marker in seen:
                continue
            seen.add(marker)
            blocks.append(child)
    return tuple(blocks)


def select_trainable_leaf_modules(root: nn.Module) -> tuple[nn.Module, ...]:
    """Return PEFT-style trainable leaves that should shard before blocks."""
    modules: list[nn.Module] = []
    seen: set[int] = set()
    for module in root.modules():
        if module is root:
            continue
        if any(True for _ in module.children()):
            continue
        if type(module).forward is nn.Module.forward:
            continue
        if not any(param.requires_grad for param in module.parameters(recurse=False)):
            continue
        marker = id(module)
        if marker in seen:
            continue
        seen.add(marker)
        modules.append(module)
    return tuple(modules)


def select_vlm_fsdp_wrap_modules(
    vlm: nn.Module,
    *,
    use_lora: bool,
) -> tuple[nn.Module, ...]:
    """Select VLM FSDP groups in bottom-up order."""
    blocks = select_vlm_block_modules(vlm)
    if not use_lora:
        return blocks
    return (*select_trainable_leaf_modules(vlm), *blocks)
