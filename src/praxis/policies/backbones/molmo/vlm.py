# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Molmo2-ER loading and processor helpers shared by policy families."""

from __future__ import annotations

import logging
from types import MethodType
from typing import Any, cast

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForImageTextToText, AutoProcessor

from praxis.policies.backbones.molmo.single_resize import configure_single_resize
from praxis.policies.core.vlm.dtypes import parse_vlm_dtype

MOLMO2_ER_MODEL_NAME = "allenai/Molmo2-ER"


def _checkpointed_vision_forward(self: Any, x: Tensor) -> list[Tensor]:
    hidden_states = []
    for block in self.resblocks:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            x = checkpoint(block, x, use_reentrant=False)
        else:
            x = block(x)
        hidden_states.append(x)
    return hidden_states


def configure_molmo_vision_checkpointing(vlm: Any, enabled: bool) -> None:
    """Checkpoint complete ViT blocks while preserving intermediate features."""
    vision = getattr(getattr(vlm, "model", None), "vision_backbone", None)
    if vision is None:
        return
    transformer = vision.image_vit.transformer
    transformer.gradient_checkpointing = enabled
    transformer.forward = MethodType(_checkpointed_vision_forward, transformer)


def configure_molmo_vision_precision(vlm: Any, dtype: torch.dtype) -> None:
    """Keep every vision attention (including pooling) in the model dtype."""
    vision = vlm.model.vision_backbone
    vision.to(dtype=dtype)
    for module in vision.modules():
        if hasattr(module, "float32_attention"):
            module.float32_attention = dtype == torch.float32
    for name in ("vit_config", "adapter_config"):
        config = getattr(vlm.config, name, None)
        if config is not None:
            config.float32_attention = dtype == torch.float32

    # Molmo2's image pooling uses a one-token query with a dynamically sized
    # key sequence.  On Hopper, PyTorch 2.7's SDPA CUDA kernel can reject some
    # of these shapes with ``invalid configuration argument`` (especially at
    # larger batches).  The eager implementation is stable for this small
    # pooling operation; the main ViT and language-model attention remain SDPA.
    pooling = getattr(vision, "image_pooling_2d", None)
    if pooling is not None and hasattr(pooling, "attn_implementation"):
        pooling.attn_implementation = "eager"


def load_molmo_vlm(model_name: str, dtype_str: str) -> tuple[Any, Any]:
    """Load the remote-code Molmo2 model and processor.

    Molmo2 is published with an ``auto_map`` entry, so both model and
    processor need ``trust_remote_code=True`` for versions of Transformers
    that do not ship the Molmo2 classes yet.
    """
    dtype = parse_vlm_dtype(dtype_str)
    vlm = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    configure_molmo_vision_precision(vlm, dtype)
    processor = AutoProcessor.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,
    )
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.padding_side = "left"
    configure_single_resize(processor, vlm)
    logging.getLogger(__name__).info(
        "Molmo images: single resize to %s, one crop per camera; existing ViT weights retained.",
        processor.image_processor.size,
    )
    return vlm, processor


def freeze_molmo_vision(vlm: Any) -> None:
    """Freeze Molmo2's SigLIP vision backbone."""
    vision_backbone = getattr(cast(Any, vlm).model, "vision_backbone", None)
    if vision_backbone is not None:
        vision_backbone.requires_grad_(False)
        vision_backbone.eval()


def ensure_molmo_vision_eval(vlm: Any) -> None:
    """Keep the frozen Molmo2 vision backbone in eval mode."""
    vision_backbone = getattr(cast(Any, vlm).model, "vision_backbone", None)
    if vision_backbone is not None:
        vision_backbone.eval()


def run_molmo_processor(
    *,
    processor: Any,
    text: list[str] | str | None,
    image_batch: Tensor,
    channels_last: bool = False,
    padding: bool = True,
    return_tensors: str = "pt",
) -> Any:
    """Run the native Molmo2 processor on a batch-first image tensor.

    The processor accepts a list of images for each sample. Keeping that
    nesting mirrors the existing Pali/Qwen input contract and lets the
    processor construct ``image_token_pooling`` and crop metadata itself.
    """
    bsz = int(image_batch.shape[0])
    if text is not None and isinstance(text, list) and len(text) != bsz:
        raise ValueError(f"Text batch size mismatch: got {len(text)}, expected {bsz}.")
    nested_images = [
        [
            image_batch[i, j].permute(1, 2, 0).contiguous()
            if not channels_last
            and image_batch[i, j].ndim == 3
            and int(image_batch[i, j].shape[0]) in (1, 3, 4)
            else image_batch[i, j]
            for j in range(int(image_batch.shape[1]))
        ]
        for i in range(bsz)
    ]
    return processor(
        text=text,
        images=nested_images,
        padding=padding,
        return_tensors=return_tensors,
    )
