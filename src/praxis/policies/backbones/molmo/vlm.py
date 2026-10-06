# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Molmo2-ER loading and processor helpers shared by policy families."""

from __future__ import annotations

from typing import Any, cast

from torch import Tensor
from transformers import AutoModelForImageTextToText, AutoProcessor

from praxis.policies.core.vlm.dtypes import parse_vlm_dtype

MOLMO2_ER_MODEL_NAME = "allenai/Molmo2-ER"


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
    processor = AutoProcessor.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,
    )
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.padding_side = "left"
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
        [image_batch[i, j] for j in range(int(image_batch.shape[1]))]
        for i in range(bsz)
    ]
    return processor(
        text=text,
        images=nested_images,
        padding=padding,
        return_tensors=return_tensors,
    )
