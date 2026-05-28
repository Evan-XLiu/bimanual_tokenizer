"""Qwen3-VL loading and processor helpers shared by policy families."""

from __future__ import annotations

from typing import Any, cast

from torch import Tensor
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from praxis.policies.core.vlm.dtypes import parse_vlm_dtype

QWEN3_VL_MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"


def load_qwen_vlm(model_name: str, dtype_str: str) -> tuple[Any, Any]:
    """Load a Qwen3-VL model/processor pair with Praxis defaults."""
    dtype = parse_vlm_dtype(dtype_str)
    vlm = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    processor.tokenizer.padding_side = "left"
    return vlm, processor


def freeze_qwen_vision(vlm: Any) -> None:
    visual = cast(Any, vlm).visual
    visual.requires_grad_(False)
    visual.eval()


def ensure_qwen_vision_eval(vlm: Any) -> None:
    cast(Any, vlm).visual.eval()


def run_qwen_processor(
    *,
    processor: Any,
    text: list[str] | str | None,
    image_batch: Tensor,
    padding: bool = True,
    return_tensors: str = "pt",
) -> Any:
    """Run the native Qwen processor on a batch-first image tensor."""
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
