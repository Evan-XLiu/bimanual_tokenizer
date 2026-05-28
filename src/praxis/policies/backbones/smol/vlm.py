"""SmolVLM loading helpers shared by policy families."""

from __future__ import annotations

from typing import Any, cast

from transformers import AutoModelForImageTextToText, AutoProcessor

from praxis.policies.core.vlm.dtypes import parse_vlm_dtype

SMOLVLM_MODEL_NAME = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"


def load_smol_vlm(model_name: str, dtype_str: str) -> tuple[Any, Any]:
    """Load a SmolVLM model/processor pair with Praxis defaults."""
    dtype = parse_vlm_dtype(dtype_str)
    vlm = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.padding_side = "left"
    return vlm, processor


def freeze_smol_vision(vlm: Any) -> None:
    vision_model = cast(Any, vlm).model.vision_model
    vision_model.requires_grad_(False)
    vision_model.eval()


def ensure_smol_vision_eval(vlm: Any) -> None:
    cast(Any, vlm).model.vision_model.eval()
