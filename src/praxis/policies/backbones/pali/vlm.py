# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""PaliGemma VLM loading and processor helpers shared by policy families."""

from __future__ import annotations

from typing import Any, Protocol, cast

from torch import Tensor
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from praxis.policies.core.vlm.dtypes import parse_vlm_dtype

PALIGEMMA_MODEL_NAME = "google/paligemma2-3b-mix-224"


class PaliTokenizerLike(Protocol):
    padding_side: str
    eos_token_id: int | None
    pad_token_id: int | None

    def get_vocab(self) -> dict[str, int]: ...

    def add_special_tokens(self, payload: dict[str, list[str]]) -> int: ...


class PaliProcessorLike(Protocol):
    tokenizer: PaliTokenizerLike
    image_token: str

    def __call__(
        self,
        *,
        text: list[str] | str | None,
        suffix: list[str] | str | None,
        images: list[list[Tensor]],
        padding: bool,
        return_tensors: str,
    ) -> Any: ...


def load_paligemma_vlm(
    model_name: str,
    dtype_str: str,
) -> tuple[Any, PaliProcessorLike]:
    """Load a PaliGemma model/processor pair with Praxis defaults."""
    dtype = parse_vlm_dtype(dtype_str)
    vlm = PaliGemmaForConditionalGeneration.from_pretrained(
        model_name,
        dtype=dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    processor = cast(
        PaliProcessorLike,
        AutoProcessor.from_pretrained(model_name, use_fast=True),
    )
    processor.tokenizer.padding_side = "left"
    return vlm, processor


def freeze_paligemma_vision(vlm: Any) -> None:
    vlm.vision_tower.requires_grad_(False)
    vlm.vision_tower.eval()


def ensure_paligemma_vision_eval(vlm: Any) -> None:
    vlm.vision_tower.eval()


def render_paligemma_texts(
    *,
    processor: PaliProcessorLike,
    instructions: list[str],
    action_token_strings: list[str] | None,
    bsz: int,
    num_images: int,
    state_texts: list[str] | None = None,
) -> list[str]:
    """Render PaliGemma text prompts with optional VLM state-prefix text."""
    image_token = getattr(processor, "image_token", "<image>")
    image_prefix = image_token * max(0, int(num_images))
    texts: list[str] = []
    for i in range(bsz):
        parts = [f"Task: {instructions[i]}."]
        if state_texts is not None and i < len(state_texts) and state_texts[i]:
            parts.append(f"{state_texts[i]}.")
        if action_token_strings is None:
            parts.append("Action: ")
        else:
            parts.append(f"Action: {action_token_strings[i]}")
        texts.append(f"{image_prefix}\n" + "\n".join(parts))
    return texts


def run_paligemma_processor(
    *,
    processor: PaliProcessorLike,
    text: list[str] | str | None,
    suffix: list[str] | str | None = None,
    image_batch: Tensor,
    padding: bool = True,
    return_tensors: str = "pt",
) -> Any:
    """Run the PaliGemma processor on a batch-first image tensor."""
    bsz = int(image_batch.shape[0])
    if text is not None and isinstance(text, list) and len(text) != bsz:
        raise ValueError(f"Text batch size mismatch: got {len(text)}, expected {bsz}.")
    if suffix is not None and isinstance(suffix, list) and len(suffix) != bsz:
        raise ValueError(
            f"Suffix batch size mismatch: got {len(suffix)}, expected {bsz}."
        )
    nested_images = [
        [image_batch[i, j] for j in range(int(image_batch.shape[1]))]
        for i in range(bsz)
    ]
    return processor(
        text=text,
        suffix=suffix,
        images=nested_images,
        padding=padding,
        return_tensors=return_tensors,
    )
