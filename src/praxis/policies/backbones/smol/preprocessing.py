# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""SmolVLA-style preprocessing shared by Smol policy families."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers.feature_extraction_utils import BatchFeature


def run_processor(
    processor: Any,
    *,
    text: list[str] | str | None,
    image_batch: Tensor,
    resize_imgs_with_padding: Sequence[int] | None,
    padding: bool = True,
    return_tensors: str = "pt",
) -> Any:
    """Run the SmolVLA-style image path used by Smol action policies."""
    inputs: dict[str, object] = {}
    vision_inputs = preprocess_images(
        processor,
        image_batch,
        resize_imgs_with_padding=resize_imgs_with_padding,
    )
    vision_inputs.pop("rows", None)
    vision_inputs.pop("cols", None)
    inputs.update(vision_inputs)

    text_batch = _normalize_text_batch(text)
    if text_batch is not None:
        _validate_text_image_count(
            processor,
            text_batch,
            num_images=int(image_batch.shape[1]),
        )
        text_batch = expand_text_with_plain_image_tokens(processor, text_batch)
        text_inputs = processor.tokenizer(
            text_batch,
            padding=padding,
            return_tensors=return_tensors,
        )
        processor._check_special_mm_tokens(
            text_batch,
            text_inputs,
            modalities=["image"],
        )
        inputs.update(text_inputs)

    return BatchFeature(inputs, tensor_type=return_tensors)


def expand_text_with_plain_image_tokens(
    processor: Any,
    text: list[str],
) -> list[str]:
    """Use SmolVLA-style image blocks without fake/global wrapper tokens."""
    image_token = str(processor.image_token)
    image_seq_len = int(getattr(processor, "image_seq_len", 64))
    return [sample.replace(image_token, image_token * image_seq_len) for sample in text]


def preprocess_images(
    processor: Any,
    image_batch: Tensor,
    *,
    resize_imgs_with_padding: Sequence[int] | None,
) -> dict[str, object]:
    """Preprocess a ``(B, N, 3, H, W)`` Smol image batch on-device."""
    if image_batch.dim() != 5:
        raise TypeError(
            f"Expected rank-5 image tensor, got {tuple(image_batch.shape)}."
        )
    if image_batch.shape[2] != 3:
        raise TypeError(
            f"Expected 3-channel CHW images, got shape {tuple(image_batch.shape)}."
        )

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise TypeError("Smol processor missing image_processor.")
    target_width, target_height = _resolve_resize_size(resize_imgs_with_padding)

    bsz, num_images = int(image_batch.shape[0]), int(image_batch.shape[1])
    if num_images == 0:
        return _empty_image_outputs(
            batch_size=bsz,
            target_width=target_width,
            target_height=target_height,
            device=image_batch.device,
        )

    flat = image_batch.reshape(-1, 3, image_batch.shape[-2], image_batch.shape[-1])
    if flat.dtype == torch.uint8:
        flat = flat.to(dtype=torch.float32).mul_(1.0 / 255.0)
    else:
        flat = flat.to(dtype=torch.float32).clamp_(0.0, 1.0)

    flat, pixel_attention_mask = resize_with_pad(
        flat,
        width=target_width,
        height=target_height,
        pad_value=0.0,
    )
    flat = _normalize_pixels(flat, image_processor=image_processor)

    pixel_values = flat.view(
        bsz, num_images, 3, target_height, target_width
    ).contiguous()
    pixel_attention_mask = pixel_attention_mask.view(
        bsz, num_images, target_height, target_width
    ).contiguous()
    rows = [[0] * num_images for _ in range(bsz)]
    cols = [[0] * num_images for _ in range(bsz)]
    return {
        "pixel_values": pixel_values,
        "pixel_attention_mask": pixel_attention_mask,
        "rows": rows,
        "cols": cols,
    }


def resize_with_pad(
    image_batch: Tensor,
    *,
    width: int,
    height: int,
    pad_value: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Resize to fit inside ``(height, width)`` and pad on top/left."""
    if image_batch.dim() != 4:
        raise TypeError(
            f"Expected rank-4 image tensor, got {tuple(image_batch.shape)}."
        )

    cur_height, cur_width = int(image_batch.shape[2]), int(image_batch.shape[3])
    ratio = max(cur_width / float(width), cur_height / float(height))
    resized_height = max(1, int(cur_height / ratio))
    resized_width = max(1, int(cur_width / ratio))

    resized = F.interpolate(
        image_batch,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))
    padded = F.pad(resized, (pad_width, 0, pad_height, 0), value=pad_value)

    pixel_attention_mask = torch.zeros(
        (int(image_batch.shape[0]), height, width),
        dtype=torch.bool,
        device=image_batch.device,
    )
    pixel_attention_mask[:, pad_height:, pad_width:] = True
    return padded, pixel_attention_mask


def _normalize_text_batch(text: list[str] | str | None) -> list[str] | None:
    if text is None:
        return None
    if isinstance(text, str):
        return [text]
    if not isinstance(text, list):
        raise ValueError("Invalid input text. Expected a string or list of strings.")
    return text


def _validate_text_image_count(
    processor: Any,
    text: list[str],
    *,
    num_images: int,
) -> None:
    n_images_in_text = [sample.count(processor.image_token) for sample in text]
    if any(count != num_images for count in n_images_in_text):
        raise ValueError(
            f"The number of images in text {n_images_in_text} should be {num_images} for all samples."
        )


def _resolve_resize_size(
    resize_imgs_with_padding: Sequence[int] | None,
) -> tuple[int, int]:
    if resize_imgs_with_padding is None or len(resize_imgs_with_padding) != 2:
        raise TypeError("Smol config.resize_imgs_with_padding must be a 2-tuple/list.")
    return int(resize_imgs_with_padding[0]), int(resize_imgs_with_padding[1])


def _empty_image_outputs(
    *,
    batch_size: int,
    target_width: int,
    target_height: int,
    device: torch.device,
) -> dict[str, object]:
    pixel_values = torch.empty(
        (batch_size, 0, 3, target_height, target_width),
        dtype=torch.float32,
        device=device,
    )
    pixel_attention_mask = torch.empty(
        (batch_size, 0, target_height, target_width),
        dtype=torch.bool,
        device=device,
    )
    return {
        "pixel_values": pixel_values,
        "pixel_attention_mask": pixel_attention_mask,
        "rows": [[] for _ in range(batch_size)],
        "cols": [[] for _ in range(batch_size)],
    }


def _normalize_pixels(image_batch: Tensor, *, image_processor: Any) -> Tensor:
    mean = torch.as_tensor(
        image_processor.image_mean,
        device=image_batch.device,
        dtype=image_batch.dtype,
    )
    std = torch.as_tensor(
        image_processor.image_std,
        device=image_batch.device,
        dtype=image_batch.dtype,
    )
    if mean.numel() == 1:
        mean = mean.repeat(3)
    if std.numel() == 1:
        std = std.repeat(3)
    return (image_batch - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
