"""MolmoAct2 single-resize inputs for the Molmo2 Hugging Face interface.

Keep the released SigLIP resize, normalization, patch extraction and pooling.
The HF interface otherwise assumes a global AND tiled view for every image;
adapt both its image markers and batching to a single view.
"""

from importlib import import_module
from types import MethodType
from typing import Any

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_transforms import convert_to_rgb
from transformers.image_utils import (
    make_flat_list_of_images,
    to_numpy_array,
)


def _preprocess(self: Any, images: Any, return_tensors=None, **kwargs) -> BatchFeature:
    if kwargs.get("return_pointing_metadata"):
        raise ValueError(
            "Single-resize action inputs do not support pointing metadata."
        )
    native = import_module(type(self).__module__)
    size = kwargs.get("size") or self.size
    patch_size = kwargs.get("patch_size") or self.patch_size
    pool_h, pool_w = kwargs.get("pooling_size") or self.pooling_size
    pixels, pooling, grids = [], [], []
    images = make_flat_list_of_images(self.fetch_images(images))
    for image in images:
        if kwargs.get("do_convert_rgb", self.do_convert_rgb):
            image = convert_to_rgb(image)
        image = to_numpy_array(image)
        # Same SigLIP bilinear resize (antialias=False) as the official
        # image_preprocessor.build_single_crop, without training augmentation.
        resized, patch_idx = native.build_resized_image(
            image,
            [size["height"], size["width"]],
            kwargs.get("resample") or self.resample,
            kwargs.get("image_mean") or self.image_mean,
            kwargs.get("image_std") or self.image_std,
            patch_size,
        )
        pooled = native.arange_for_pooling(patch_idx, pool_h, pool_w)
        height, width = pooled.shape[:2]
        pixels.append(native.batch_pixels_to_patches(resized, patch_size))
        pooling.append(pooled.reshape(-1, pool_h * pool_w))
        grids.append([height, width, 0, 0])
    return BatchFeature(
        data={
            "pixel_values": np.concatenate(pixels),
            "image_token_pooling": np.concatenate(pooling),
            "image_grids": np.asarray(grids, dtype=np.int64),
            "image_num_crops": np.ones(len(images), dtype=np.int64),
        },
        tensor_type=return_tensors,
    )


def _get_image_tokens(self: Any, image_grid: np.ndarray) -> np.ndarray:
    native = import_module(type(self).__module__)
    height, width = map(int, image_grid[:2])
    row = [native.IMAGE_PATCH_TOKEN] * width
    use_cols = self.use_single_crop_col_tokens
    if self.image_use_col_tokens if use_cols is None else use_cols:
        row.append(native.IM_COL_TOKEN)
    start = (
        native.LOW_RES_IMAGE_START_TOKEN
        if self.use_single_crop_start_token
        else native.IM_START_TOKEN
    )
    return np.asarray([start, *(row * height), native.IM_END_TOKEN])


def _build_batched_images(
    self: Any,
    input_ids,
    pixel_values,
    image_token_pooling,
    image_grids,
    image_num_crops,
):
    counts = (input_ids == self.config.image_end_token_id).sum(1).tolist()
    if sum(counts) != len(image_grids) or not torch.all(image_num_crops == 1):
        raise ValueError(
            "Single-resize inputs require one crop and one marker pair per image."
        )
    pooled_counts = image_grids[:, :2].prod(-1).tolist()
    if sum(pooled_counts) != len(image_token_pooling) or len(pixel_values) != sum(
        counts
    ):
        raise ValueError("Single-resize image pixels and pooling metadata disagree.")
    batch_size = len(counts)
    patches_per_crop = pixel_values.shape[1]
    images = pixel_values.new_full(
        (batch_size, max(counts), *pixel_values.shape[1:]), -1
    )
    per_sample_pooling = []
    image_offset = pool_offset = 0
    for sample_index, count in enumerate(counts):
        images[sample_index, :count] = pixel_values[image_offset : image_offset + count]
        entries = []
        for local_image in range(count):
            num = pooled_counts[image_offset + local_image]
            entry = image_token_pooling[pool_offset : pool_offset + num]
            entries.append(
                torch.where(entry >= 0, entry + local_image * patches_per_crop, entry)
            )
            pool_offset += num
        per_sample_pooling.append(
            torch.cat(entries) if entries else image_token_pooling[:0]
        )
        image_offset += count
    pooling = image_token_pooling.new_full(
        (batch_size, max(map(len, per_sample_pooling)), image_token_pooling.shape[-1]),
        -1,
    )
    for sample_index, entries in enumerate(per_sample_pooling):
        pooling[sample_index, : len(entries)] = entries
    return images, pooling


def configure_single_resize(processor: Any, vlm: Any | None = None) -> None:
    """Install instance-local adapters; never modify cached HF source or weights."""
    processor.image_processor.preprocess = MethodType(
        _preprocess, processor.image_processor
    )
    processor.get_image_tokens = MethodType(_get_image_tokens, processor)
    if vlm is not None:
        vlm.model.build_batched_images = MethodType(_build_batched_images, vlm.model)
