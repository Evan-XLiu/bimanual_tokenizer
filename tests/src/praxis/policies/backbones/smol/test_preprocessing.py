"""Tests for SmolVLA-style preprocessing helpers."""

from __future__ import annotations

import torch

from praxis.policies.backbones.smol.preprocessing import (
    expand_text_with_plain_image_tokens,
    preprocess_images,
)


class _DummyProcessor:
    image_token = "<image>"
    image_seq_len = 4

    class image_processor:
        image_mean = [0.5, 0.5, 0.5]
        image_std = [0.5, 0.5, 0.5]


def test_plain_image_expansion_omits_hf_wrapper_tokens():
    processor = _DummyProcessor()
    expanded = expand_text_with_plain_image_tokens(
        processor,
        ["prefix <image> middle <image> suffix"],
    )

    assert expanded == [
        "prefix <image><image><image><image> middle <image><image><image><image> suffix"
    ]
    assert "<fake_token_around_image>" not in expanded[0]
    assert "<global-img>" not in expanded[0]


def test_preprocess_images_shapes_and_padding_mask():
    processor = _DummyProcessor()
    image_batch = torch.ones((2, 1, 3, 5, 7), dtype=torch.float32)
    out = preprocess_images(
        processor,
        image_batch,
        resize_imgs_with_padding=(8, 8),
    )

    assert out["pixel_values"].shape == (2, 1, 3, 8, 8)
    assert out["pixel_values"].dtype == torch.float32
    assert out["pixel_attention_mask"].shape == (2, 1, 8, 8)
    assert out["pixel_attention_mask"].dtype == torch.bool
    assert out["rows"] == [[0], [0]]
    assert out["cols"] == [[0], [0]]
    assert torch.all(out["pixel_attention_mask"][:, :, :3, :] == 0)
    assert torch.all(out["pixel_attention_mask"][:, :, 3:, :] == 1)
    assert torch.allclose(
        out["pixel_values"][:, :, :, :3, :],
        torch.full_like(out["pixel_values"][:, :, :, :3, :], -1.0),
    )
    assert torch.allclose(
        out["pixel_values"][:, :, :, 3:, :],
        torch.ones_like(out["pixel_values"][:, :, :, 3:, :]),
    )
