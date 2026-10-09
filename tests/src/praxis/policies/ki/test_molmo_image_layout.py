"""Preserve native Molmo inputs while avoiding redundant image transposes."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoProcessor

from praxis.policies.backbones.molmo.single_resize import configure_single_resize
from praxis.policies.backbones.molmo.vlm import run_molmo_processor
from praxis.policies.core.observations import stack_observation_images
from praxis.policies.core.vlm.hf_chatprompt import render_chat_texts


@pytest.mark.parametrize("input_channels_last", [False, True])
@pytest.mark.parametrize("as_uint8", [False, True])
def test_camera_padding_and_history_match_old_layout(input_channels_last, as_uint8):
    features = [SimpleNamespace(key=key) for key in ("top", "wrist")]
    generator = torch.Generator().manual_seed(20)
    obs = {
        "top": torch.rand(2, 2, 5, 9, 3, generator=generator),
        "wrist": torch.rand(2, 2, 8, 10, 3, generator=generator),
    }
    if not input_channels_last:
        obs = {
            key: value.permute(0, 1, 4, 2, 3).contiguous() for key, value in obs.items()
        }
    old = stack_observation_images(obs, image_features=features, as_uint8=as_uint8)
    new = stack_observation_images(
        obs, image_features=features, as_uint8=as_uint8, channels_last=True
    )
    assert new.shape == (2, 2, 8, 10, 3)
    torch.testing.assert_close(new, old.permute(0, 1, 3, 4, 2), rtol=0, atol=0)


def test_hwc_processor_receives_original_image_views():
    images = torch.zeros(2, 3, 8, 10, 3, dtype=torch.uint8)

    def processor(**kwargs):
        for i, sample in enumerate(kwargs["images"]):
            for j, image in enumerate(sample):
                assert image.shape == (8, 10, 3)
                assert image.data_ptr() == images[i, j].data_ptr()
        return "processed"

    assert (
        run_molmo_processor(
            processor=processor,
            text=["a", "b"],
            image_batch=images,
            channels_last=True,
        )
        == "processed"
    )


def test_native_molmo_pixels_and_tokens_are_identical():
    model_path = Path(__file__).resolve().parents[5] / ".cache/models/Molmo2-ER"
    if not model_path.is_dir():
        pytest.skip("Requires local Molmo2-ER processor")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        processor = AutoProcessor.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True,
        )
        processor.tokenizer.padding_side = "left"
        features = [SimpleNamespace(key=k) for k in ("top", "left", "right")]
        generator = torch.Generator().manual_seed(7)
        obs = {
            key: torch.rand(1, height, 640, 3, generator=generator)
            for key, height in (("top", 360), ("left", 480), ("right", 480))
        }
        texts = render_chat_texts(
            processor=processor,
            num_images=3,
            state_texts=[""],
            instructions=["Move the box"],
            action_token_strings=None,
            bsz=1,
            include_system=False,
        )
        expected = run_molmo_processor(
            processor=processor,
            text=texts,
            image_batch=stack_observation_images(
                obs, image_features=features, as_uint8=True
            ),
        )
        actual = run_molmo_processor(
            processor=processor,
            text=texts,
            channels_last=True,
            image_batch=stack_observation_images(
                obs, image_features=features, as_uint8=True, channels_last=True
            ),
        )
        assert actual.keys() == expected.keys()
        assert actual["image_num_crops"].tolist() == [5, 5, 5]
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous_threads)


def test_single_resize_pixels_tokens_and_model_batching():
    model_path = Path(__file__).resolve().parents[5] / ".cache/models/Molmo2-ER"
    if not model_path.is_dir():
        pytest.skip("Requires local Molmo2-ER processor")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        processor = AutoProcessor.from_pretrained(
            str(model_path), trust_remote_code=True, local_files_only=True
        )
        processor.tokenizer.padding_side = "left"
        images = torch.randint(0, 256, (2, 3, 480, 640, 3), dtype=torch.uint8)
        texts = render_chat_texts(
            processor=processor,
            num_images=3,
            state_texts=["", ""],
            instructions=["Move the box", "Lift it"],
            action_token_strings=None,
            bsz=2,
            include_system=False,
        )
        native = run_molmo_processor(
            processor=processor, text=texts, image_batch=images, channels_last=True
        )
        model = SimpleNamespace(
            config=SimpleNamespace(
                image_end_token_id=processor.tokenizer.convert_tokens_to_ids("<im_end>")
            )
        )
        vlm = SimpleNamespace(model=model)
        configure_single_resize(processor, vlm)
        actual = run_molmo_processor(
            processor=processor, text=texts, image_batch=images, channels_last=True
        )
        assert actual["image_num_crops"].tolist() == [1] * 6
        assert actual["image_grids"].tolist() == [[14, 14, 0, 0]] * 6
        assert actual["pixel_values"].shape == (6, 729, 588)
        assert actual["image_token_pooling"].shape == (6 * 196, 4)
        # The single crop must equal the native full-image resize exactly;
        # no local tiles, altered normalization, or duplicate global image.
        torch.testing.assert_close(
            actual["pixel_values"], native["pixel_values"][::5], rtol=0, atol=0
        )
        end_id = model.config.image_end_token_id
        assert (actual["input_ids"] == end_id).sum(1).tolist() == [3, 3]
        patch_id = processor.tokenizer.convert_tokens_to_ids("<im_patch>")
        assert (actual["input_ids"] == patch_id).sum(1).tolist() == [588, 588]
        batched, pooling = model.build_batched_images(
            **{
                key: actual[key]
                for key in (
                    "input_ids",
                    "pixel_values",
                    "image_token_pooling",
                    "image_grids",
                    "image_num_crops",
                )
            }
        )
        assert batched.shape == (2, 3, 729, 588)
        assert pooling.shape == (2, 588, 4)
        for b in range(2):
            torch.testing.assert_close(
                batched[b], actual["pixel_values"][b * 3 : (b + 1) * 3]
            )
            for camera in range(3):
                idx = pooling[b, camera * 196 : (camera + 1) * 196]
                valid = idx[idx >= 0]
                assert valid.min() == camera * 729
                assert valid.max() == (camera + 1) * 729 - 1
    finally:
        torch.set_num_threads(previous_threads)
