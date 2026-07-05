# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for action-token input preparation helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from praxis.policies.core.observations import (
    concatenate_state_features,
    extract_instructions,
    stack_observation_images,
)
from praxis.policies.core.state_prefix import (
    DEFAULT_STATE_PREFIX_TOKEN,
    prepare_state_prefix,
)

from ._test_utils import DummyFeature


def test_extract_instructions_handles_missing_and_string_batches():
    assert extract_instructions({}, language_key=None) is None
    assert extract_instructions({"lang": "pick up"}, language_key="lang") == ["pick up"]
    assert extract_instructions({"lang": ["a", "b"]}, language_key="lang") == ["a", "b"]


def test_prepare_state_prefix_formats_binned_text():
    obs = {"observation.state": torch.tensor([[-1.0, 1.0, 0.0]])}
    prepared = prepare_state_prefix(
        obs,
        state_features=[SimpleNamespace(key="observation.state", shape=(3,))],
        encoding="discrete_text",
        num_bins=5,
        placeholder_token=DEFAULT_STATE_PREFIX_TOKEN,
    )
    assert prepared.texts == ["State: 0 4 2"]


def test_concatenate_state_features_flattens_last_step_multidim_state():
    obs = {"observation.state": torch.arange(12).view(1, 2, 2, 3)}
    out = concatenate_state_features(
        obs,
        state_features=[SimpleNamespace(key="observation.state", shape=(2, 3))],
    )

    assert out is not None
    assert tuple(out.shape) == (1, 6)
    assert out.tolist() == [[6, 7, 8, 9, 10, 11]]


def test_concatenate_state_features_rejects_wrong_trailing_shape():
    with pytest.raises(ValueError, match="trailing shape"):
        concatenate_state_features(
            {"observation.state": torch.zeros(2, 4)},
            state_features=[SimpleNamespace(key="observation.state", shape=(3,))],
        )


def test_stack_observation_images_builds_bnchw_batch():
    obs = {
        "cam0": torch.tensor([[[[0.0]], [[0.5]], [[1.0]]]]).unsqueeze(1),
        "cam1": torch.tensor([[[[1.0]], [[0.0]], [[0.25]]]]).unsqueeze(1),
    }
    out = stack_observation_images(
        obs,
        image_features=[DummyFeature("cam0"), DummyFeature("cam1")],
        as_uint8=True,
    )
    assert tuple(out.shape) == (1, 2, 3, 1, 1)
    assert out.dtype == torch.uint8
    assert out[0, 0, :, 0, 0].tolist() == [0, 128, 255]
