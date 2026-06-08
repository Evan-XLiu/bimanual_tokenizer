# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for praxis.core.normalizer."""

from __future__ import annotations

import torch

from praxis.core.normalizer import Normalize, Unnormalize
from praxis.core.types import NormMode


def _make_stats():
    return {
        "state": {
            "min": torch.tensor([0.0, -1.0]),
            "max": torch.tensor([10.0, 1.0]),
            "mean": torch.tensor([5.0, 0.0]),
            "std": torch.tensor([2.5, 0.5]),
        },
        "image": {
            "min": torch.zeros(3, 1, 1),
            "max": torch.ones(3, 1, 1),
            "mean": torch.full((3, 1, 1), 0.5),
            "std": torch.full((3, 1, 1), 0.25),
        },
    }


class TestNormalize:
    def test_min_max_maps_to_neg1_pos1(self):
        stats = _make_stats()
        modes = {"state": NormMode.MIN_MAX}
        norm = Normalize(stats, modes)

        batch = {"state": torch.tensor([[0.0, -1.0], [10.0, 1.0]])}
        out = norm(batch)
        assert out["state"][0].allclose(torch.tensor([-1.0, -1.0]))
        assert out["state"][1].allclose(torch.tensor([1.0, 1.0]))

    def test_min_max_midpoint_is_zero(self):
        stats = _make_stats()
        modes = {"state": NormMode.MIN_MAX}
        norm = Normalize(stats, modes)

        batch = {"state": torch.tensor([[5.0, 0.0]])}
        out = norm(batch)
        assert out["state"][0].allclose(torch.tensor([0.0, 0.0]))

    def test_mean_std(self):
        stats = _make_stats()
        modes = {"image": NormMode.MEAN_STD}
        norm = Normalize(stats, modes)

        batch = {"image": torch.full((1, 3, 4, 4), 0.5)}
        out = norm(batch)
        assert out["image"].allclose(torch.zeros(1, 3, 4, 4))

    def test_none_mode_passes_through(self):
        stats = _make_stats()
        modes = {"state": NormMode.NONE}
        norm = Normalize(stats, modes)

        batch = {"state": torch.tensor([[3.14]])}
        out = norm(batch)
        assert out["state"].allclose(batch["state"])

    def test_missing_key_passes_through(self):
        stats = _make_stats()
        modes = {"state": NormMode.MIN_MAX}
        norm = Normalize(stats, modes)

        batch = {"state": torch.ones(1, 2), "extra": torch.tensor([42.0])}
        out = norm(batch)
        assert out["extra"].item() == 42.0

    def test_non_tensor_values_are_ignored(self):
        stats = _make_stats()
        modes = {"state": NormMode.MIN_MAX}
        norm = Normalize(stats, modes)

        batch = {"state": None, "task": ["do task"]}
        out = norm(batch)
        assert out["state"] is None
        assert out["task"] == ["do task"]


class TestUnnormalize:
    def test_roundtrip(self):
        stats = _make_stats()
        modes = {"state": NormMode.MIN_MAX}
        norm = Normalize(stats, modes)
        unnorm = Unnormalize(stats, modes)

        original = torch.tensor([[3.0, 0.5], [7.0, -0.3]])
        batch = {"state": original.clone()}
        normalized = norm(batch)
        recovered = unnorm(normalized)
        assert recovered["state"].allclose(original, atol=1e-6)

    def test_roundtrip_mean_std(self):
        stats = _make_stats()
        modes = {"image": NormMode.MEAN_STD}
        norm = Normalize(stats, modes)
        unnorm = Unnormalize(stats, modes)

        original = torch.rand(2, 3, 4, 4)
        batch = {"image": original.clone()}
        recovered = unnorm(norm(batch))
        assert recovered["image"].allclose(original, atol=1e-6)

    def test_device_movement(self):
        """Verify normalizer buffers move with .to()."""
        stats = _make_stats()
        modes = {"state": NormMode.MIN_MAX}
        norm = Normalize(stats, modes)
        # Just check it doesn't error — CPU→CPU is the only portable test
        norm = norm.to("cpu")
        batch = {"state": torch.ones(1, 2)}
        out = norm(batch)
        assert out["state"].device.type == "cpu"

    def test_unnormalize_non_tensor_values_are_ignored(self):
        stats = _make_stats()
        modes = {"state": NormMode.MIN_MAX}
        unnorm = Unnormalize(stats, modes)

        batch = {"state": None, "task": ["do task"]}
        out = unnorm(batch)
        assert out["state"] is None
        assert out["task"] == ["do task"]
