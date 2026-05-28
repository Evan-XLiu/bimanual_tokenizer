"""Tests for normalizer runtime persistence behavior."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from torch import Tensor

from praxis.core.normalizer import Normalize, Unnormalize
from praxis.core.types import NormMode


def _action_stats() -> dict[str, dict[str, Tensor]]:
    return {
        "action": {
            "min": torch.tensor([0.0, -1.0, 0.5]),
            "max": torch.tensor([10.0, 1.0, 5.0]),
        }
    }


def _obs_stats() -> dict[str, dict[str, Tensor]]:
    return {
        "observation.state": {
            "mean": torch.tensor([1.0, 2.0]),
            "std": torch.tensor([0.5, 0.3]),
        }
    }


class TestNormalizerRuntime:
    """Normalize/Unnormalize are pure runtime modules with file-stable stats."""

    def test_normalize_no_buffers(self):
        norm = Normalize(_action_stats(), {"action": NormMode.MIN_MAX})
        assert len(norm.state_dict()) == 0

    def test_unnormalize_no_buffers(self):
        unnorm = Unnormalize(_action_stats(), {"action": NormMode.MIN_MAX})
        assert len(unnorm.state_dict()) == 0

    def test_normalize_roundtrip_via_file(self):
        stats = _action_stats()
        norm = Normalize(stats, {"action": NormMode.MIN_MAX})
        before = norm({"action": torch.tensor([[5.0, 0.0, 2.5]])})["action"]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "norm_stats.pt"
            torch.save({"stats": stats, "modes": {"action": "min_max"}}, path)
            data = torch.load(path, map_location="cpu", weights_only=True)

        loaded_modes = {k: NormMode(v) for k, v in data["modes"].items()}
        norm2 = Normalize(data["stats"], loaded_modes)
        after = norm2({"action": torch.tensor([[5.0, 0.0, 2.5]])})["action"]
        torch.testing.assert_close(before, after)

    def test_dotted_key_works(self):
        norm = Normalize(_obs_stats(), {"observation.state": NormMode.MEAN_STD})
        out = norm({"observation.state": torch.tensor([[3.0, 4.0]])})[
            "observation.state"
        ]
        assert out.shape == (1, 2)
