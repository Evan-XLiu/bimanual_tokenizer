"""Tests for LeRobot adapter runtime video backend resolution."""

from __future__ import annotations

import importlib

import torch

from praxis.data.lerobot_adapter import LeRobotDatasetAdapter, _resolve_video_backend


class _TinyDataset:
    def __len__(self) -> int:
        return 5

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"x": torch.tensor(idx)}


def test_make_dataloader_drops_last_by_default() -> None:
    adapter = LeRobotDatasetAdapter.__new__(LeRobotDatasetAdapter)
    adapter.lerobot_dataset = _TinyDataset()

    loader = adapter.make_dataloader(
        batch_size=2,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
        use_episode_aware_sampler=False,
    )

    assert [batch["x"].tolist() for batch in loader] == [[0, 1], [2, 3]]


def test_make_dataloader_can_keep_partial_final_batch() -> None:
    adapter = LeRobotDatasetAdapter.__new__(LeRobotDatasetAdapter)
    adapter.lerobot_dataset = _TinyDataset()

    loader = adapter.make_dataloader(
        batch_size=2,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
        use_episode_aware_sampler=False,
        drop_last=False,
    )

    assert [batch["x"].tolist() for batch in loader] == [[0, 1], [2, 3], [4]]


def test_resolve_video_backend_prefers_torchcodec_when_runtime_import_works(
    monkeypatch,
) -> None:
    def _fake_import_module(name: str):
        assert name == "torchcodec"
        return object()

    monkeypatch.setattr(importlib, "import_module", _fake_import_module)

    assert _resolve_video_backend(None) == "torchcodec"


def test_resolve_video_backend_falls_back_to_pyav_when_torchcodec_import_fails(
    monkeypatch,
) -> None:
    def _fake_import_module(name: str):
        assert name == "torchcodec"
        raise RuntimeError("broken runtime deps")

    monkeypatch.setattr(importlib, "import_module", _fake_import_module)

    assert _resolve_video_backend(None) == "pyav"


def test_resolve_video_backend_respects_explicit_override() -> None:
    assert _resolve_video_backend("pyav") == "pyav"
