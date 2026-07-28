# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for LeRobot adapter runtime behavior."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pyarrow as pa
import torch

from praxis.data.lerobot_adapter import (
    LeRobotActionOnlyDatasetAdapter,
    LeRobotDatasetAdapter,
    _ActionColumnCache,
    _list_column_to_2d_float32,
    _resolve_video_backend,
)


class _TinyDataset:
    def __len__(self) -> int:
        return 5

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"x": torch.tensor(idx)}


def _action_only_adapter(cache: _ActionColumnCache) -> LeRobotActionOnlyDatasetAdapter:
    adapter = LeRobotActionOnlyDatasetAdapter.__new__(LeRobotActionOnlyDatasetAdapter)
    adapter.io = SimpleNamespace(action_feature=SimpleNamespace(key="action"))
    adapter._action_cache = cache
    return adapter


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


def test_action_cache_uses_identity_rows_for_full_dataset() -> None:
    actions = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    adapter = _action_only_adapter(
        _ActionColumnCache(
            actions=actions,
            episode_index=torch.zeros(6, dtype=torch.int64),
            abs_index=torch.arange(6, dtype=torch.int64),
            rel_of_abs=None,
            ep_from=torch.tensor([0]),
            ep_to=torch.tensor([6]),
            action_deltas=torch.tensor([-1, 0, 2]),
        )
    )

    item = adapter[2]

    assert torch.equal(item["action"], actions[[1, 2, 4]])
    assert torch.equal(item["action_is_pad"], torch.tensor([False, False, False]))


def test_action_cache_remaps_absolute_rows_for_filtered_episodes() -> None:
    actions = torch.tensor([[10.0], [11.0], [20.0], [21.0]])
    rel_of_abs = torch.full((22,), -1, dtype=torch.int64)
    rel_of_abs[torch.tensor([10, 11, 20, 21])] = torch.arange(4)
    adapter = _action_only_adapter(
        _ActionColumnCache(
            actions=actions,
            episode_index=torch.tensor([1, 1, 3, 3]),
            abs_index=torch.tensor([10, 11, 20, 21]),
            rel_of_abs=rel_of_abs,
            ep_from=torch.tensor([0, 10, 12, 20]),
            ep_to=torch.tensor([10, 12, 20, 22]),
            action_deltas=torch.tensor([-1, 0, 1]),
        )
    )

    item = adapter[3]

    assert torch.equal(item["action"], actions[[2, 3, 3]])
    assert torch.equal(item["action_is_pad"], torch.tensor([False, False, True]))


def test_action_cache_clamps_and_marks_episode_tail_padding() -> None:
    actions = torch.arange(6, dtype=torch.float32).unsqueeze(1)
    adapter = _action_only_adapter(
        _ActionColumnCache(
            actions=actions,
            episode_index=torch.zeros(6, dtype=torch.int64),
            abs_index=torch.arange(6, dtype=torch.int64),
            rel_of_abs=None,
            ep_from=torch.tensor([0]),
            ep_to=torch.tensor([6]),
            action_deltas=torch.tensor([0, 1, 2]),
        )
    )

    item = adapter[5]

    assert torch.equal(item["action"], actions[[5, 5, 5]])
    assert torch.equal(item["action_is_pad"], torch.tensor([False, True, True]))


def test_action_column_conversion_combines_arrow_chunks() -> None:
    action_type = pa.list_(pa.float32(), 2)
    column = pa.chunked_array(
        [
            pa.array([[1.0, 2.0], [3.0, 4.0]], type=action_type),
            pa.array([[5.0, 6.0]], type=action_type),
        ]
    )

    actual = _list_column_to_2d_float32(column, num_rows=3)

    assert torch.equal(
        torch.from_numpy(actual),
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    )


def test_action_cache_materialization_failure_uses_arrow_fallback(
    monkeypatch, caplog
) -> None:
    monkeypatch.setattr(
        LeRobotDatasetAdapter,
        "__init__",
        lambda self, *args, **kwargs: None,
    )
    monkeypatch.setattr(
        LeRobotActionOnlyDatasetAdapter,
        "_prune_hf_columns",
        lambda self: None,
    )

    def _raise_materialization_error(self) -> _ActionColumnCache:
        raise RuntimeError("cache unavailable")

    monkeypatch.setattr(
        LeRobotActionOnlyDatasetAdapter,
        "_materialize_action_cache",
        _raise_materialization_error,
    )

    adapter = LeRobotActionOnlyDatasetAdapter()
    expected = {"action": torch.tensor([[1.0]])}
    monkeypatch.setattr(adapter, "_getitem_arrow", lambda idx: expected)

    assert adapter[7] is expected
    assert adapter._action_cache is None
    assert "falling back to per-row Arrow access" in caplog.text
