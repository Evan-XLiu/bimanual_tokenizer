"""Tests for distributed LeRobot cache pre-sync helpers."""

from __future__ import annotations

from pathlib import Path

from praxis.data.lerobot_adapter import ensure_lerobot_dataset_available


class _FakeRuntime:
    def __init__(self, *, is_main_process: bool, num_processes: int) -> None:
        self.is_main_process = is_main_process
        self.num_processes = num_processes
        self.wait_calls = 0
        self.broadcast_inputs: list[tuple[object, int]] = []

    def wait_for_everyone(self) -> None:
        self.wait_calls += 1

    def broadcast_object(self, obj, *, src: int = 0):
        self.broadcast_inputs.append((obj, src))
        return obj


def test_ensure_lerobot_dataset_available_distributed_main_process_syncs(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[dict] = []
    sync_calls: list[dict] = []
    ready_calls: list[Path] = []

    class _DummyDataset:
        def __init__(self, **kwargs) -> None:
            calls.append(kwargs)

        def _ensure_hf_dataset_loaded(self) -> None:
            calls.append({"loaded": True})

    monkeypatch.setattr("praxis.data.lerobot_adapter.LeRobotDataset", _DummyDataset)
    monkeypatch.setattr(
        "praxis.data.lerobot_adapter._sync_lerobot_metadata",
        lambda **kwargs: sync_calls.append(kwargs),
    )
    monkeypatch.setattr(
        "praxis.data.lerobot_adapter._wait_for_lerobot_metadata_ready",
        lambda root: ready_calls.append(Path(root)),
    )

    acc = _FakeRuntime(is_main_process=True, num_processes=4)
    root = tmp_path / "cache"
    ensure_lerobot_dataset_available(
        repo_id="lerobot/libero",
        runtime=acc,
        root=root,
        episodes=[0, 1],
        download_videos=True,
    )

    assert acc.wait_calls == 0
    assert acc.broadcast_inputs == [(None, 0)]
    assert sync_calls == [{"repo_id": "lerobot/libero", "root": root}]
    assert ready_calls == [root]
    assert calls == [
        {
            "repo_id": "lerobot/libero",
            "root": root,
            "episodes": [0, 1],
            "force_cache_sync": False,
            "download_videos": True,
        },
        {"loaded": True},
    ]


def test_ensure_lerobot_dataset_available_distributed_non_main_process_waits_only(
    tmp_path: Path,
) -> None:
    acc = _FakeRuntime(is_main_process=False, num_processes=4)
    ensure_lerobot_dataset_available(
        repo_id="lerobot/libero",
        runtime=acc,
        root=tmp_path / "cache",
        episodes=None,
        download_videos=False,
    )

    assert acc.wait_calls == 0
    assert acc.broadcast_inputs == [(None, 0)]


def test_ensure_lerobot_dataset_available_distributed_main_process_keeps_force_sync_by_default_when_complete(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[dict] = []
    sync_calls: list[dict] = []
    root = tmp_path / "cache"
    episodes_dir = root / "meta" / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    for rel in (
        "meta/info.json",
        "meta/stats.json",
        "meta/tasks.parquet",
        "meta/episodes/chunk-000/file-000.parquet",
    ):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")

    class _DummyDataset:
        def __init__(self, **kwargs) -> None:
            calls.append(kwargs)

        def _ensure_hf_dataset_loaded(self) -> None:
            calls.append({"loaded": True})

    monkeypatch.setattr("praxis.data.lerobot_adapter.LeRobotDataset", _DummyDataset)
    monkeypatch.setattr(
        "praxis.data.lerobot_adapter._sync_lerobot_metadata",
        lambda **kwargs: sync_calls.append(kwargs),
    )

    acc = _FakeRuntime(is_main_process=True, num_processes=2)
    ensure_lerobot_dataset_available(
        repo_id="lerobot/libero",
        runtime=acc,
        root=root,
        episodes=None,
        download_videos=False,
    )

    assert acc.wait_calls == 0
    assert acc.broadcast_inputs == [(None, 0)]
    assert sync_calls == [{"repo_id": "lerobot/libero", "root": root}]
    assert calls == [
        {
            "repo_id": "lerobot/libero",
            "root": root,
            "episodes": None,
            "force_cache_sync": False,
            "download_videos": False,
        },
        {"loaded": True},
    ]


def test_ensure_lerobot_dataset_available_distributed_main_process_can_bypass_force_sync_when_complete(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[dict] = []
    sync_calls: list[dict] = []
    root = tmp_path / "cache"
    episodes_dir = root / "meta" / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    for rel in (
        "meta/info.json",
        "meta/stats.json",
        "meta/tasks.parquet",
        "meta/episodes/chunk-000/file-000.parquet",
    ):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")

    class _DummyDataset:
        def __init__(self, **kwargs) -> None:
            calls.append(kwargs)

        def _ensure_hf_dataset_loaded(self) -> None:
            calls.append({"loaded": True})

    monkeypatch.setattr("praxis.data.lerobot_adapter.LeRobotDataset", _DummyDataset)
    monkeypatch.setattr(
        "praxis.data.lerobot_adapter._sync_lerobot_metadata",
        lambda **kwargs: sync_calls.append(kwargs),
    )

    acc = _FakeRuntime(is_main_process=True, num_processes=2)
    ensure_lerobot_dataset_available(
        repo_id="lerobot/libero",
        runtime=acc,
        root=root,
        episodes=None,
        download_videos=False,
        bypass_force_cache_sync=True,
    )

    assert acc.wait_calls == 0
    assert acc.broadcast_inputs == [(None, 0)]
    assert sync_calls == []
    assert calls == [
        {
            "repo_id": "lerobot/libero",
            "root": root,
            "episodes": None,
            "force_cache_sync": False,
            "download_videos": False,
        },
        {"loaded": True},
    ]
