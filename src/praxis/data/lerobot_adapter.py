# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Thin adapter around LeRobotDataset for our training pipeline."""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep
from typing import Any, cast

import numpy as np
import pyarrow as pa
import torch
from huggingface_hub import snapshot_download
from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import get_safe_version, is_valid_version
from lerobot.utils.constants import HF_LEROBOT_HOME
from torch.utils.data import DataLoader, Dataset

from praxis.core.types import NormMode, PolicyIO
from praxis.data.utils import resolve_normalization_modes_for_io, stats_to_tensors

logger = logging.getLogger(__name__)

_METADATA_READY_TIMEOUT_SECONDS = 120.0
_METADATA_READY_POLL_SECONDS = 0.25


def ensure_lerobot_dataset_available(
    *,
    repo_id: str,
    runtime: Any | None = None,
    root: str | Path | None = None,
    episodes: list[int] | None = None,
    download_videos: bool,
    bypass_force_cache_sync: bool = False,
) -> None:
    """Pre-sync a LeRobot dataset cache before distributed consumers fan out.

    Distributed launch can start multiple ranks that all try to populate the same
    LeRobot cache simultaneously. This helper pre-downloads on one process so
    direct trainer entry points get the same protection.
    """

    root_path = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
    metadata_incomplete = _lerobot_metadata_incomplete(root_path)
    is_distributed = bool(getattr(runtime, "num_processes", 1) > 1)
    needs_sync = metadata_incomplete or is_distributed
    if not needs_sync:
        return

    # Default behavior:
    # - refresh metadata in distributed startup
    # - refresh metadata when metadata is incomplete
    # Optional escape hatch:
    # - bypass_force_cache_sync=True disables the distributed-only metadata refresh
    #   to avoid hangs in problematic cache-sync environments.
    refresh_metadata = metadata_incomplete or (
        is_distributed and not bypass_force_cache_sync
    )

    def _sync_on_main_rank() -> None:
        if refresh_metadata:
            _sync_lerobot_metadata(repo_id=repo_id, root=root_path)
            _wait_for_lerobot_metadata_ready(root_path)
        dataset = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            force_cache_sync=False,
            download_videos=download_videos,
        )
        dataset._ensure_hf_dataset_loaded()

    is_main_process = bool(getattr(runtime, "is_main_process", True))
    if runtime is not None and is_distributed:
        main_error = None
        if is_main_process:
            try:
                _sync_on_main_rank()
            except BaseException as exc:
                main_error = (
                    "LeRobot dataset cache sync failed on rank 0: "
                    f"{type(exc).__name__}: {exc}"
                )
        broadcast_object = getattr(runtime, "broadcast_object", None)
        if callable(broadcast_object):
            main_error = broadcast_object(main_error, src=0)
            if main_error is not None:
                raise RuntimeError(str(main_error))
            return
        if main_error is not None:
            raise RuntimeError(str(main_error))
        runtime.wait_for_everyone()
        return

    if is_main_process:
        _sync_on_main_rank()


def _lerobot_metadata_incomplete(root: Path) -> bool:
    return bool(_missing_lerobot_metadata_paths(root))


def _missing_lerobot_metadata_paths(root: Path) -> list[Path]:
    meta_dir = root / "meta"
    required_paths = (
        meta_dir / "info.json",
        meta_dir / "stats.json",
        meta_dir / "tasks.parquet",
    )
    missing = [path for path in required_paths if not path.exists()]

    episodes_dir = meta_dir / "episodes"
    if not episodes_dir.exists():
        missing.append(episodes_dir)
    elif not any(episodes_dir.rglob("*.parquet")):
        missing.append(episodes_dir / "*.parquet")
    return missing


def _sync_lerobot_metadata(*, repo_id: str, root: Path) -> None:
    revision = CODEBASE_VERSION
    if is_valid_version(revision):
        revision = get_safe_version(repo_id, revision)
    root.mkdir(exist_ok=True, parents=True)
    snapshot_download(
        repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=root,
        allow_patterns=["meta/**"],
    )


def _wait_for_lerobot_metadata_ready(root: Path) -> None:
    deadline = monotonic() + _METADATA_READY_TIMEOUT_SECONDS
    while True:
        missing = _missing_lerobot_metadata_paths(root)
        if not missing:
            return
        if monotonic() >= deadline:
            missing_text = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(
                f"LeRobot metadata cache at {root} is still incomplete after "
                f"{_METADATA_READY_TIMEOUT_SECONDS:.0f}s; missing {missing_text}."
            )
        sleep(_METADATA_READY_POLL_SECONDS)


def _resolve_video_backend(video_backend: str | None) -> str | None:
    if video_backend is not None:
        return video_backend

    try:
        importlib.import_module("torchcodec")
    except Exception as exc:
        logger.warning(
            "torchcodec is unavailable at runtime (%s); falling back to 'pyav' video decoding.",
            exc,
        )
        return "pyav"

    return "torchcodec"


class LeRobotDatasetAdapter(Dataset):
    """Wraps LeRobotDataset with our PolicyIO-driven delta_timestamps.

    This is intentionally thin — we rely on LeRobotDataset for data loading,
    caching, and video decoding. We only add:
    1. Automatic delta_timestamps from PolicyIO
    2. Convenient stats access as torch tensors
    3. DataLoader factory method
    """

    def __init__(
        self,
        repo_id: str,
        io: PolicyIO,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms=None,
        video_backend: str | None = None,
    ) -> None:
        delta_timestamps = io.build_delta_timestamps()
        resolved_video_backend = _resolve_video_backend(video_backend)

        self.lerobot_dataset = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            video_backend=resolved_video_backend,
        )
        self.io = io

    def __len__(self) -> int:
        return len(self.lerobot_dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.lerobot_dataset[idx]

    @property
    def fps(self) -> int:
        return self.lerobot_dataset.fps

    @property
    def num_episodes(self) -> int:
        return self.lerobot_dataset.num_episodes

    @property
    def num_frames(self) -> int:
        return self.lerobot_dataset.num_frames

    @property
    def meta(self):
        return self.lerobot_dataset.meta

    @property
    def stats(self) -> dict[str, dict[str, torch.Tensor]]:
        """Return aggregated normalization stats as torch tensors."""
        raw_stats = getattr(self.lerobot_dataset.meta, "stats", None)
        if not isinstance(raw_stats, dict):
            return {}
        return stats_to_tensors(raw_stats)

    @property
    def normalization_modes(self) -> dict[str, NormMode]:
        """Return resolved per-feature normalization modes for active IO features."""
        meta_features = getattr(self.lerobot_dataset.meta, "features", None)
        return resolve_normalization_modes_for_io(meta_features, self.io)

    def make_dataloader(
        self,
        batch_size: int = 64,
        num_workers: int = 4,
        shuffle: bool = True,
        pin_memory: bool = True,
        prefetch_factor: int | None = None,
        use_episode_aware_sampler: bool = True,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        drop_last: bool = True,
    ) -> DataLoader:
        """Create a DataLoader for this dataset."""
        sampler = None
        if use_episode_aware_sampler:
            ep_indices = self.lerobot_dataset.hf_dataset["episode_index"]
            dataset_from_indices, dataset_to_indices = _episode_bounds_from_indices(
                ep_indices
            )
            sampler = EpisodeAwareSampler(
                dataset_from_indices=dataset_from_indices,
                dataset_to_indices=dataset_to_indices,
                drop_n_first_frames=drop_n_first_frames,
                drop_n_last_frames=drop_n_last_frames,
                shuffle=shuffle,
            )

        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=(shuffle and sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            drop_last=drop_last,
            prefetch_factor=prefetch_factor,
        )


@dataclass
class _ActionColumnCache:
    """In-memory action data for arrow-free ``__getitem__``.

    Row-indexed tensors follow the local (possibly episode-filtered) hf_dataset
    row order; episode bound tensors are indexed by global episode id, matching
    ``LeRobotDataset._get_query_indices`` semantics.
    """

    actions: torch.Tensor  # (num_rows, action_dim) float32
    episode_index: torch.Tensor  # (num_rows,) int64, global episode ids
    abs_index: torch.Tensor  # (num_rows,) int64, global frame indices
    rel_of_abs: torch.Tensor | None  # abs frame idx -> local row (None = identity)
    ep_from: torch.Tensor  # (num_episodes,) int64 global episode start frame
    ep_to: torch.Tensor  # (num_episodes,) int64 global episode end frame (excl.)
    action_deltas: torch.Tensor  # (action_horizon,) int64 delta indices


class LeRobotActionOnlyDatasetAdapter(LeRobotDatasetAdapter):
    """LeRobot adapter specialized for action-only training.

    This path bypasses video decoding in ``LeRobotDataset.__getitem__`` and only
    returns action tensors needed by tokenizer training. On construction the
    action column is materialized into plain in-memory tensors (a few tens of
    MB even for millions of frames) so per-sample access is pure tensor
    slicing instead of per-row Arrow queries, which otherwise bottleneck
    tokenizer training at a few hundred samples per second per worker.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._prune_hf_columns()
        self._action_cache: _ActionColumnCache | None = None
        try:
            self._action_cache = self._materialize_action_cache()
        except Exception as exc:
            logger.warning(
                "Failed to materialize in-memory action cache (%s: %s); "
                "falling back to per-row Arrow access.",
                type(exc).__name__,
                exc,
            )

    def _prune_hf_columns(self) -> None:
        """Trim HF columns so row fetches don't materialize unused features."""
        ds = self.lerobot_dataset
        ds._ensure_hf_dataset_loaded()
        if self.io.action_feature is None:
            raise ValueError(
                "LeRobotActionOnlyDatasetAdapter requires io.action_feature to be set."
            )
        action_key = self.io.action_feature.key
        keep_cols = [action_key, "episode_index", "index"]
        try:
            ds.hf_dataset = ds.hf_dataset.select_columns(keep_cols)
        except Exception:
            # Best-effort optimization: fall back to unpruned columns if backend
            # does not support select_columns.
            return

    def _materialize_action_cache(self) -> _ActionColumnCache:
        """Load action/episode/index columns into contiguous tensors."""
        ds = self.lerobot_dataset
        ds._ensure_hf_dataset_loaded()
        assert self.io.action_feature is not None  # validated in _prune_hf_columns
        action_key = self.io.action_feature.key
        if ds.delta_indices is None or action_key not in ds.delta_indices:
            raise ValueError("delta_indices for the action key are required.")
        episodes_meta = ds.meta.episodes
        if episodes_meta is None:
            raise ValueError("dataset meta.episodes is unavailable.")

        table = cast(pa.Table, ds.hf_dataset.with_format("arrow")[:])
        num_rows = table.num_rows

        actions_np = _list_column_to_2d_float32(table.column(action_key), num_rows)
        episode_index_np = _int64_column(table.column("episode_index"))
        abs_index_np = _int64_column(table.column("index"))

        rel_of_abs: torch.Tensor | None = None
        if ds._absolute_to_relative_idx is not None:
            rel_of_abs_np = np.full(int(abs_index_np.max()) + 1, -1, dtype=np.int64)
            rel_of_abs_np[abs_index_np] = np.arange(num_rows, dtype=np.int64)
            rel_of_abs = torch.from_numpy(rel_of_abs_np)

        return _ActionColumnCache(
            actions=torch.from_numpy(actions_np),
            episode_index=torch.from_numpy(episode_index_np),
            abs_index=torch.from_numpy(abs_index_np),
            rel_of_abs=rel_of_abs,
            ep_from=torch.tensor(
                episodes_meta["dataset_from_index"], dtype=torch.int64
            ),
            ep_to=torch.tensor(episodes_meta["dataset_to_index"], dtype=torch.int64),
            action_deltas=torch.tensor(ds.delta_indices[action_key], dtype=torch.int64),
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        cache = self._action_cache
        if cache is None:
            return self._getitem_arrow(idx)

        assert self.io.action_feature is not None
        action_key = self.io.action_feature.key
        ep_idx = int(cache.episode_index[idx])
        abs_idx = int(cache.abs_index[idx])

        # Mirrors LeRobotDataset._get_query_indices: clamp to episode bounds and
        # mark out-of-episode steps as padding.
        ep_start = int(cache.ep_from[ep_idx])
        ep_end = int(cache.ep_to[ep_idx])
        shifted = abs_idx + cache.action_deltas
        query = shifted.clamp(min=ep_start, max=ep_end - 1)
        is_pad = (shifted < ep_start) | (shifted >= ep_end)

        rows = query if cache.rel_of_abs is None else cache.rel_of_abs[query]
        return {
            action_key: cache.actions[rows],
            f"{action_key}_is_pad": is_pad,
        }

    def _getitem_arrow(self, idx: int) -> dict[str, torch.Tensor]:
        """Original per-row Arrow access path, kept as the non-cached fallback."""
        ds = self.lerobot_dataset
        ds._ensure_hf_dataset_loaded()

        item = ds.hf_dataset[idx]
        ep_idx = int(item["episode_index"].item())
        abs_idx = int(item["index"].item())

        if ds.delta_indices is not None:
            query_indices, padding = ds._get_query_indices(abs_idx, ep_idx)
            query_result = ds._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if self.io.action_feature is None:
            raise ValueError(
                "LeRobotActionOnlyDatasetAdapter requires io.action_feature to be set."
            )
        action_key = self.io.action_feature.key
        if action_key not in item:
            raise KeyError(
                f"Action key {action_key!r} missing from LeRobot item keys={sorted(item.keys())}."
            )
        out: dict[str, torch.Tensor] = {action_key: item[action_key]}
        pad_key = f"{action_key}_is_pad"
        action_is_pad = item.get(pad_key)
        if isinstance(action_is_pad, torch.Tensor):
            out[pad_key] = action_is_pad
        return out


def _combined_chunks(column: pa.ChunkedArray | pa.Array) -> pa.Array:
    """Return a single contiguous pyarrow Array for a table column."""
    if isinstance(column, pa.ChunkedArray):
        if column.num_chunks == 1:
            return column.chunk(0)
        combined = column.combine_chunks()
        return combined.chunk(0) if isinstance(combined, pa.ChunkedArray) else combined
    return column


def _list_column_to_2d_float32(
    column: pa.ChunkedArray | pa.Array, num_rows: int
) -> np.ndarray:
    """Convert a fixed-width list column to a contiguous (num_rows, dim) array."""
    array = _combined_chunks(column)
    flat = array.flatten().to_numpy(zero_copy_only=False)
    return np.array(flat, dtype=np.float32, copy=True).reshape(num_rows, -1)


def _int64_column(column: pa.ChunkedArray | pa.Array) -> np.ndarray:
    array = _combined_chunks(column)
    return np.array(array.to_numpy(zero_copy_only=False), dtype=np.int64, copy=True)


def _episode_bounds_from_indices(
    episode_indices: list[int] | list[torch.Tensor],
) -> tuple[list[int], list[int]]:
    """Build episode [from, to) frame ranges from per-frame episode ids."""
    if len(episode_indices) == 0:
        return [], []

    ep_ids = [
        int(x.item()) if isinstance(x, torch.Tensor) else int(x)
        for x in episode_indices
    ]
    from_indices: list[int] = [0]
    to_indices: list[int] = []

    current_ep = ep_ids[0]
    for i in range(1, len(ep_ids)):
        if ep_ids[i] != current_ep:
            to_indices.append(i)
            from_indices.append(i)
            current_ep = ep_ids[i]

    to_indices.append(len(ep_ids))
    return from_indices, to_indices
