# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Thin adapter around LeRobotDataset for our training pipeline."""

from __future__ import annotations

import importlib
import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep
from typing import Any, cast
from uuid import uuid4

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
from praxis.data.bridge_xtokenizer import (
    BridgeXTokenizerEpisode,
    build_bridge_xtokenizer_episode,
)
from praxis.data.utils import resolve_normalization_modes_for_io, stats_to_tensors
from praxis.data.xtokenizer_canonical_stats import CanonicalActionStatistics
from praxis.data.xtokenizer_vl import VLFeatureSidecar

logger = logging.getLogger(__name__)

_METADATA_READY_TIMEOUT_SECONDS = 120.0
_METADATA_READY_POLL_SECONDS = 0.25
_CACHE_PROVENANCE_FILENAME = ".praxis-dataset-provenance.json"
_CACHE_PROVENANCE_FORMAT_VERSION = 1
_CACHE_PROVENANCE_REMEDIATION = (
    "Rerun real prepare_vl_features.py for the configured dataset revision."
)


def ensure_lerobot_dataset_available(
    *,
    repo_id: str,
    runtime: Any | None = None,
    root: str | Path | None = None,
    episodes: list[int] | None = None,
    download_videos: bool,
    bypass_force_cache_sync: bool = False,
    revision: str | None = None,
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
            sync_kwargs: dict[str, Any] = {"repo_id": repo_id, "root": root_path}
            if revision is not None:
                sync_kwargs["revision"] = revision
            _sync_lerobot_metadata(**sync_kwargs)
            _wait_for_lerobot_metadata_ready(root_path)
        dataset_kwargs: dict[str, Any] = {
            "repo_id": repo_id,
            "root": root,
            "episodes": episodes,
            "force_cache_sync": False,
            "download_videos": download_videos,
        }
        if revision is not None:
            dataset_kwargs["revision"] = revision
        dataset = LeRobotDataset(**dataset_kwargs)
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


def read_lerobot_cache_provenance(
    *,
    root: str | Path | None,
    repo_id: str,
) -> str:
    """Read the independently recorded immutable revision for a local cache."""
    path = _lerobot_cache_root(root, repo_id) / _CACHE_PROVENANCE_FILENAME
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError:
        raise ValueError(
            f"LeRobot cache provenance marker is missing at {path}. "
            f"{_CACHE_PROVENANCE_REMEDIATION}"
        ) from None
    except (json.JSONDecodeError, OSError):
        raise ValueError(
            f"LeRobot cache provenance marker is malformed at {path}. "
            f"{_CACHE_PROVENANCE_REMEDIATION}"
        ) from None
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("format_version"), int)
        or isinstance(payload.get("format_version"), bool)
        or payload["format_version"] != _CACHE_PROVENANCE_FORMAT_VERSION
        or not isinstance(payload.get("dataset_repo_id"), str)
        or not payload["dataset_repo_id"].strip()
        or not isinstance(payload.get("dataset_revision"), str)
    ):
        raise ValueError(
            f"LeRobot cache provenance marker is malformed at {path}. "
            f"{_CACHE_PROVENANCE_REMEDIATION}"
        )
    marker_repo_id = str(payload["dataset_repo_id"])
    if marker_repo_id != repo_id:
        raise ValueError(
            "LeRobot cache provenance repository "
            f"{marker_repo_id!r} does not match configured dataset repository "
            f"{repo_id!r}. {_CACHE_PROVENANCE_REMEDIATION}"
        )
    try:
        return _require_immutable_hub_revision(
            str(payload["dataset_revision"]),
            label="LeRobot cache provenance revision",
        )
    except ValueError as exc:
        raise ValueError(f"{exc} {_CACHE_PROVENANCE_REMEDIATION}") from None


def write_lerobot_cache_provenance(
    *,
    root: str | Path | None,
    repo_id: str,
    revision: str,
) -> None:
    """Atomically record which immutable dataset revision populated a cache."""
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError("dataset repository must be a non-empty string.")
    immutable_revision = _require_immutable_hub_revision(
        revision,
        label="dataset revision",
    )
    root_path = _lerobot_cache_root(root, repo_id)
    root_path.mkdir(parents=True, exist_ok=True)
    path = root_path / _CACHE_PROVENANCE_FILENAME
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    payload = {
        "format_version": _CACHE_PROVENANCE_FORMAT_VERSION,
        "dataset_repo_id": repo_id,
        "dataset_revision": immutable_revision,
    }
    try:
        with temporary_path.open("x") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _lerobot_cache_root(root: str | Path | None, repo_id: str) -> Path:
    return Path(root) if root is not None else HF_LEROBOT_HOME / repo_id


def _require_immutable_hub_revision(value: str, *, label: str) -> str:
    if len(value) != 40:
        raise ValueError(
            f"{label} must be an immutable 40-character hexadecimal Hub commit SHA."
        )
    try:
        int(value, 16)
    except ValueError:
        raise ValueError(
            f"{label} must be an immutable 40-character hexadecimal Hub commit SHA."
        ) from None
    return value.lower()


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


def _sync_lerobot_metadata(
    *,
    repo_id: str,
    root: Path,
    revision: str | None = None,
) -> None:
    resolved_revision = revision or CODEBASE_VERSION
    if is_valid_version(resolved_revision):
        resolved_revision = get_safe_version(repo_id, resolved_revision)
    root.mkdir(exist_ok=True, parents=True)
    snapshot_download(
        repo_id,
        repo_type="dataset",
        revision=resolved_revision,
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
        revision: str | None = None,
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
            revision=revision,
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
    states: torch.Tensor | None = None  # (num_rows, state_dim) float32
    frame_index: torch.Tensor | None = None  # (num_rows,) local episode frame
    robot_type: torch.Tensor | None = None  # (num_rows,) int64 embodiment id
    bridge_absolute_actions: torch.Tensor | None = None  # (num_rows, canonical_dim)
    bridge_observation_states: torch.Tensor | None = None  # (num_rows, canonical_dim)
    bridge_dof_mask: torch.Tensor | None = None  # (num_rows, canonical_dim)


class LeRobotActionOnlyDatasetAdapter(LeRobotDatasetAdapter):
    """LeRobot adapter specialized for action-only training.

    This path bypasses video decoding in ``LeRobotDataset.__getitem__`` and only
    returns action tensors needed by tokenizer training. On construction the
    action column is materialized into plain in-memory tensors (a few tens of
    MB even for millions of frames) so per-sample access is pure tensor
    slicing instead of per-row Arrow queries, which otherwise bottleneck
    tokenizer training at a few hundred samples per second per worker.
    """

    def __init__(
        self,
        *args,
        vl_sidecar: VLFeatureSidecar | None = None,
        robot_type: int = 0,
        include_xtokenizer_context: bool = False,
        require_valid_next_vl: bool = True,
        xtokenizer_bridge_dof_dims: Mapping[str, int] | None = None,
        xtokenizer_bridge_dof_order: tuple[str, ...] | list[str] | None = None,
        canonical_action_statistics: CanonicalActionStatistics | None = None,
        **kwargs,
    ) -> None:
        self.vl_sidecar = vl_sidecar
        self.default_robot_type = int(robot_type)
        self.include_xtokenizer_context = bool(include_xtokenizer_context)
        self.require_valid_next_vl = bool(require_valid_next_vl)
        self.xtokenizer_bridge_dof_dims = (
            None
            if xtokenizer_bridge_dof_dims is None
            else {
                str(name): int(dim) for name, dim in xtokenizer_bridge_dof_dims.items()
            }
        )
        self.xtokenizer_bridge_dof_order = (
            None
            if xtokenizer_bridge_dof_order is None
            else tuple(xtokenizer_bridge_dof_order)
        )
        self.canonical_action_statistics = canonical_action_statistics
        super().__init__(*args, **kwargs)
        self._prune_hf_columns()
        self._bridge_episode_cache: dict[int, BridgeXTokenizerEpisode] = {}
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
        self._validate_vl_geometry()

    @property
    def stats(self) -> dict[str, dict[str, torch.Tensor]]:
        artifact = getattr(self, "canonical_action_statistics", None)
        if artifact is None:
            return super().stats
        if self.io.action_feature is None:
            raise ValueError("Canonical action statistics require an action feature.")
        return artifact.normalization_stats(self.io.action_feature.key)

    @property
    def normalization_modes(self) -> dict[str, NormMode]:
        artifact = getattr(self, "canonical_action_statistics", None)
        if artifact is None:
            return super().normalization_modes
        if self.io.action_feature is None:
            raise ValueError("Canonical action statistics require an action feature.")
        return artifact.normalization_modes(self.io.action_feature.key)

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
        available = set(getattr(ds.hf_dataset, "column_names", ()))
        if (
            self.vl_sidecar is not None
            or self.include_xtokenizer_context
            or self.xtokenizer_bridge_dof_dims is not None
        ):
            keep_cols.extend(
                key
                for key in ("frame_index", "observation.state", "robot_type")
                if key in available
            )
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
        episode_bounds = _episode_bounds_from_metadata(episodes_meta)

        table = cast(pa.Table, ds.hf_dataset.with_format("arrow")[:])
        num_rows = table.num_rows

        actions_np = _list_column_to_2d_float32(table.column(action_key), num_rows)
        episode_index_np = _int64_column(table.column("episode_index"))
        abs_index_np = _int64_column(table.column("index"))
        ep_from, ep_to = _episode_bounds_to_tensors(episode_bounds)

        states: torch.Tensor | None = None
        if "observation.state" in table.column_names:
            states = torch.from_numpy(
                _list_column_to_2d_float32(
                    table.column("observation.state"),
                    num_rows,
                )
            )
        if "frame_index" in table.column_names:
            frame_index = torch.from_numpy(_int64_column(table.column("frame_index")))
        else:
            episode_rows = torch.from_numpy(episode_index_np)
            frame_index = torch.from_numpy(abs_index_np) - ep_from[episode_rows]
        robot_type: torch.Tensor | None = None
        if "robot_type" in table.column_names:
            robot_type = torch.from_numpy(_int64_column(table.column("robot_type")))

        rel_of_abs: torch.Tensor | None = None
        if ds._absolute_to_relative_idx is not None:
            rel_of_abs_np = np.full(int(abs_index_np.max()) + 1, -1, dtype=np.int64)
            rel_of_abs_np[abs_index_np] = np.arange(num_rows, dtype=np.int64)
            rel_of_abs = torch.from_numpy(rel_of_abs_np)

        bridge_absolute_actions: torch.Tensor | None = None
        bridge_observation_states: torch.Tensor | None = None
        bridge_dof_mask: torch.Tensor | None = None
        dof_dims = getattr(self, "xtokenizer_bridge_dof_dims", None)
        if dof_dims is not None:
            if states is None:
                raise ValueError(
                    "Bridge X-Tokenizer cache requires observation.state rows."
                )
            boundaries = (
                np.flatnonzero(episode_index_np[1:] != episode_index_np[:-1]) + 1
            )
            run_starts = np.concatenate(([0], boundaries))
            run_ends = np.concatenate((boundaries, [num_rows]))
            absolute_parts: list[np.ndarray] = []
            observation_parts: list[np.ndarray] = []
            mask_parts: list[np.ndarray] = []
            seen_episodes: set[int] = set()
            for run_start, run_end in zip(run_starts, run_ends, strict=True):
                episode_id = int(episode_index_np[run_start])
                if episode_id in seen_episodes:
                    raise ValueError(
                        f"Bridge episode {episode_id} rows must be contiguous."
                    )
                seen_episodes.add(episode_id)
                expected_indices = np.arange(
                    int(ep_from[episode_id]),
                    int(ep_to[episode_id]),
                    dtype=np.int64,
                )
                if not np.array_equal(
                    abs_index_np[run_start:run_end],
                    expected_indices,
                ):
                    raise ValueError(
                        f"Bridge episode {episode_id} cache must contain every "
                        "frame in metadata order."
                    )
                episode = build_bridge_xtokenizer_episode(
                    states[run_start:run_end].numpy(),
                    actions_np[run_start:run_end],
                    dof_dims,
                    dof_order=getattr(self, "xtokenizer_bridge_dof_order", None),
                )
                absolute_parts.append(episode.absolute_actions)
                observation_parts.append(episode.observation_states_absolute)
                mask_parts.append(episode.dof_mask)
            bridge_absolute_actions = torch.from_numpy(
                np.concatenate(absolute_parts, axis=0)
            )
            bridge_observation_states = torch.from_numpy(
                np.concatenate(observation_parts, axis=0)
            )
            bridge_dof_mask = torch.from_numpy(np.concatenate(mask_parts, axis=0))

        return _ActionColumnCache(
            actions=torch.from_numpy(actions_np),
            episode_index=torch.from_numpy(episode_index_np),
            abs_index=torch.from_numpy(abs_index_np),
            rel_of_abs=rel_of_abs,
            ep_from=ep_from,
            ep_to=ep_to,
            action_deltas=torch.tensor(ds.delta_indices[action_key], dtype=torch.int64),
            states=states,
            frame_index=frame_index,
            robot_type=robot_type,
            bridge_absolute_actions=bridge_absolute_actions,
            bridge_observation_states=bridge_observation_states,
            bridge_dof_mask=bridge_dof_mask,
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
        out = {
            action_key: cache.actions[rows],
            f"{action_key}_is_pad": is_pad,
        }
        bridge_bound = False
        if getattr(self, "xtokenizer_bridge_dof_dims", None) is not None:
            if (
                cache.bridge_absolute_actions is None
                or cache.bridge_observation_states is None
                or cache.bridge_dof_mask is None
            ):
                raise ValueError(
                    "Bridge X-Tokenizer fast cache is missing its episode trajectory."
                )
            out[action_key] = cache.bridge_absolute_actions[rows]
            out["observation.state"] = cache.bridge_observation_states[idx]
            out["action_dof_mask"] = cache.bridge_dof_mask[rows]
            bridge_bound = True
        if cache.states is not None and not bridge_bound:
            out["observation.state"] = cache.states[idx]
        robot_type = (
            cache.robot_type[idx]
            if cache.robot_type is not None
            else torch.tensor(
                getattr(self, "default_robot_type", 0),
                dtype=torch.int64,
            )
        )
        include_context = bool(
            getattr(self, "include_xtokenizer_context", False)
            or getattr(self, "vl_sidecar", None) is not None
        )
        if include_context:
            out["robot_type"] = robot_type
        if getattr(self, "vl_sidecar", None) is not None:
            local_frame = (
                int(cache.frame_index[idx])
                if cache.frame_index is not None
                else abs_idx - ep_start
            )
            out.update(
                self._lookup_vl_fields(
                    episode_index=ep_idx,
                    frame_index=local_frame,
                    action_deltas=cache.action_deltas,
                    action_is_pad=is_pad,
                    episode_length=ep_end - ep_start,
                )
            )
        return out

    def _getitem_arrow(self, idx: int) -> dict[str, torch.Tensor]:
        """Original per-row Arrow access path, kept as the non-cached fallback."""
        ds = self.lerobot_dataset
        ds._ensure_hf_dataset_loaded()

        item = ds.hf_dataset[idx]
        ep_idx = int(item["episode_index"].item())
        abs_idx = int(item["index"].item())
        action_query_indices: list[int] | None = None

        if ds.delta_indices is not None:
            query_indices, padding = ds._get_query_indices(abs_idx, ep_idx)
            query_result = ds._query_hf_dataset(query_indices)
            if self.io.action_feature is not None:
                action_query_indices = query_indices.get(self.io.action_feature.key)
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
        state = item.get("observation.state")
        if isinstance(state, torch.Tensor):
            out["observation.state"] = state
        include_context = bool(
            getattr(self, "include_xtokenizer_context", False)
            or getattr(self, "vl_sidecar", None) is not None
        )
        if include_context:
            robot_type = item.get("robot_type")
            out["robot_type"] = (
                robot_type.long()
                if isinstance(robot_type, torch.Tensor)
                else torch.tensor(
                    getattr(self, "default_robot_type", 0),
                    dtype=torch.int64,
                )
            )
        if getattr(self, "vl_sidecar", None) is not None:
            if action_is_pad is None:
                action_is_pad = torch.zeros(
                    item[action_key].shape[0],
                    dtype=torch.bool,
                )
            delta_indices = ds.delta_indices
            if delta_indices is None:
                raise ValueError("VL sidecar attachment requires action delta indices.")
            frame_value = item.get("frame_index")
            episode_bounds = _episode_bounds_from_metadata(ds.meta.episodes)
            try:
                episode_start, episode_end = episode_bounds[ep_idx]
            except KeyError:
                raise ValueError(
                    f"dataset meta.episodes does not contain episode {ep_idx}."
                ) from None
            frame_index = (
                int(frame_value.item())
                if isinstance(frame_value, torch.Tensor)
                else abs_idx - episode_start
            )
            action_deltas = torch.tensor(
                delta_indices[action_key],
                dtype=torch.int64,
            )
            episode_length = episode_end - episode_start
            out.update(
                self._lookup_vl_fields(
                    episode_index=ep_idx,
                    frame_index=frame_index,
                    action_deltas=action_deltas,
                    action_is_pad=action_is_pad,
                    episode_length=episode_length,
                )
            )
        if getattr(self, "xtokenizer_bridge_dof_dims", None) is not None:
            if action_query_indices is None:
                raise ValueError(
                    "Bridge X-Tokenizer Arrow fallback requires action query indices."
                )
            episode_bounds = _episode_bounds_from_metadata(ds.meta.episodes)
            try:
                episode_start, _ = episode_bounds[ep_idx]
            except KeyError:
                raise ValueError(
                    f"dataset meta.episodes does not contain episode {ep_idx}."
                ) from None
            episode = self._get_or_build_bridge_episode(ep_idx)
            action_rows = (
                np.asarray(action_query_indices, dtype=np.int64) - episode_start
            )
            observation_row = abs_idx - episode_start
            out[action_key] = torch.from_numpy(episode.absolute_actions[action_rows])
            out["observation.state"] = torch.from_numpy(
                episode.observation_states_absolute[observation_row]
            )
            out["action_dof_mask"] = torch.from_numpy(episode.dof_mask[action_rows])
            return out
        return out

    def _get_or_build_bridge_episode(
        self,
        episode_index: int,
    ) -> BridgeXTokenizerEpisode:
        cache = getattr(self, "_bridge_episode_cache", None)
        if cache is None:
            cache = {}
            self._bridge_episode_cache = cache
        cached = cache.get(episode_index)
        if cached is not None:
            return cached
        if self.io.action_feature is None:
            raise ValueError("Bridge X-Tokenizer binding requires an action feature.")
        dof_dims = getattr(self, "xtokenizer_bridge_dof_dims", None)
        if dof_dims is None:
            raise ValueError("Bridge X-Tokenizer DoF layout is not configured.")

        ds = self.lerobot_dataset
        episode_bounds = _episode_bounds_from_metadata(ds.meta.episodes)
        try:
            episode_start, episode_end = episode_bounds[episode_index]
        except KeyError:
            raise ValueError(
                f"dataset meta.episodes does not contain episode {episode_index}."
            ) from None
        episode_rows = list(range(episode_start, episode_end))
        action_key = self.io.action_feature.key
        raw = ds._query_hf_dataset(
            {
                action_key: episode_rows,
                "observation.state": episode_rows,
            }
        )
        actions = raw.get(action_key)
        states = raw.get("observation.state")
        if not isinstance(actions, torch.Tensor) or not isinstance(
            states, torch.Tensor
        ):
            raise ValueError(
                "Bridge X-Tokenizer episode query requires raw action and "
                "observation.state tensors."
            )
        episode = build_bridge_xtokenizer_episode(
            states.detach().cpu().float().numpy(),
            actions.detach().cpu().float().numpy(),
            dof_dims,
            dof_order=getattr(self, "xtokenizer_bridge_dof_order", None),
        )
        cache[episode_index] = episode
        return episode

    def _validate_vl_geometry(self) -> None:
        sidecar = getattr(self, "vl_sidecar", None)
        if sidecar is None:
            return
        action_deltas: torch.Tensor | None = None
        if self._action_cache is not None:
            action_deltas = self._action_cache.action_deltas
        elif (
            self.io.action_feature is not None
            and self.lerobot_dataset.delta_indices is not None
        ):
            raw_deltas = self.lerobot_dataset.delta_indices.get(
                self.io.action_feature.key
            )
            if raw_deltas is not None:
                action_deltas = torch.tensor(raw_deltas, dtype=torch.int64)
        if action_deltas is None:
            raise ValueError("VL sidecar attachment requires action delta indices.")
        if action_deltas.ndim != 1 or action_deltas.numel() == 0:
            raise ValueError(
                "VL sidecar attachment requires a non-empty 1-D action horizon."
            )
        if bool((action_deltas[1:] <= action_deltas[:-1]).any()):
            raise ValueError("VL sidecar attachment requires increasing action deltas.")

    def _lookup_vl_fields(
        self,
        *,
        episode_index: int,
        frame_index: int,
        action_deltas: torch.Tensor,
        action_is_pad: torch.Tensor,
        episode_length: int,
    ) -> dict[str, torch.Tensor]:
        sidecar = self.vl_sidecar
        if sidecar is None:
            return {}
        stride = sidecar.stride
        if action_deltas.ndim != 1 or action_deltas.numel() == 0:
            raise ValueError("VL lookup requires a non-empty 1-D action horizon.")
        if action_is_pad.shape != action_deltas.shape:
            raise ValueError("Action padding mask must match the action horizon.")
        if bool((action_deltas[1:] <= action_deltas[:-1]).any()):
            raise ValueError("VL lookup requires increasing action deltas.")

        chunk_start = frame_index + int(action_deltas[0])
        chunk_stop = frame_index + int(action_deltas[-1]) + 1
        bounded_start = max(chunk_start, 0)
        bounded_stop = min(chunk_stop, episode_length)
        first_vl_frame = (bounded_start + stride - 1) // stride
        stop_vl_frame = (bounded_stop + stride - 1) // stride
        if first_vl_frame >= stop_vl_frame:
            raise ValueError("Action chunk contains no sampled VL frame.")
        vl_frames = torch.arange(
            first_vl_frame,
            stop_vl_frame,
            dtype=torch.int64,
        )
        lookup = sidecar.lookup(
            torch.full_like(vl_frames, episode_index),
            vl_frames,
        )
        vl_feature_is_pad = lookup["vl_feature_is_pad"]
        vl_features = lookup["vl_features"].masked_fill(
            vl_feature_is_pad.unsqueeze(-1),
            0,
        )
        next_vl_features, next_vl_feature_is_pad = _select_next_vl_feature(
            lookup["next_vl_features"][-1],
            lookup["next_vl_feature_is_pad"][-1],
            vl_feature_is_pad,
        )
        if bool(next_vl_feature_is_pad) and getattr(
            self,
            "require_valid_next_vl",
            True,
        ):
            raise ValueError(
                "Action chunk has no real next-frame VL target; configure the "
                "episode-aware sampler to exclude one additional tail frame."
            )
        if bool(next_vl_feature_is_pad):
            next_vl_features = torch.zeros_like(next_vl_features)
        return {
            "vl_features": vl_features,
            "next_vl_features": next_vl_features,
            "next_vl_feature_is_pad": next_vl_feature_is_pad,
            "vl_feature_is_pad": vl_feature_is_pad,
        }


def _select_next_vl_feature(
    next_features: torch.Tensor,
    next_feature_is_pad: torch.Tensor,
    chunk_view_is_pad: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select face first, then a random present view, for next-frame prediction."""
    if next_features.ndim != 2:
        raise ValueError("next_features must have shape [V, H_vl].")
    if next_feature_is_pad.shape != next_features.shape[:1]:
        raise ValueError("next_feature_is_pad must have shape [V].")
    if chunk_view_is_pad.ndim != 2 or chunk_view_is_pad.shape[1] != len(next_features):
        raise ValueError("chunk_view_is_pad must have shape [T_v, V].")
    if next_feature_is_pad.dtype != torch.bool or chunk_view_is_pad.dtype != torch.bool:
        raise ValueError("VL padding masks must have boolean dtype.")

    present_in_chunk = (~chunk_view_is_pad).any(dim=0)
    candidates = (~next_feature_is_pad) & present_in_chunk
    if bool(candidates[0]):
        selected_view = torch.tensor(0, device=next_features.device)
    else:
        available = candidates.nonzero(as_tuple=False).flatten()
        if available.numel() == 0:
            return (
                torch.zeros_like(next_features[0]),
                torch.tensor(True, device=next_features.device),
            )
        random_index = torch.randint(
            available.numel(),
            (),
            generator=generator,
            device=available.device,
        )
        selected_view = available[random_index]
    return (
        next_features[selected_view],
        torch.tensor(False, device=next_features.device),
    )


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


def _episode_bounds_from_metadata(
    episodes_meta: Any,
) -> dict[int, tuple[int, int]]:
    """Normalize supported LeRobot episode metadata into global frame bounds."""
    column_names = set(getattr(episodes_meta, "column_names", ()))
    is_legacy_mapping = isinstance(episodes_meta, Mapping) and (
        "dataset_from_index" in episodes_meta and "dataset_to_index" in episodes_meta
    )
    if is_legacy_mapping or {
        "dataset_from_index",
        "dataset_to_index",
    }.issubset(column_names):
        starts = list(episodes_meta["dataset_from_index"])
        ends = list(episodes_meta["dataset_to_index"])
        if len(starts) != len(ends):
            raise ValueError(
                "dataset episode start/end metadata must have the same length."
            )
        bounds = {
            episode_index: (int(start), int(end))
            for episode_index, (start, end) in enumerate(zip(starts, ends, strict=True))
        }
    elif isinstance(episodes_meta, Mapping) and episodes_meta:
        bounds = {}
        cursor = 0
        try:
            ordered_items = sorted(
                episodes_meta.items(),
                key=lambda item: int(item[0]),
            )
        except (TypeError, ValueError):
            raise ValueError(
                "dataset meta.episodes keys must be integer episode ids."
            ) from None
        for raw_episode_index, raw_episode in ordered_items:
            episode_index = int(raw_episode_index)
            if not isinstance(raw_episode, Mapping):
                raise ValueError(
                    f"dataset metadata for episode {episode_index} must be a mapping."
                )
            stored_index = int(raw_episode.get("episode_index", episode_index))
            if stored_index != episode_index:
                raise ValueError(
                    "dataset episode metadata key/index mismatch: "
                    f"key={episode_index}, episode_index={stored_index}."
                )
            if (
                "dataset_from_index" in raw_episode
                and "dataset_to_index" in raw_episode
            ):
                start = int(raw_episode["dataset_from_index"])
                end = int(raw_episode["dataset_to_index"])
            else:
                try:
                    length = int(raw_episode["length"])
                except (KeyError, TypeError, ValueError):
                    raise ValueError(
                        f"dataset metadata for episode {episode_index} requires "
                        "a positive integer length."
                    ) from None
                start = cursor
                end = start + length
            bounds[episode_index] = (start, end)
            cursor = end
    else:
        raise ValueError(
            "dataset meta.episodes must be keyed episode metadata or a "
            "columnar table with dataset frame bounds."
        )

    for episode_index, (start, end) in bounds.items():
        if episode_index < 0 or start < 0 or end <= start:
            raise ValueError(
                f"dataset metadata has invalid bounds for episode {episode_index}: "
                f"[{start}, {end})."
            )
    return bounds


def _episode_bounds_to_tensors(
    bounds: dict[int, tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    max_episode_index = max(bounds)
    starts = torch.full((max_episode_index + 1,), -1, dtype=torch.int64)
    ends = torch.full((max_episode_index + 1,), -1, dtype=torch.int64)
    for episode_index, (start, end) in bounds.items():
        starts[episode_index] = start
        ends[episode_index] = end
    return starts, ends


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
