"""Read several local LeRobot tasks as one policy training dataset.

Video stays in the source folders; the dataset only combines sample indices.
"""

from __future__ import annotations

import bisect
import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from lerobot.datasets.sampler import EpisodeAwareSampler
from torch.utils.data import DataLoader, Dataset

from praxis.core.types import PolicyIO
from praxis.data.lerobot_adapter import (
    LeRobotDatasetAdapter,
    _episode_bounds_from_indices,
)


def source_episode_counts(sources: Sequence[Any]) -> list[int]:
    return [
        int(json.loads((Path(str(source.root)) / "meta/info.json").read_text())["total_episodes"])
        for source in sources
    ]


class MultiSourceLeRobotDatasetAdapter(Dataset):
    """Concat local video datasets while preserving per-episode sampling bounds."""

    def __init__(
        self,
        *,
        sources: Sequence[Any],
        io: PolicyIO,
        episodes: Sequence[int],
        video_backend: str | None,
    ) -> None:
        self.io = io
        self.sources = sources
        self.datasets: list[LeRobotDatasetAdapter] = []
        self.cumulative_lengths: list[int] = []
        self.episode_starts: list[int] = []
        self.episode_ends: list[int] = []
        global_offset = 0
        for source, count in zip(sources, source_episode_counts(sources), strict=True):
            local_episodes = [
                int(index - global_offset)
                for index in episodes
                if global_offset <= index < global_offset + count
            ]
            global_offset += count
            if not local_episodes:
                continue
            dataset = LeRobotDatasetAdapter(
                repo_id=str(source.repo_id),
                io=io,
                root=str(source.root),
                episodes=local_episodes,
                video_backend=video_backend,
            )
            if self.datasets and dataset.fps != self.datasets[0].fps:
                raise ValueError("All multi-source tasks must have the same FPS")
            dataset.lerobot_dataset._ensure_hf_dataset_loaded()
            from_indices, to_indices = _episode_bounds_from_indices(
                dataset.lerobot_dataset.hf_dataset["episode_index"]
            )
            previous = self.cumulative_lengths[-1] if self.cumulative_lengths else 0
            self.episode_starts.extend(previous + index for index in from_indices)
            self.episode_ends.extend(previous + index for index in to_indices)
            self.datasets.append(dataset)
            self.cumulative_lengths.append(previous + len(dataset))
        if not self.datasets:
            raise ValueError("Multi-source dataset contains no selected episodes")
        self.fps = self.datasets[0].fps
        self.meta = SimpleNamespace(tasks=[str(source.name) for source in sources])
        self.normalization_modes = self.datasets[0].normalization_modes
        self.stats = self._aggregate_state_stats()

    def _aggregate_state_stats(self) -> dict[str, dict[str, torch.Tensor]]:
        result: dict[str, dict[str, torch.Tensor]] = {}
        for feature in self.io.state_features:
            per_source = []
            for source in self.sources:
                stats_file = Path(str(source.root)) / "meta/stats.json"
                stats = json.loads(stats_file.read_text()).get(feature.key)
                if stats is not None:
                    per_source.append(stats)
            if not per_source:
                continue
            mins = np.stack([np.asarray(s["min"], dtype=np.float32) for s in per_source])
            maxs = np.stack([np.asarray(s["max"], dtype=np.float32) for s in per_source])
            result[feature.key] = {
                "min": torch.from_numpy(mins.min(axis=0)),
                "max": torch.from_numpy(maxs.max(axis=0)),
            }
        return result

    def __len__(self) -> int:
        return self.cumulative_lengths[-1]

    def __getitem__(self, index: int) -> dict:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        source_index = bisect.bisect_right(self.cumulative_lengths, index)
        previous = 0 if source_index == 0 else self.cumulative_lengths[source_index - 1]
        return self.datasets[source_index][index - previous]

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
        sampler = None
        if use_episode_aware_sampler:
            sampler = EpisodeAwareSampler(
                self.episode_starts,
                self.episode_ends,
                drop_n_first_frames=drop_n_first_frames,
                drop_n_last_frames=drop_n_last_frames,
                shuffle=shuffle,
            )
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            drop_last=drop_last,
            prefetch_factor=prefetch_factor,
        )
