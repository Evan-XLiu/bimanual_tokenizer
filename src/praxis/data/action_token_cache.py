# SPDX-License-Identifier: Apache-2.0

"""Read immutable offline action-token targets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.datasets.sampler import EpisodeAwareSampler
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class ActionTokenCache:
    root: Path
    manifest: dict[str, Any]
    tokens: np.ndarray

    @classmethod
    def open(cls, root: str | Path) -> ActionTokenCache:
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text())
        required = {
            "format_version",
            "tokenizer",
            "codebook_size",
            "num_quantizers",
            "compression_ratio",
            "action_horizon",
            "token_order",
            "num_samples",
            "tokens_file",
        }
        missing = sorted(required - manifest.keys())
        if missing:
            raise ValueError(
                f"Action-token cache manifest is missing: {', '.join(missing)}"
            )
        if manifest["format_version"] != 1:
            raise ValueError(
                f"Unsupported action-token cache format_version={manifest['format_version']!r}"
            )
        if manifest["tokenizer"] != "x-tokenizer":
            raise ValueError(f"Unsupported tokenizer={manifest['tokenizer']!r}")
        if manifest["token_order"] != "time_major":
            raise ValueError("X-Tokenizer cache token_order must be 'time_major'.")
        tokens = np.load(root / manifest["tokens_file"], mmap_mode="r")
        latent_horizon, remainder = divmod(
            int(manifest["action_horizon"]), int(manifest["compression_ratio"])
        )
        expected = (
            int(manifest["num_samples"]),
            latent_horizon,
            int(manifest["num_quantizers"]),
        )
        if remainder or tokens.shape != expected:
            raise ValueError(
                f"Action-token cache token shape must be {expected}, got {tokens.shape}."
            )
        if not np.issubdtype(tokens.dtype, np.integer):
            raise ValueError(
                f"Action-token cache dtype must be integer, got {tokens.dtype}."
            )
        if tokens.size and (
            int(tokens.min()) < 0 or int(tokens.max()) >= int(manifest["codebook_size"])
        ):
            raise ValueError(
                "Action-token cache contains ids outside the configured codebook."
            )
        return cls(root=root, manifest=manifest, tokens=tokens)

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    def token_ids(self, index: int) -> np.ndarray:
        return np.asarray(self.tokens[index]).reshape(-1)


class CachedActionTokenDataset(Dataset):
    """Attach cached token targets while preserving the continuous action sample."""

    def __init__(self, dataset: Dataset, cache: ActionTokenCache) -> None:
        if len(dataset) != len(cache):
            raise ValueError(
                f"Dataset/cache length mismatch: {len(dataset)} != {len(cache)}"
            )
        self.dataset = dataset
        self.cache = cache

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.dataset[index])
        token_ids = torch.from_numpy(self.cache.token_ids(index).copy()).long()
        sample["action_token_ids"] = token_ids
        sample["action_token_length"] = torch.tensor(
            token_ids.numel(), dtype=torch.long
        )
        return sample

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
        """Create a loader whose dataset remains this cache wrapper."""
        sampler = None
        if use_episode_aware_sampler:
            source = getattr(self.dataset, "lerobot_dataset", None)
            if source is None:
                raise ValueError(
                    "Episode-aware cached loading requires a LeRobot-backed dataset."
                )
            episode_indices = source.hf_dataset["episode_index"]
            from_indices, to_indices = _episode_bounds(episode_indices)
            sampler = EpisodeAwareSampler(
                dataset_from_indices=from_indices,
                dataset_to_indices=to_indices,
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

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)


def _episode_bounds(
    episode_indices: list[int] | list[torch.Tensor],
) -> tuple[list[int], list[int]]:
    if not episode_indices:
        return [], []
    episode_ids = [
        int(value.item()) if isinstance(value, torch.Tensor) else int(value)
        for value in episode_indices
    ]
    from_indices = [0]
    to_indices: list[int] = []
    current = episode_ids[0]
    for index, episode_id in enumerate(episode_ids[1:], start=1):
        if episode_id != current:
            to_indices.append(index)
            from_indices.append(index)
            current = episode_id
    to_indices.append(len(episode_ids))
    return from_indices, to_indices
