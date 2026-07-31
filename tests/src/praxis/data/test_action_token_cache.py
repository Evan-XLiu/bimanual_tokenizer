# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from praxis.data.action_token_cache import ActionTokenCache, CachedActionTokenDataset


class _Dataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"action": torch.tensor([float(index)])}


def _write_cache(tmp_path) -> None:
    np.save(tmp_path / "tokens.npy", np.arange(64, dtype=np.int64).reshape(2, 8, 4))
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "tokenizer": "x-tokenizer",
                "source_revision": "9733990b16a07d51fb8b5d305ad8c16abb57aef2",
                "checkpoint_sha256": "aa14c832e68df679ba60665f0eb82fd4b4efac48b0c2b6b065d21fa663beec89",
                "codebook_size": 2048,
                "num_quantizers": 4,
                "compression_ratio": 4,
                "action_horizon": 32,
                "token_order": "time_major",
                "num_samples": 2,
                "tokens_file": "tokens.npy",
            }
        )
    )


def test_cache_preserves_structured_rvq_tokens_and_flattens_time_major(
    tmp_path,
) -> None:
    _write_cache(tmp_path)

    cache = ActionTokenCache.open(tmp_path)

    assert cache.tokens.shape == (2, 8, 4)
    assert cache.token_ids(1).tolist() == list(range(32, 64))
    assert cache.block_schedule == (4,) * 8


def test_cache_rejects_manifest_shape_mismatch(tmp_path) -> None:
    _write_cache(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["num_quantizers"] = 3
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="token shape"):
        ActionTokenCache.open(tmp_path)


def test_dataset_attaches_cached_targets_without_mutating_source(tmp_path) -> None:
    _write_cache(tmp_path)
    source = _Dataset()
    dataset = CachedActionTokenDataset(source, ActionTokenCache.open(tmp_path))

    sample = dataset[1]

    assert "action_token_ids" not in source[1]
    assert sample["action_token_ids"].tolist() == list(range(32, 64))
    assert sample["action_token_length"].item() == 32


def test_cached_dataset_dataloader_keeps_cache_wrapper(tmp_path) -> None:
    _write_cache(tmp_path)
    dataset = CachedActionTokenDataset(_Dataset(), ActionTokenCache.open(tmp_path))

    loader = dataset.make_dataloader(
        batch_size=2,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
        use_episode_aware_sampler=False,
        drop_last=False,
    )
    batch = next(iter(loader))

    assert loader.dataset is dataset
    assert batch["action_token_ids"].shape == (2, 32)
    assert batch["action_token_length"].tolist() == [32, 32]
