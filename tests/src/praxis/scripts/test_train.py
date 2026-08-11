# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from praxis.scripts.train import configure_local_dataset_environment


def test_local_dataset_configures_shared_hf_cache_before_trainer_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_DATASETS_CACHE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_DATASETS_OFFLINE", raising=False)
    dataset_root = tmp_path / "robotwin"
    cfg = OmegaConf.create(
        {
            "dataset": {
                "repo_id": "local/robotwin",
                "root": str(dataset_root),
                "hf_datasets_cache": None,
            }
        }
    )

    cache = configure_local_dataset_environment(cfg)

    assert cache is not None
    assert cache == tmp_path / ".huggingface" / "datasets"
    assert cache.is_dir()
    assert os.environ["HF_DATASETS_CACHE"] == str(cache)
    assert "HF_HUB_OFFLINE" not in os.environ
    assert os.environ["HF_DATASETS_OFFLINE"] == "1"


def test_remote_dataset_does_not_change_hf_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "existing"))
    cfg = OmegaConf.create(
        {"dataset": {"repo_id": "lerobot/libero", "root": None}}
    )

    assert configure_local_dataset_environment(cfg) is None
    assert os.environ["HF_DATASETS_CACHE"] == str(tmp_path / "existing")
