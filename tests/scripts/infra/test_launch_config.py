# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts.infra.launch.config import (
    DATASET_CONFIG_ROOT,
    DATASET_REPO_IDS,
    resolve_dataset_train_overrides,
)


def test_launcher_dataset_repo_ids_match_hydra_dataset_configs() -> None:
    for name, launcher_repo_id in DATASET_REPO_IDS.items():
        config_path = DATASET_CONFIG_ROOT / f"{name}.yaml"
        if not config_path.exists():
            continue

        cfg = OmegaConf.load(config_path)
        assert launcher_repo_id == cfg.repo_id


@pytest.mark.parametrize("task", ["box", "bottles"])
@pytest.mark.parametrize("config_name", ["train_codecs", "train_policy"])
def test_bicat_launch_config_selects_joint_actions(task, config_name, monkeypatch):
    monkeypatch.setenv("BICAT_DATA_ROOT", "/tmp/bicat")
    name = f"bicat_{task}"
    with initialize_config_dir(
        config_dir=str(DATASET_CONFIG_ROOT.parent), version_base=None
    ):
        cfg = compose(
            config_name=config_name, overrides=resolve_dataset_train_overrides(name)
        )
    assert cfg.dataset.action_key == "action_joint_angles"
    assert cfg.dataset.root == f"/tmp/bicat/{task}"
    assert cfg.dataset.observation_keys == [
        "observation.state_joint_angles",
        "observation.images.top",
        "observation.images.left",
        "observation.images.right",
    ]
    assert cfg.dataset.bypass_force_cache_sync
