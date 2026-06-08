# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from omegaconf import OmegaConf

from scripts.infra.launch.config import DATASET_CONFIG_ROOT, DATASET_REPO_IDS


def test_launcher_dataset_repo_ids_match_hydra_dataset_configs() -> None:
    for name, launcher_repo_id in DATASET_REPO_IDS.items():
        config_path = DATASET_CONFIG_ROOT / f"{name}.yaml"
        if not config_path.exists():
            continue

        cfg = OmegaConf.load(config_path)
        assert launcher_repo_id == cfg.repo_id
