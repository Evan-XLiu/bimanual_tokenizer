# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Training entry point."""

from __future__ import annotations

import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

CONFIG_PATH = str(Path(__file__).parents[1] / "configs")


def configure_local_dataset_environment(cfg: DictConfig) -> Path | None:
    """Configure Hugging Face before importing trainers for a local dataset."""
    repo_id = str(cfg.dataset.get("repo_id", ""))
    root = cfg.dataset.get("root")
    if not repo_id.startswith("local/") or root in {None, ""}:
        return None

    configured_cache = cfg.dataset.get("hf_datasets_cache")
    cache = (
        Path(str(configured_cache)).expanduser()
        if configured_cache not in {None, ""}
        else Path(str(root)).expanduser().parent / ".huggingface" / "datasets"
    )
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = str(cache)
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    return cache


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train_policy")
def main(cfg: DictConfig) -> None:
    """Train a model (policy or codec).

    Trainer selection:
        - ``trainer: policy``     -> PolicyTrainer (+ reconst eval + sim eval)
        - ``trainer: codecs``     -> CodecsTrainer
    """
    configure_local_dataset_environment(cfg)
    print(OmegaConf.to_yaml(cfg))

    trainer_name = cfg.trainer._trainer_type

    if trainer_name == "policy":
        from praxis.training.policy_trainer import PolicyTrainer

        trainer = PolicyTrainer(cfg)
    elif trainer_name == "codecs":
        from praxis.training.codecs_trainer import CodecsTrainer

        trainer = CodecsTrainer(cfg)
    else:
        raise ValueError(
            f"Unsupported trainer type: {trainer_name!r}. Use trainer=policy or trainer=codecs."
        )

    trainer.train()


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
