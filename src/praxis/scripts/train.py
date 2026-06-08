# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Training entry point."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

CONFIG_PATH = str(Path(__file__).parents[1] / "configs")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train_policy")
def main(cfg: DictConfig) -> None:
    """Train a model (policy or codec).

    Trainer selection:
        - ``trainer: policy``     -> PolicyTrainer (+ reconst eval + sim eval)
        - ``trainer: codecs``     -> CodecsTrainer
    """
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
    main()
