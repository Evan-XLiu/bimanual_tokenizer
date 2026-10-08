# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Hydra composition tests for trainer/train_recipe ownership."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


def _compose(config_name: str, overrides: list[str] | None = None):
    config_dir = Path("src/praxis/configs").resolve()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name=config_name, overrides=overrides or [])


def test_default_policy_train_recipe_uses_bar_lrs_without_scalar_lr() -> None:
    cfg = _compose("train_policy")

    assert cfg.policy.name == "bar_oat_pali"
    assert "lr" not in cfg.trainer
    assert dict(cfg.trainer.lrs) == pytest.approx(
        {"vlm": 1.0e-4, "vision": 1.0e-4, "bridge": 1.0e-4}
    )
    assert "action_expert" not in cfg.trainer.lrs
    assert cfg.trainer.lr_scheduler == "cosine"


def test_fm_train_recipe_uses_action_expert_lrs_without_policy_lr_fields() -> None:
    cfg = _compose(
        "train_policy",
        overrides=["policy=fm_pali", "train_recipe/policy@_global_=fm"],
    )

    assert cfg.policy.name == "fm_pali"
    assert "vlm_lr" not in cfg.policy
    assert "vision_lr" not in cfg.policy
    assert "connector_lr" not in cfg.policy
    assert "action_expert_lr" not in cfg.policy
    assert "lr" not in cfg.trainer
    assert dict(cfg.trainer.lrs) == pytest.approx(
        {
            "vlm": 5.0e-5,
            "vision": 5.0e-5,
            "bridge": 5.0e-5,
            "action_expert": 5.0e-5,
        }
    )
    assert cfg.trainer.lr_scheduler == "constant"
    assert cfg.trainer.warmup_steps == 10000


def test_ki_train_recipe_adds_warmstart_without_sim_eval_gate() -> None:
    cfg = _compose(
        "train_policy",
        overrides=["policy=ki_oat_pali", "train_recipe/policy@_global_=ki"],
    )

    assert cfg.policy.name == "ki_oat_pali"
    assert cfg.policy.ce_warmstart_steps == 6000
    assert cfg.trainer.find_unused_parameters is True
    assert cfg.trainer.sim_eval_start_after_step == 0
    assert dict(cfg.trainer.lrs) == pytest.approx(
        {
            "vlm": 5.0e-5,
            "vision": 5.0e-5,
            "bridge": 5.0e-5,
            "action_expert": 5.0e-5,
        }
    )
    assert cfg.trainer.lr_scheduler == "constant"
    assert cfg.trainer.warmup_steps == 6000


def test_codec_train_recipe_uses_scalar_lr_without_family_lrs() -> None:
    cfg = _compose("train_codecs")

    assert cfg.codec.name == "oat"
    assert "lrs" not in cfg.trainer
    assert cfg.trainer.lr == pytest.approx(5.0e-5)
    assert cfg.trainer.lr_scheduler == "constant"
    assert cfg.trainer.max_steps == 300000
