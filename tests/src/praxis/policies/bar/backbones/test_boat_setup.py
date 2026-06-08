# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Cross-backbone BOAT registration and Hydra setup tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from praxis.policies.bar.variants import (
    BOATPaliPolicyConfig,
    BOATQwenPolicyConfig,
    BOATSmolPolicyConfig,
)
from praxis.policies.factory import available_policies

_BOAT_CONFIGS = (
    ("bar_boat_pali", BOATPaliPolicyConfig),
    ("bar_boat_qwen", BOATQwenPolicyConfig),
    ("bar_boat_smol", BOATSmolPolicyConfig),
)


@pytest.mark.parametrize("policy_name,_config_cls", _BOAT_CONFIGS)
def test_boat_policy_is_registered(policy_name, _config_cls):
    assert policy_name in available_policies()


@pytest.mark.parametrize("policy_name,_config_cls", _BOAT_CONFIGS)
def test_boat_hydra_policy_config_is_fixed_budget(policy_name, _config_cls):
    config_dir = Path("src/praxis/configs").resolve()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="train_policy", overrides=[f"policy={policy_name}"])

    assert cfg.policy.name == policy_name
    assert cfg.policy.action_tokenizer.name == "boat"
    assert OmegaConf.select(cfg, "policy.decode_stop_mode") is None
    assert cfg.policy.decode_keep_k is None
    assert cfg.policy.block_ce_weighting == "block_mean"
    assert OmegaConf.select(cfg, "policy.stop_loss_weight") is None


@pytest.mark.parametrize("policy_name,config_cls", _BOAT_CONFIGS)
def test_boat_config_defaults_are_fixed_budget(policy_name, config_cls):
    cfg = config_cls.from_dict({"name": policy_name})

    assert cfg.name == policy_name
    assert cfg.action_tokenizer == {
        "kind": "tokens",
        "name": "boat",
        "pretrained_path": None,
    }
    assert cfg.decode_stop_mode == "fixed_budget"
    assert cfg.decode_keep_k is None
    assert cfg.block_ce_weighting == "block_mean"
