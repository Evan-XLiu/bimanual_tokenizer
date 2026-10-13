# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Registry and Hydra coverage for FM policy surfaces."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from praxis.policies.catalog import FM_POLICY_BACKBONES, policy_name
from praxis.policies.factory import available_policies, get_policy_class

_EXPECTED_FM_POLICIES = tuple(
    policy_name("fm", None, backbone) for backbone in FM_POLICY_BACKBONES
)


def test_registry_exposes_codecless_fm_policy_matrix() -> None:
    names = set(available_policies())

    assert set(_EXPECTED_FM_POLICIES).issubset(names)
    assert not any(name.startswith("fm_oat_") for name in names)


@pytest.mark.parametrize("policy_name", _EXPECTED_FM_POLICIES)
def test_fm_policy_class_imports(policy_name: str) -> None:
    policy_cls = get_policy_class(policy_name)

    assert policy_cls.__name__.endswith("Policy")


@pytest.mark.parametrize("policy_name", _EXPECTED_FM_POLICIES)
def test_fm_hydra_policy_config_composes_without_tokenizer(policy_name: str) -> None:
    config_dir = Path("src/praxis/configs").resolve()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="train_policy", overrides=[f"policy={policy_name}"])

    assert cfg.policy.name == policy_name
    assert "action_tokenizer" not in cfg.policy
    assert "ce_loss_weight" not in cfg.policy
    assert cfg.policy.gradient_checkpointing is (policy_name == "fm_molmo")
