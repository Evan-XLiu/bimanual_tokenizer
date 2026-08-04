# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Registry and Hydra coverage for KI policy surfaces."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from praxis.policies.catalog import (
    KI_POLICY_BACKBONES,
    KI_POLICY_CODECS,
    policy_name,
)
from praxis.policies.factory import (
    available_policies,
    get_policy_class,
)

_EXPECTED_KI_POLICIES = tuple(
    policy_name("ki", codec, backbone)
    for backbone in KI_POLICY_BACKBONES
    for codec in KI_POLICY_CODECS
)


def test_registry_exposes_requested_ki_policy_matrix_without_boat() -> None:
    names = set(available_policies())

    assert set(_EXPECTED_KI_POLICIES).issubset(names)
    assert not any(name.startswith("ki_boat_") for name in names)


def test_xtokenizer_qwen_uses_offline_targets_without_runtime_checkpoint() -> None:
    config_dir = Path("src/praxis/configs").resolve()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="train_policy", overrides=["policy=ki_xtokenizer_qwen"]
        )

    assert cfg.policy.name == "ki_xtokenizer_qwen"
    assert cfg.policy.action_tokenizer.name == "offline_targets"
    assert cfg.policy.action_tokenizer.codebook_size == 2048
    assert list(cfg.policy.action_tokenizer.block_schedule) == [1] * 32
    assert cfg.policy.action_token_target_key == "action_token_ids"
    assert "pretrained_path" not in cfg.policy.action_tokenizer


@pytest.mark.parametrize("policy_name", _EXPECTED_KI_POLICIES)
def test_ki_policy_class_imports(policy_name: str) -> None:
    policy_cls = get_policy_class(policy_name)

    assert policy_cls.__name__.endswith("Policy")


@pytest.mark.parametrize("policy_name", _EXPECTED_KI_POLICIES)
def test_ki_hydra_policy_config_composes(policy_name: str) -> None:
    config_dir = Path("src/praxis/configs").resolve()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="train_policy", overrides=[f"policy={policy_name}"])

    codec = policy_name.split("_")[1]
    assert cfg.policy.name == policy_name
    assert cfg.policy.action_tokenizer.name == codec
    assert cfg.policy.action_tokenizer.kind == "tokens"
    assert cfg.policy.freeze_action_tokenizer is True
    assert cfg.policy.block_ce_weighting == "block_mean"
    assert cfg.policy.ce_vocab_scope == "action"
    assert bool(cfg.policy.get("ce_supervise_eos", False)) is (codec == "fast")


@pytest.mark.parametrize("policy_name", _EXPECTED_KI_POLICIES)
def test_ki_hydra_policy_config_accepts_full_ce_vocab_scope(
    policy_name: str,
) -> None:
    config_dir = Path("src/praxis/configs").resolve()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="train_policy",
            overrides=[f"policy={policy_name}", "policy.ce_vocab_scope=full"],
        )

    assert cfg.policy.ce_vocab_scope == "full"
