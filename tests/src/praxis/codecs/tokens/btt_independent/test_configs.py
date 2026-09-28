# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from praxis.policies.factory import get_policy_class


def _compose(config_name: str, overrides: list[str]):
    config_dir = Path("src/praxis/configs").resolve()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name=config_name, overrides=overrides)


def test_tokenizer_training_config_composes() -> None:
    cfg = _compose(
        "train_codecs",
        [
            "codecs@codec=tokens/btt_independent",
            "dataset=robotwin_new_demo_clean",
        ],
    )

    assert cfg.codec.name == "btt_independent"
    assert cfg.codec.kind == "tokens"
    assert cfg.codec.action_dim == 14
    assert cfg.codec.action_horizon == 32
    assert cfg.codec.latent_horizon_per_arm == 8
    assert cfg.codec.right_arm_sign == [-1, 1, 1, 1, -1, -1, 1]


def test_tokenizer_config_changes_only_the_intended_ablation_axes() -> None:
    config_root = Path("src/praxis/configs/codecs/tokens")
    baseline = OmegaConf.to_container(OmegaConf.load(config_root / "btt.yaml"))
    independent = OmegaConf.to_container(
        OmegaConf.load(config_root / "btt_independent.yaml")
    )
    assert isinstance(baseline, dict)
    assert isinstance(independent, dict)

    normalization_keys = {
        "shared_action_min",
        "shared_action_max",
        "left_action_min",
        "left_action_max",
        "right_action_min",
        "right_action_max",
    }
    for key, baseline_value in baseline.items():
        if key == "name" or key in normalization_keys:
            continue
        assert independent[key] == baseline_value

    assert independent["name"] == "btt_independent"
    assert independent["right_arm_sign"] == baseline["right_arm_sign"]
    assert "shared_action_min" not in independent
    assert "shared_action_max" not in independent


@pytest.mark.parametrize("backbone", ("pali", "qwen", "smol"))
def test_bar_policy_configs_and_classes_are_connected(backbone: str) -> None:
    policy_name = f"bar_btt_independent_{backbone}"
    cfg = _compose("train_policy", [f"policy={policy_name}"])

    assert cfg.policy.name == policy_name
    assert cfg.policy.action_tokenizer.name == "btt_independent"
    assert cfg.policy.action_tokenizer.kind == "tokens"
    assert cfg.policy.freeze_action_tokenizer is True
    assert cfg.policy.block_ce_weighting == "block_mean"
    assert cfg.policy.ce_vocab_scope == "full"
    assert get_policy_class(policy_name).__name__ == (
        f"BTTIndependent{backbone.capitalize()}Policy"
    )


def test_paligemma_policy_uses_the_expected_backbone() -> None:
    cfg = _compose("train_policy", ["policy=bar_btt_independent_pali"])

    assert cfg.policy.vlm_model_name == "google/paligemma2-3b-mix-224"
