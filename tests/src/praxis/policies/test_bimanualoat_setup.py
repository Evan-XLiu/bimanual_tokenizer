# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Policy registry and Hydra tests for bimanual OAT."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from praxis.policies.factory import get_policy_class
from scripts.infra.launch.commands import (
    build_policy_command,
    build_tokenizer_command,
)

_POLICIES = tuple(
    f"{method}_bimanualoat_{backbone}"
    for method in ("bar", "ki")
    for backbone in ("pali", "qwen", "smol")
)


@pytest.mark.parametrize("policy_name", _POLICIES)
def test_bimanualoat_policy_registration_and_hydra(policy_name: str) -> None:
    policy_class = get_policy_class(policy_name)
    config_dir = Path("src/praxis/configs").resolve()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="train_policy", overrides=[f"policy={policy_name}"])

    assert policy_class.__name__.startswith("BimanualOAT")
    assert cfg.policy.name == policy_name
    assert cfg.policy.action_tokenizer.name == "bimanualoat"
    assert cfg.policy.action_tokenizer.kind == "tokens"


def test_bimanualoat_codec_training_config_composes() -> None:
    config_dir = Path("src/praxis/configs").resolve()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="train_codecs",
            overrides=["codecs@codec=tokens/bimanualoat"],
        )

    assert cfg.codec.name == "bimanualoat"
    assert list(cfg.codec.common_quantizer_levels) == [8, 6]
    assert list(cfg.codec.relative_quantizer_levels) == [8, 6]


def test_bimanualoat_launcher_commands_select_codec_and_bar_pali() -> None:
    tokenizer_command = build_tokenizer_command(
        codec="bimanualoat",
        dataset="robotwin2_clean",
        gpus=1,
        num_nodes=1,
        main_process_ip=None,
        main_process_port=29500,
        run_name="bimanualoat_test",
        output_dir="outputs/bimanualoat_test",
        skip_codec_fit=False,
        extra_overrides=[],
    )
    policy_command = build_policy_command(
        method="bar",
        codec="bimanualoat",
        backbone="pali",
        dataset="robotwin2_clean",
        gpus=1,
        num_nodes=1,
        main_process_ip=None,
        main_process_port=29500,
        run_name="bar_bimanualoat_pali_test",
        output_dir="outputs/bar_bimanualoat_pali_test",
        tokenizer_ckpt="outputs/bimanualoat/pretrained_model",
        extra_overrides=[],
    )

    assert "codecs@codec=tokens/bimanualoat" in tokenizer_command
    assert "policy=bar_bimanualoat_pali" in policy_command
    assert (
        "policy.action_tokenizer.pretrained_path="
        "outputs/bimanualoat/pretrained_model" in policy_command
    )
