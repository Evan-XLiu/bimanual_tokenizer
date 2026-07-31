# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Policy catalog and registry consistency tests."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml

from praxis.policies.backbones.pali.vlm import PALIGEMMA_MODEL_NAME
from praxis.policies.backbones.qwen.vlm import QWEN3_VL_MODEL_NAME
from praxis.policies.backbones.smol.vlm import SMOLVLM_MODEL_NAME
from praxis.policies.bar.backbones.pali.pali_config import (
    PaliPolicyConfig as BarPaliPolicyConfig,
)
from praxis.policies.bar.backbones.qwen.qwen_config import (
    QwenPolicyConfig as BarQwenPolicyConfig,
)
from praxis.policies.bar.backbones.smol.smol_config import (
    SmolPolicyConfig as BarSmolPolicyConfig,
)
from praxis.policies.catalog import (
    EXTRA_POLICY_SPECS,
    POLICY_FAMILIES,
    iter_policy_specs,
    policy_name,
    policy_spec,
    supported_policy_names,
)
from praxis.policies.factory import available_policies, get_policy_class
from praxis.policies.fm.backbones.pali.pali_config import (
    PaliPolicyConfig as FmPaliPolicyConfig,
)
from praxis.policies.fm.backbones.qwen.qwen_config import (
    QwenPolicyConfig as FmQwenPolicyConfig,
)
from praxis.policies.fm.backbones.smol.smol_config import (
    SmolPolicyConfig as FmSmolPolicyConfig,
)
from praxis.policies.ki.backbones.pali.pali_config import (
    PaliPolicyConfig as KiPaliPolicyConfig,
)
from praxis.policies.ki.backbones.qwen.qwen_config import (
    QwenPolicyConfig as KiQwenPolicyConfig,
)
from praxis.policies.ki.backbones.smol.smol_config import (
    SmolPolicyConfig as KiSmolPolicyConfig,
)

_BACKBONE_MODEL_NAMES = {
    "pali": PALIGEMMA_MODEL_NAME,
    "qwen": QWEN3_VL_MODEL_NAME,
    "smol": SMOLVLM_MODEL_NAME,
}

_BACKBONE_CONFIG_CLASSES = {
    ("bar", "pali"): BarPaliPolicyConfig,
    ("bar", "qwen"): BarQwenPolicyConfig,
    ("bar", "smol"): BarSmolPolicyConfig,
    ("ki", "pali"): KiPaliPolicyConfig,
    ("ki", "qwen"): KiQwenPolicyConfig,
    ("ki", "smol"): KiSmolPolicyConfig,
    ("fm", "pali"): FmPaliPolicyConfig,
    ("fm", "qwen"): FmQwenPolicyConfig,
    ("fm", "smol"): FmSmolPolicyConfig,
}


def test_catalog_names_match_top_level_policy_yaml_aliases() -> None:
    policy_config_root = Path("src/praxis/configs/policy").resolve()
    alias_names = sorted(path.stem for path in policy_config_root.glob("*.yaml"))

    assert sorted(supported_policy_names()) == alias_names


def test_default_registry_is_seeded_from_policy_catalog() -> None:
    assert set(available_policies()) == set(supported_policy_names())


def test_catalog_import_paths_resolve_policy_aliases() -> None:
    for spec in iter_policy_specs():
        module = importlib.import_module(spec.module_path)

        assert getattr(module, spec.class_name) is get_policy_class(spec.name)


def test_top_level_policy_yaml_aliases_are_thin_axis_compositions() -> None:
    policy_config_root = Path("src/praxis/configs/policy").resolve()

    for spec in iter_policy_specs():
        alias_path = policy_config_root / f"{spec.name}.yaml"
        alias_config = yaml.safe_load(alias_path.read_text())
        defaults: list[dict[str, str] | str] = [
            {f"{spec.method}/backbone@_here_": spec.backbone}
        ]
        if spec.codec is not None:
            defaults.append({f"{spec.method}/codec@_here_": spec.codec})
        defaults.append("_self_")

        assert alias_config == {"defaults": defaults, "name": spec.name}


def test_method_axis_policy_yaml_groups_match_catalog() -> None:
    policy_config_root = Path("src/praxis/configs/policy").resolve()
    method_dirs = sorted(
        path.name for path in policy_config_root.iterdir() if path.is_dir()
    )

    assert method_dirs == sorted(family.method for family in POLICY_FAMILIES)
    for family in POLICY_FAMILIES:
        method_root = policy_config_root / family.method
        assert sorted(
            path.stem for path in (method_root / "backbone").glob("*.yaml")
        ) == sorted(family.backbones)
        codec_root = method_root / "codec"
        if family.codecs is None:
            assert not codec_root.exists()
        else:
            extra_codecs = {
                spec.codec
                for spec in EXTRA_POLICY_SPECS
                if spec.method == family.method and spec.codec is not None
            }
            assert sorted(path.stem for path in codec_root.glob("*.yaml")) == sorted(
                set(family.codecs) | extra_codecs
            )


def test_backbone_axis_yaml_defaults_match_config_classes() -> None:
    policy_config_root = Path("src/praxis/configs/policy").resolve()
    shared_fields = (
        "vlm_model_name",
        "vlm_dtype",
        "freeze_vision_encoder",
        "use_lora",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "gradient_checkpointing",
        "default_instruction",
        "action_dim",
        "n_obs_steps",
        "action_horizon",
        "n_action_steps",
    )
    flow_fields = (
        "flow_loss_weight",
        "flow_sampling_steps",
        "flow_time_sampling",
        "flow_beta_alpha",
        "flow_beta_beta",
        "flow_min_time",
        "expert_hidden_size",
        "expert_width_multiplier",
        "expert_num_layers",
        "expert_self_attn_every_n_layers",
        "expert_dropout",
        "expert_time_min_period",
        "expert_time_max_period",
    )
    action_token_fields = (
        "action_token_template",
        "action_embed_init",
    )

    for family in POLICY_FAMILIES:
        for backbone in family.backbones:
            axis_path = (
                policy_config_root / family.method / "backbone" / f"{backbone}.yaml"
            )
            axis_config = yaml.safe_load(axis_path.read_text())
            config = _BACKBONE_CONFIG_CLASSES[(family.method, backbone)]()

            assert axis_config["vlm_model_name"] == _BACKBONE_MODEL_NAMES[backbone]
            for field in shared_fields:
                assert axis_config[field] == getattr(config, field)
            for field in flow_fields:
                if family.method == "bar":
                    assert field not in axis_config
                else:
                    assert axis_config[field] == getattr(config, field)
            assert (
                tuple(axis_config["lora_target_modules"]) == config.lora_target_modules
            )
            assert (
                tuple(axis_config["lora_modules_to_save"])
                == config.lora_modules_to_save
            )
            for field in action_token_fields:
                if family.method == "fm":
                    assert field not in axis_config
                else:
                    assert axis_config[field] == getattr(config, field)
            assert "ce_warmstart_steps" not in axis_config
            assert "vlm_lr" not in axis_config
            assert "vision_lr" not in axis_config
            assert "connector_lr" not in axis_config
            assert "action_expert_lr" not in axis_config

            if backbone == "smol":
                assert tuple(axis_config["resize_imgs_with_padding"]) == (
                    config.resize_imgs_with_padding
                )
            else:
                assert "resize_imgs_with_padding" not in axis_config


def test_catalog_rejects_unsupported_policy_matrix_entries() -> None:
    with pytest.raises(ValueError, match="Unsupported policy codec 'boat'"):
        policy_spec("ki", "boat", "pali")
    with pytest.raises(ValueError, match="does not use a codec"):
        policy_spec("fm", "oat", "pali")


def test_policy_name_validates_and_normalizes_axes() -> None:
    assert policy_name("KI", "OAT", "PALI") == "ki_oat_pali"
    assert policy_name("FM", None, "PALI") == "fm_pali"
