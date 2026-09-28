# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from praxis.codecs.factory import make_codec
from praxis.codecs.tokens.btt import BTTTokenizer


def _compose_ablation_config():
    config_dir = Path("src/praxis/configs").resolve()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(
            config_name="train_codecs",
            overrides=["codecs@codec=tokens/btt_wo_convention"],
        )


def test_wo_convention_config_changes_only_action_preprocessing() -> None:
    cfg = _compose_ablation_config()
    baseline_path = Path("src/praxis/configs/codecs/tokens/btt.yaml")
    baseline = OmegaConf.load(baseline_path)
    ablation = cfg.codec

    preprocessing_keys = {
        "right_arm_sign",
        "shared_action_min",
        "shared_action_max",
    }
    for key in baseline:
        if key not in preprocessing_keys:
            assert ablation[key] == baseline[key]

    assert list(ablation.right_arm_sign) == [1] * 7
    assert list(ablation.shared_action_min) == pytest.approx(
        [-2.9455960, -0.5742835, -0.1190902, -1.9893787, -1.6339467, -3.4687879, 0.0]
    )
    assert list(ablation.shared_action_max) == pytest.approx(
        [1.3892723, 3.3889313, 3.4514999, 1.9530925, 1.5632697, 3.4247944, 1.0]
    )


def test_wo_convention_preserves_raw_arms_and_shared_normalization() -> None:
    cfg = _compose_ablation_config()
    cfg.codec.model_dim = 32
    cfg.codec.num_heads = 4
    cfg.codec.decoder_nhead = 4
    cfg.codec.decoder_nlayer = 1
    cfg.codec.dropout = 0.0
    tokenizer = make_codec(cfg.codec)

    assert isinstance(tokenizer, BTTTokenizer)
    left = torch.randn(2, 32, 7)
    right = torch.randn(2, 32, 7)
    split_left, split_right = tokenizer.encoder.split_and_canonicalize(
        torch.cat((left, right), dim=-1)
    )
    torch.testing.assert_close(split_left, left)
    torch.testing.assert_close(split_right, right)

    lower = tokenizer.encoder.shared_action_min.view(1, 1, 7).expand(1, 32, 7)
    upper = tokenizer.encoder.shared_action_max.view(1, 1, 7).expand(1, 32, 7)
    norm_left, norm_right = tokenizer.encoder.split_canonicalize_and_normalize(
        torch.cat((lower, upper), dim=-1)
    )
    torch.testing.assert_close(norm_left, -torch.ones_like(norm_left))
    torch.testing.assert_close(norm_right, torch.ones_like(norm_right))
    restored = tokenizer.encoder.unnormalize_and_decanonicalize(norm_left, norm_right)
    torch.testing.assert_close(restored, torch.cat((lower, upper), dim=-1))
