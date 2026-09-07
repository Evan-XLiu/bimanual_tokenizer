# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the paired bimanual OAT tokenizer."""

from __future__ import annotations

import torch
import torch.nn as nn

from praxis.codecs.factory import available_codecs, make_codec
from praxis.codecs.tokens.bimanualoat.config import BimanualOATTokenizerConfig
from praxis.codecs.tokens.bimanualoat.convention import apply_bimanual_convention
from praxis.codecs.tokens.bimanualoat.quantizer import (
    PairedFiniteScalarQuantization,
)
from praxis.codecs.tokens.bimanualoat.tokenizer import BimanualOATTokenizer
from praxis.core.types import FeatureSpec, FeatureType, NormMode, PolicyIO


def _tiny_config(**overrides) -> BimanualOATTokenizerConfig:
    values = {
        "action_dim": 14,
        "action_horizon": 8,
        "latent_horizon": 4,
        "emb_dim": 32,
        "head_dim": 8,
        "encoder_depth": 1,
        "decoder_depth": 1,
        "dropout": 0.0,
    }
    values.update(overrides)
    return BimanualOATTokenizerConfig(**values)


def _policy_io(action_dim: int = 14) -> PolicyIO:
    return PolicyIO(
        action_feature=FeatureSpec(
            key="action",
            type=FeatureType.ACTION,
            shape=(action_dim,),
            norm_mode=NormMode.MIN_MAX,
        ),
        n_action_steps=8,
    )


def test_default_stream_levels_and_paired_vocabulary() -> None:
    tokenizer = BimanualOATTokenizer(config=_tiny_config())

    assert tokenizer.config.common_quantizer_levels == (8, 6)
    assert tokenizer.config.relative_quantizer_levels == (8, 6)
    assert tokenizer.quantizer.common_codebook_size == 48
    assert tokenizer.quantizer.relative_codebook_size == 48
    assert tokenizer.codebook_size == 2304


def test_pairing_is_invertible_for_every_token_id() -> None:
    quantizer = PairedFiniteScalarQuantization(
        common_levels=(8, 6), relative_levels=(8, 6)
    )
    paired = torch.arange(2304)

    common, relative = quantizer.unpair_tokens(paired)

    torch.testing.assert_close(quantizer.pair_tokens(common, relative), paired)


def test_right_arm_convention_is_self_inverse() -> None:
    sign = torch.tensor([-1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0])
    actions = torch.randn(2, 8, 14)

    canonical = apply_bimanual_convention(actions, sign)

    torch.testing.assert_close(apply_bimanual_convention(canonical, sign), actions)
    torch.testing.assert_close(canonical[..., :7], actions[..., :7])
    torch.testing.assert_close(canonical[..., 7:], actions[..., 7:] * sign)


def test_config_validates_bimanual_geometry() -> None:
    try:
        _tiny_config(action_dim=13)
    except ValueError as exc:
        assert "even action_dim" in str(exc)
    else:
        raise AssertionError("odd action_dim should fail")

    try:
        _tiny_config(action_dim=4)
    except ValueError as exc:
        assert "right_arm_sign length" in str(exc)
    else:
        raise AssertionError("mismatched right_arm_sign should fail")


def test_common_and_relative_features_use_shared_arm_embedding() -> None:
    tokenizer = BimanualOATTokenizer(
        config=_tiny_config(
            action_dim=4,
            right_arm_sign=(-1.0, 1.0),
            emb_dim=2,
            head_dim=1,
        )
    )
    projection = tokenizer.encoder.shared_arm_emb.patch_proj
    assert isinstance(projection, nn.Linear)
    with torch.no_grad():
        projection.weight.copy_(torch.eye(2))
        projection.bias.zero_()
    sample = torch.tensor([[[2.0, 4.0, 6.0, 0.0]]])
    position = tokenizer.encoder.temporal_pos_emb(
        [1], device=sample.device, dtype=sample.dtype
    )

    common, relative = tokenizer.encoder.encode_arm_features(sample)

    torch.testing.assert_close(common - position, torch.tensor([[[4.0, 2.0]]]))
    torch.testing.assert_close(relative - position, torch.tensor([[[-2.0, 2.0]]]))
    assert (
        tokenizer.encoder.common_registers is not tokenizer.encoder.relative_registers
    )


def test_forward_tokenize_decode_prefix_and_gradients() -> None:
    tokenizer = BimanualOATTokenizer(config=_tiny_config())
    actions = torch.randn(2, 8, 14)

    loss, metrics = tokenizer({"action": actions})
    encoded = tokenizer.tokenize_with_latents(actions)

    assert loss.ndim == 0 and torch.isfinite(loss)
    assert metrics == {}
    assert encoded.latents.shape == (2, 4, 4)
    assert encoded.token_ids.shape == (2, 4)
    assert tokenizer.decode(encoded.latents).shape == actions.shape
    assert tokenizer.detokenize(encoded.token_ids).shape == actions.shape
    assert tokenizer.detokenize([[1, 2], [3, 4, 5]]).shape == actions.shape

    loss.backward()
    assert tokenizer.encoder.common_registers.grad is not None
    assert tokenizer.encoder.relative_registers.grad is not None


def test_nested_dropout_uses_the_same_prefix_for_both_streams() -> None:
    tokenizer = BimanualOATTokenizer(config=_tiny_config())
    decoder = tokenizer.decoder
    common = torch.randn(2, 4, 32)
    relative = torch.randn(2, 4, 32)

    common_out, relative_out = decoder._synchronized_nested_dropout(
        common, relative, [2, 4]
    )

    torch.testing.assert_close(common_out[0, :2], common[0, :2])
    torch.testing.assert_close(relative_out[0, :2], relative[0, :2])
    common_mask = decoder.common_nested_dropout.dropout_mask_token
    relative_mask = decoder.relative_nested_dropout.dropout_mask_token
    torch.testing.assert_close(
        common_out[0, 2:], common_mask.expand_as(common_out[0, 2:])
    )
    torch.testing.assert_close(
        relative_out[0, 2:], relative_mask.expand_as(relative_out[0, 2:])
    )
    torch.testing.assert_close(common_out[1], common[1])
    torch.testing.assert_close(relative_out[1], relative[1])


def test_normalization_canonicalizes_and_pools_corresponding_joints() -> None:
    tokenizer = BimanualOATTokenizer(
        config=_tiny_config(action_dim=4, right_arm_sign=(-1.0, 1.0))
    )
    tokenizer.set_normalization_stats(
        {
            "action": {
                "min": torch.tensor([0.0, 10.0, -5.0, 20.0]),
                "max": torch.tensor([2.0, 12.0, -1.0, 24.0]),
                "mean": torch.tensor([1.0, 11.0, -3.0, 22.0]),
                "std": torch.tensor([1.0, 2.0, 4.0, 6.0]),
                "count": torch.tensor(100),
            }
        },
        modes={"action": NormMode.MIN_MAX},
    )
    assert tokenizer.normalize_action is not None
    pooled = tokenizer.normalize_action.stats["action"]

    torch.testing.assert_close(pooled["min"], torch.tensor([0.0, 10.0, 0.0, 10.0]))
    torch.testing.assert_close(pooled["max"], torch.tensor([5.0, 24.0, 5.0, 24.0]))
    torch.testing.assert_close(pooled["mean"], torch.tensor([2.0, 16.5, 2.0, 16.5]))
    expected_std = torch.tensor([9.5**0.5, 50.25**0.5]).repeat(2)
    torch.testing.assert_close(pooled["std"], expected_std)

    physical = torch.tensor([[[1.0, 12.0, -1.0, 12.0]]])
    normalized = tokenizer._normalize_action_tensor(physical)
    torch.testing.assert_close(normalized[..., :2], normalized[..., 2:])
    restored = tokenizer._unnormalize_action_tensor(normalized)
    torch.testing.assert_close(restored, physical)


def test_factory_and_pretrained_round_trip(tmp_path) -> None:
    assert "tokens/bimanualoat" in available_codecs()
    tokenizer = make_codec(
        {
            "kind": "tokens",
            "name": "bimanualoat",
            **_tiny_config().to_dict(),
        }
    )
    assert isinstance(tokenizer, BimanualOATTokenizer)
    tokenizer.set_io(_policy_io())
    tokenizer.set_normalization_stats(
        {"action": {"min": torch.full((14,), -2.0), "max": torch.full((14,), 2.0)}},
        modes={"action": NormMode.MIN_MAX},
    )
    tokenizer.save_pretrained(tmp_path)

    restored = BimanualOATTokenizer.from_pretrained(tmp_path)

    assert restored.codebook_size == 2304
    assert restored.io == tokenizer.io
    assert restored.normalize_action is not None
    torch.testing.assert_close(
        restored.normalize_action.stats["action"]["min"],
        tokenizer.normalize_action.stats["action"]["min"],
    )
