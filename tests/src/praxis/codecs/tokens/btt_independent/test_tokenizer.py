# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from praxis.codecs.factory import available_codecs, make_codec
from praxis.codecs.tokens.btt_independent import (
    BTTIndependentConfig,
    BTTIndependentTokenizer,
)
from praxis.core.types import FeatureSpec, FeatureType, NormMode, PolicyIO


def _tiny_tokenizer() -> BTTIndependentTokenizer:
    return BTTIndependentTokenizer(
        BTTIndependentConfig(
            model_dim=32,
            num_heads=4,
            decoder_nhead=4,
            decoder_nlayer=1,
            dropout=0.0,
        )
    )


def _policy_io() -> PolicyIO:
    return PolicyIO(
        action_feature=FeatureSpec(
            key="action",
            type=FeatureType.ACTION,
            shape=(14,),
            norm_mode=NormMode.MIN_MAX,
        ),
        n_action_steps=32,
    )


def test_factory_exposes_a_distinct_tokenizer_identity() -> None:
    assert "tokens/btt_independent" in available_codecs()

    tokenizer = make_codec(
        {
            "kind": "tokens",
            "name": "btt_independent",
            "model_dim": 32,
            "num_heads": 4,
            "decoder_nhead": 4,
            "decoder_nlayer": 1,
            "dropout": 0.0,
        }
    )

    assert isinstance(tokenizer, BTTIndependentTokenizer)
    assert tokenizer.config.name == "btt_independent"
    assert tokenizer.codebook_size == 1920
    assert tokenizer.max_code_seq_len == 16
    assert tokenizer.block_schedule == (8, 8)


def test_pre_quant_is_independent_but_decoder_remains_joint() -> None:
    tokenizer = _tiny_tokenizer()

    left_ids = {id(parameter) for parameter in tokenizer.left_pre_quant.parameters()}
    right_ids = {id(parameter) for parameter in tokenizer.right_pre_quant.parameters()}
    assert not left_ids & right_ids
    assert tokenizer.decoder.post_quant.in_features == 8
    assert tokenizer.decoder.action_head.out_features == 14
    assert not hasattr(tokenizer, "pre_quant")


def test_token_shapes_and_joint_decoder_contract() -> None:
    tokenizer = _tiny_tokenizer().eval()
    actions = torch.randn(2, 32, 14)

    encoded = tokenizer.tokenize_with_latents(actions)

    assert encoded.left_tokens.shape == (2, 8)
    assert encoded.right_tokens.shape == (2, 8)
    assert encoded.left_latents.shape == (2, 8, 4)
    assert encoded.right_latents.shape == (2, 8, 4)
    assert encoded.token_ids.shape == (2, 16)
    assert tokenizer.detokenize(encoded.token_ids).shape == actions.shape
    paired = tokenizer.decoder.pair_latents(encoded.left_latents, encoded.right_latents)
    torch.testing.assert_close(paired[..., :4], encoded.left_latents)
    torch.testing.assert_close(paired[..., 4:], encoded.right_latents)


def test_training_backward_reaches_both_arm_paths_and_joint_decoder() -> None:
    tokenizer = _tiny_tokenizer()
    actions = torch.randn(2, 32, 14)

    loss, metrics = tokenizer({"action": actions})
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics == {}
    assert tokenizer.left_pre_quant.weight.grad is not None
    assert tokenizer.right_pre_quant.weight.grad is not None
    assert tokenizer.encoder.left_arm_encoder.action_projection.weight.grad is not None
    assert tokenizer.encoder.right_arm_encoder.action_projection.weight.grad is not None
    assert tokenizer.decoder.post_quant.weight.grad is not None
    assert tokenizer.decoder.action_head.weight.grad is not None


def test_pretrained_round_trip_preserves_the_distinct_architecture(tmp_path) -> None:
    tokenizer = _tiny_tokenizer()
    tokenizer.set_io(_policy_io())
    tokenizer.set_normalization_stats(
        {
            "action": {
                "min": torch.full((14,), -2.0),
                "max": torch.full((14,), 2.0),
            }
        },
        modes={"action": NormMode.MIN_MAX},
    )
    tokenizer.save_pretrained(tmp_path)

    restored = BTTIndependentTokenizer.from_pretrained(tmp_path)

    assert restored.config.name == "btt_independent"
    assert restored.io == tokenizer.io
    assert "left_pre_quant.weight" in restored.state_dict()
    assert "right_pre_quant.weight" in restored.state_dict()
    assert "pre_quant.weight" not in restored.state_dict()
    torch.testing.assert_close(
        restored.left_pre_quant.weight, tokenizer.left_pre_quant.weight
    )
    torch.testing.assert_close(
        restored.right_pre_quant.weight, tokenizer.right_pre_quant.weight
    )
