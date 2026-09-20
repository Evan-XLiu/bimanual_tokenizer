# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the bimanual QueST tokenizer."""

from __future__ import annotations

import pytest
import torch

from praxis.codecs.factory import available_codecs, make_codec
from praxis.codecs.tokens.bquest.config import BQuestTokenizerConfig
from praxis.codecs.tokens.bquest.modeling import _causal_mask
from praxis.codecs.tokens.bquest.tokenizer import BQuestTokenizer
from praxis.core.types import FeatureSpec, FeatureType, NormMode, PolicyIO


def _tiny_config(**overrides: object) -> BQuestTokenizerConfig:
    values: dict[str, object] = {
        "encoder_dim": 32,
        "encoder_nhead": 4,
        "encoder_nlayer": 1,
        "decoder_dim": 32,
        "decoder_nhead": 4,
        "decoder_nlayer": 1,
        "dropout": 0.0,
    }
    values.update(overrides)
    return BQuestTokenizerConfig(**values)


def _policy_io() -> PolicyIO:
    return PolicyIO(
        action_feature=FeatureSpec(
            key="action",
            type=FeatureType.ACTION,
            shape=(14,),
            norm_mode=NormMode.MIN_MAX,
        ),
        n_action_steps=16,
    )


def test_factory_and_default_contract() -> None:
    assert "tokens/bquest" in available_codecs()
    tokenizer = make_codec({"kind": "tokens", "name": "bquest", "action_dim": 14})

    assert isinstance(tokenizer, BQuestTokenizer)
    assert tokenizer.codebook_size == 1920
    assert tokenizer.max_code_seq_len == 16
    assert tokenizer.block_schedule == (8, 8)


def test_config_rejects_non_bimanual_geometry() -> None:
    with pytest.raises(ValueError, match="action_dim=14"):
        BQuestTokenizerConfig(action_dim=8)
    with pytest.raises(ValueError, match="action_horizon=32"):
        BQuestTokenizerConfig(action_horizon=16)
    with pytest.raises(ValueError, match="downsample_factor=4"):
        BQuestTokenizerConfig(downsample_factor=2)
    with pytest.raises(ValueError, match="exactly 7"):
        BQuestTokenizerConfig(right_arm_sign=(1.0,) * 6)


def test_token_shape_order_and_decode_paths() -> None:
    tokenizer = BQuestTokenizer(_tiny_config())
    actions = torch.randn(2, 32, 14)

    encoded = tokenizer.tokenize_with_latents(actions)
    assert encoded.coarse_latents.shape == (2, 8, 4)
    assert encoded.latents.shape == (2, 8, 4)
    assert encoded.token_ids.shape == (2, 16)
    torch.testing.assert_close(
        encoded.token_ids[:, :8], tokenizer.coarse_quantizer.codes_to_indices(
            encoded.coarse_latents
        ).long()
    )
    assert tokenizer.decode(encoded.coarse_latents).shape == actions.shape
    assert tokenizer.decode(encoded.latents).shape == actions.shape
    assert tokenizer.detokenize(encoded.token_ids).shape == actions.shape
    torch.testing.assert_close(
        tokenizer._tokens_to_latents(encoded.token_ids), encoded.latents
    )


def test_residual_levels_share_fsq_code_space() -> None:
    tokenizer = BQuestTokenizer(_tiny_config())
    raw_latent = torch.tensor(
        [[[-100.0, -2.0, 0.0, 100.0]]], dtype=torch.float32
    ).expand(1, 8, 4)

    latent_code = tokenizer._latent_to_fsq_code_space(raw_latent)
    coarse, _ = tokenizer.coarse_quantizer(raw_latent)
    residual = latent_code - coarse.detach()
    residual_scale = tokenizer._fsq_residual_scale(residual)
    refine_code, refine_tokens = tokenizer._quantize_fsq_code_space(
        residual / residual_scale, tokenizer.refine_quantizer
    )
    refine = refine_code * residual_scale

    levels = torch.tensor(tokenizer.config.fsq_levels, dtype=torch.float32)
    half_width = levels // 2
    code_max = (levels - 1.0 - half_width) / half_width
    assert torch.all(latent_code >= -1.0)
    assert torch.all(latent_code <= code_max)
    assert torch.all(coarse >= -1.0)
    assert torch.all(coarse <= code_max)
    torch.testing.assert_close(
        refine,
        tokenizer.refine_quantizer.indices_to_embedding(refine_tokens)
        * residual_scale,
    )


def test_refine_path_does_not_reapply_fsq_bound(monkeypatch) -> None:
    tokenizer = BQuestTokenizer(_tiny_config())

    def fail_bound(*_args, **_kwargs):
        raise AssertionError("refine path must not call FSQ.bound")

    monkeypatch.setattr(tokenizer.refine_quantizer, "bound", fail_bound)
    encoded = tokenizer._tokenize_norm_actions(torch.randn(2, 32, 14))
    assert encoded.token_ids.shape == (2, 16)


def test_shared_encoder_and_cross_attention_mask() -> None:
    tokenizer = BQuestTokenizer(_tiny_config())
    calls: list[tuple[int, ...]] = []
    hook = tokenizer.arm_encoder.register_forward_hook(
        lambda _module, inputs, _output: calls.append(tuple(inputs[0].shape))
    )
    try:
        tokenizer._encode_fused_norm_actions(torch.randn(2, 32, 14))
    finally:
        hook.remove()
    assert calls == [(2, 32, 7), (2, 32, 7)]

    mask = _causal_mask(8, torch.device("cpu"))
    assert mask.shape == (8, 8)
    assert not bool(mask[0, 0])
    assert not bool(mask[7, 0])
    assert bool(mask[0, 1])


def test_right_arm_canonicalization_and_stats() -> None:
    tokenizer = BQuestTokenizer(_tiny_config())
    actions = torch.zeros(1, 32, 14)
    actions[..., 7:] = 2.0
    canonical = tokenizer.canonicalize(actions)
    expected = torch.tensor([-2.0, 2.0, 2.0, 2.0, -2.0, -2.0, 2.0])
    torch.testing.assert_close(canonical[0, 0, 7:], expected)
    torch.testing.assert_close(tokenizer.decanonicalize(canonical), actions)

    stats = tokenizer._canonicalize_stats(
        {
            "min": torch.tensor([0.0] * 7 + [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]),
            "max": torch.tensor([1.0] * 7 + [11.0, 21.0, 31.0, 41.0, 51.0, 61.0, 71.0]),
            "mean": torch.zeros(14),
            "std": torch.ones(14),
        }
    )
    expected_min = torch.tensor([-11.0, 0.0, 0.0, 0.0, -51.0, -61.0, 0.0] * 2)
    expected_max = torch.tensor([1.0, 21.0, 31.0, 41.0, 1.0, 1.0, 71.0] * 2)
    torch.testing.assert_close(stats["min"], expected_min)
    torch.testing.assert_close(stats["max"], expected_max)

    no_canonical = BQuestTokenizer(_tiny_config(canonicalize_right_arm=False))
    torch.testing.assert_close(no_canonical.canonicalize(actions), actions)


def test_joint_pair_stats_are_used_by_runtime_normalization() -> None:
    tokenizer = BQuestTokenizer(_tiny_config())
    tokenizer.set_io(_policy_io())
    tokenizer.set_normalization_stats(
        {
            "action": {
                "min": torch.tensor([0.0] * 7 + [-20.0] * 7),
                "max": torch.tensor([10.0] * 7 + [-10.0] * 7),
            }
        },
        modes={"action": NormMode.MIN_MAX},
    )

    actions = torch.zeros(1, 32, 14)
    actions[..., 0] = 20.0
    actions[..., 7] = -20.0
    normalized = tokenizer._normalize_action_tensor(actions)

    # Joint 0 uses the shared canonical range [0, 20] for both arms.
    torch.testing.assert_close(normalized[..., 0], normalized[..., 7])


def test_forward_is_two_reconstruction_losses_and_backpropagates() -> None:
    tokenizer = BQuestTokenizer(_tiny_config())
    actions = torch.randn(2, 32, 14)
    loss, metrics = tokenizer({"action": actions})

    assert torch.isfinite(loss)
    assert set(metrics) == {"coarse_recon_loss", "refine_recon_loss"}
    torch.testing.assert_close(loss, torch.tensor(sum(metrics.values())), rtol=1e-5, atol=1e-5)
    loss.backward()
    assert tokenizer.arm_encoder.action_proj.weight.grad is not None
    assert tokenizer.fusion.weight.grad is not None
    assert tokenizer.decoder.action_head.weight.grad is not None


def test_pretrained_round_trip_preserves_normalization(tmp_path) -> None:
    tokenizer = BQuestTokenizer(_tiny_config())
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

    restored = BQuestTokenizer.from_pretrained(tmp_path)
    assert restored.io == tokenizer.io
    assert restored.config.fsq_levels == (8, 8, 6, 5)
    assert restored.normalize_action is not None
    torch.testing.assert_close(
        restored.normalize_action.stats["action"]["min"],
        tokenizer.normalize_action.stats["action"]["min"],
    )
