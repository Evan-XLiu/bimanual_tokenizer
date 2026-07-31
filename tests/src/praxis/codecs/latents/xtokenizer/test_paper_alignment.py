# SPDX-FileCopyrightText: 2026 Xiaoshen Han
#
# SPDX-License-Identifier: Apache-2.0

"""Paper-contract tests for X-Tokenizer semantic pretraining objectives."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import pytest
import torch
import torch.nn as nn
from torch import Tensor

import praxis.codecs.latents.xtokenizer.codec as codec_module
import praxis.codecs.latents.xtokenizer.quantizer as quantizer_module
from praxis.codecs.latents.xtokenizer import XTokenizerCodec, XTokenizerModelConfig
from praxis.codecs.latents.xtokenizer.codec import MAMHead, NextFramePredictionHead


def _tiny_config(**overrides: object) -> XTokenizerModelConfig:
    values: dict[str, Any] = {
        "action_dim": 10,
        "action_horizon": 4,
        "compression_ratio": 2,
        "hidden_dim": 8,
        "latent_dim": 8,
        "encoder_num_heads": 2,
        "encoder_depth": 1,
        "decoder_num_heads": 2,
        "decoder_depth": 1,
        "mam_num_heads": 4,
        "mam_depth": 2,
        "prediction_num_heads": 4,
        "prediction_depth": 2,
        "dropout": 0.0,
        "codebook_size": 8,
        "num_quantizers": 4,
        "dead_code_threshold": 0,
        "embodiment_vocab_size": 2,
        "vl_feature_dim": 4,
        "reconstruction_weight": 0.0,
        "commitment_weight": 0.0,
        "rotation_weight": 0.0,
        "dct_weight": 0.0,
        "smoothness_weight": 0.0,
        "mam_weight": 0.0,
        "alignment_weight": 0.0,
        "prediction_weight": 0.0,
    }
    values.update(overrides)
    return XTokenizerModelConfig.from_dict(values)


def _required_codec_helper(name: str) -> Callable[..., Any]:
    helper = getattr(codec_module, name, None)
    assert callable(helper), f"codec.py must provide {name} for the paper contract"
    return cast(Callable[..., Any], helper)


def test_paper_objective_weights_are_the_model_defaults() -> None:
    config = XTokenizerModelConfig.from_dict({"action_dim": 10})

    actual = {
        "reconstruction_weight": config.reconstruction_weight,
        "commitment_weight": config.commitment_weight,
        "rotation_weight": config.rotation_weight,
        "dct_weight": config.dct_weight,
        "smoothness_weight": config.smoothness_weight,
        "mam_weight": config.mam_weight,
        "alignment_weight": config.alignment_weight,
        "prediction_weight": config.prediction_weight,
    }
    assert actual == pytest.approx(
        {
            "reconstruction_weight": 1.0,
            "commitment_weight": 0.25,
            "rotation_weight": 0.2,
            "dct_weight": 0.5,
            "smoothness_weight": 0.3,
            "mam_weight": 0.1,
            "alignment_weight": 0.5,
            "prediction_weight": 0.2,
        }
    )


def test_mam_uses_two_dedicated_four_head_transformer_layers() -> None:
    head = MAMHead(
        _tiny_config(
            action_horizon=32,
            compression_ratio=4,
            hidden_dim=16,
            latent_dim=16,
            encoder_num_heads=8,
            decoder_num_heads=8,
            dropout=0.1,
        )
    )

    layers = cast(list[nn.TransformerEncoderLayer], list(head.context.layers))
    assert len(layers) == 2
    assert all(layer.self_attn.num_heads == 4 for layer in layers)
    assert all(layer.dropout.p == pytest.approx(0.1) for layer in layers)


def test_mam_uses_learned_absolute_positions_with_paper_maximum_length() -> None:
    head = MAMHead(
        _tiny_config(
            action_horizon=32,
            compression_ratio=4,
            hidden_dim=16,
            latent_dim=16,
            encoder_num_heads=8,
            decoder_num_heads=8,
        )
    )

    assert isinstance(head.position, nn.Embedding)
    assert head.position.num_embeddings == 16
    assert head.position.embedding_dim == 16
    assert head.position.weight.requires_grad


def test_q0_straight_through_uses_hard_codeword_with_encoder_only_gradient() -> None:
    straight_through = getattr(quantizer_module, "_q0_straight_through", None)
    assert callable(straight_through)
    latents = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]]],
        requires_grad=True,
    )
    q0_codewords = torch.tensor(
        [[[30.0, 40.0], [10.0, 20.0]]],
        requires_grad=True,
    )

    q0_st = cast(Tensor, straight_through(latents, q0_codewords))

    torch.testing.assert_close(q0_st, torch.tensor([[[30.0, 40.0], [10.0, 20.0]]]))
    q0_st.sum().backward()
    torch.testing.assert_close(latents.grad, torch.ones_like(latents))
    assert q0_codewords.grad is None


def test_mam_consumes_continuous_q0_representations_without_id_embedding() -> None:
    head = MAMHead(_tiny_config())
    representations = torch.randn(2, 2, 8, requires_grad=True)
    codebook = torch.randn(8, 8)
    selected = torch.tensor([[True, False], [False, True]])
    valid = torch.ones(2, 2, dtype=torch.bool)

    logits = head(
        representations,
        codebook,
        selected,
        valid,
        generator=torch.Generator().manual_seed(13),
    )

    assert not hasattr(head, "embedding")
    assert isinstance(head.mask_token, nn.Parameter)
    assert logits.shape == (2, 2, 8)
    logits.sum().backward()
    assert representations.grad is not None
    assert torch.isfinite(representations.grad).all()


def test_mam_forward_delegates_input_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = MAMHead(_tiny_config())
    representations = torch.randn(2, 2, 8)
    codebook = torch.randn(8, 8)
    selected = torch.tensor([[True, False], [False, True]])
    valid = torch.ones(2, 2, dtype=torch.bool)
    captured: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []

    def _capture_validation(
        q0_representations: Tensor,
        q0_codebook: Tensor,
        masked_positions: Tensor,
        valid_mask: Tensor,
    ) -> None:
        captured.append((q0_representations, q0_codebook, masked_positions, valid_mask))

    monkeypatch.setattr(head, "_validate_input", _capture_validation)

    head(
        representations,
        codebook,
        selected,
        valid,
        generator=torch.Generator().manual_seed(17),
    )

    assert captured == [(representations, codebook, selected, valid)]


def test_mam_corruption_is_deterministic_80_10_10_on_q0_representations() -> None:
    head = MAMHead(_tiny_config(latent_dim=2))
    representations = torch.arange(512, dtype=torch.float32).reshape(16, 16, 2)
    representations.requires_grad_()
    selected = torch.ones(16, 16, dtype=torch.bool)
    codebook = torch.arange(16, dtype=torch.float32).reshape(8, 2) * 100.0
    mask_token = torch.tensor([-1.0, -2.0])
    with torch.no_grad():
        head.mask_token.copy_(mask_token)

    expected_generator = torch.Generator().manual_seed(19)
    draws = torch.rand(selected.shape, generator=expected_generator)
    random_codes = torch.randint(
        0,
        8,
        selected.shape,
        generator=expected_generator,
    )
    expected = representations.detach().clone()
    expected[draws < 0.8] = mask_token
    random_positions = (draws >= 0.8) & (draws < 0.9)
    expected[random_positions] = codebook[random_codes[random_positions]]

    actual = head._corrupt_mam_representations(
        representations,
        selected,
        q0_codebook=codebook,
        generator=torch.Generator().manual_seed(19),
    )

    assert not hasattr(codec_module, "_corrupt_mam_representations")
    torch.testing.assert_close(actual, expected)
    assert bool((draws < 0.8).any())
    assert bool(random_positions.any())
    assert bool((draws >= 0.9).any())

    actual.sum().backward()
    expected_gradient = (draws >= 0.9).unsqueeze(-1).expand_as(representations)
    torch.testing.assert_close(
        representations.grad,
        expected_gradient.to(representations.dtype),
    )


def test_mam_sampling_never_selects_padding_and_keeps_one_valid_target() -> None:
    sample_mask = _required_codec_helper("_sample_mam_mask")
    valid = torch.tensor(
        [
            [True, True, False, False],
            [True, False, False, False],
        ]
    )
    torch.manual_seed(43)

    selected = sample_mask(valid, probability=0.01)

    assert not bool((selected & ~valid).any())
    assert selected.any(dim=1).all()


def test_next_frame_head_uses_two_four_head_layers_and_normalized_projection() -> None:
    head = NextFramePredictionHead(latent_dim=8, vl_feature_dim=6)

    context = getattr(head, "context", None)
    assert isinstance(context, nn.TransformerEncoder)
    layers = cast(list[nn.TransformerEncoderLayer], list(context.layers))
    assert len(layers) == 2
    assert all(layer.self_attn.num_heads == 4 for layer in layers)
    assert isinstance(getattr(head, "output_norm", None), nn.LayerNorm)
    assert isinstance(head.projection, nn.Linear)


def test_next_frame_head_forward_delegates_input_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = NextFramePredictionHead(latent_dim=8, vl_feature_dim=6)
    quantized = torch.randn(2, 4, 8)
    valid_mask = torch.ones(2, 4, dtype=torch.bool)
    captured: list[tuple[Tensor, Tensor]] = []

    def _capture_validation(
        received_quantized: Tensor,
        received_valid_mask: Tensor,
    ) -> None:
        captured.append((received_quantized, received_valid_mask))

    monkeypatch.setattr(head, "_validate_input", _capture_validation)

    head(quantized, valid_mask)

    assert captured == [(quantized, valid_mask)]


def test_next_frame_head_reads_the_last_valid_transformer_output() -> None:
    torch.manual_seed(23)
    head = NextFramePredictionHead(latent_dim=8, vl_feature_dim=6)
    head.eval()
    context = getattr(head, "context", None)
    output_norm = getattr(head, "output_norm", None)
    assert isinstance(context, nn.TransformerEncoder)
    assert isinstance(output_norm, nn.LayerNorm)

    context_outputs: list[Tensor] = []
    readout_inputs: list[Tensor] = []
    context_hook = context.register_forward_hook(
        lambda _module, _args, output: context_outputs.append(output.detach())
    )
    norm_hook = output_norm.register_forward_pre_hook(
        lambda _module, args: readout_inputs.append(args[0].detach())
    )
    latents = torch.randn(2, 4, 8)
    valid_mask = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, True],
        ]
    )

    head(latents, valid_mask)
    context_hook.remove()
    norm_hook.remove()

    assert len(context_outputs) == 1
    assert len(readout_inputs) == 1
    last_valid = valid_mask.sum(dim=1) - 1
    expected = context_outputs[0][torch.arange(latents.shape[0]), last_valid]
    torch.testing.assert_close(readout_inputs[0], expected)


def test_next_frame_head_is_sensitive_to_latent_order() -> None:
    torch.manual_seed(29)
    head = NextFramePredictionHead(latent_dim=8, vl_feature_dim=6)
    head.eval()
    latents = torch.zeros(1, 4, 8)
    latents[0, 0, 0] = 1.0
    latents[0, -1, 0] = -1.0
    valid_mask = torch.ones(1, 4, dtype=torch.bool)

    forward_order = head(latents, valid_mask)
    reverse_order = head(latents.flip(1), valid_mask)

    assert not torch.allclose(forward_order, reverse_order)


@pytest.mark.parametrize(
    ("action_values", "vl_values", "expected_action", "expected_vl"),
    [
        ([1.0, 3.0, 5.0, 7.0], [10.0, 20.0], [2.0, 6.0], [10.0, 20.0]),
        ([10.0, 20.0], [1.0, 3.0, 5.0, 7.0], [10.0, 20.0], [2.0, 6.0]),
    ],
)
def test_temporal_alignment_adaptively_pools_only_the_longer_stream(
    action_values: list[float],
    vl_values: list[float],
    expected_action: list[float],
    expected_vl: list[float],
) -> None:
    align = _required_codec_helper("_align_temporal_features")
    action_features = torch.tensor(action_values).reshape(1, -1, 1)
    vl_features = torch.tensor(vl_values).reshape(1, -1, 1)
    action_valid = torch.ones(action_features.shape[:2], dtype=torch.bool)
    vl_valid = torch.ones(vl_features.shape[:2], dtype=torch.bool)

    aligned_action, aligned_vl, aligned_valid = align(
        action_features=action_features,
        action_valid_mask=action_valid,
        vl_features=vl_features,
        vl_valid_mask=vl_valid,
    )

    torch.testing.assert_close(
        aligned_action,
        torch.tensor(expected_action).reshape(1, 2, 1),
    )
    torch.testing.assert_close(
        aligned_vl,
        torch.tensor(expected_vl).reshape(1, 2, 1),
    )
    assert torch.equal(aligned_valid, torch.ones(1, 2, dtype=torch.bool))


def test_alignment_uses_shared_vl_projection_and_three_learned_view_logits() -> None:
    codec = XTokenizerCodec(config=_tiny_config(latent_dim=8, vl_feature_dim=6))

    assert codec.alignment_projection.in_features == 6
    assert codec.alignment_projection.out_features == 8
    assert isinstance(codec.vl_view_logits, nn.Parameter)
    assert codec.vl_view_logits.shape == (3,)


def test_vl_fusion_masks_missing_views_per_sample_before_softmax() -> None:
    fuse = _required_codec_helper("_fuse_vl_views")
    projected = torch.tensor(
        [
            [[[[1.0], [float("nan")], [3.0]]]],
            [[[[float("nan")], [10.0], [float("nan")]]]],
        ]
    ).reshape(2, 1, 3, 1)
    is_pad = torch.tensor(
        [
            [[False, True, False]],
            [[True, False, True]],
        ]
    )

    fused, valid = fuse(projected, is_pad, torch.zeros(3))

    torch.testing.assert_close(fused, torch.tensor([[[2.0]], [[10.0]]]))
    assert torch.equal(valid, torch.ones(2, 1, dtype=torch.bool))


def test_prediction_loss_excludes_padded_next_vl_targets() -> None:
    torch.manual_seed(31)
    codec = XTokenizerCodec(config=_tiny_config(prediction_weight=0.2))
    codec.eval()
    batch = {
        "action": torch.zeros(2, 4, 10),
        "action_is_pad": torch.zeros(2, 4, dtype=torch.bool),
        "observation.state": torch.zeros(2, 10),
        "robot_type": torch.zeros(2, dtype=torch.long),
        "next_vl_features": torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [float("nan"), float("nan"), float("nan"), float("nan")],
            ]
        ),
        "next_vl_feature_is_pad": torch.tensor([False, True]),
    }

    nan_loss, nan_metrics = codec(batch)
    finite_batch = dict(batch)
    finite_batch["next_vl_features"] = batch["next_vl_features"].nan_to_num(nan=1.0e6)
    large_loss, large_metrics = codec(finite_batch)

    assert torch.isfinite(nan_loss)
    assert torch.isfinite(large_loss)
    torch.testing.assert_close(nan_loss, large_loss)
    assert nan_metrics["prediction_loss"] == pytest.approx(
        large_metrics["prediction_loss"]
    )


@pytest.mark.parametrize(
    ("state_probability", "embodiment_probability", "none_probability"),
    [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
)
def test_cfg_corruption_distinguishes_zero_state_dropped_embedding_and_none_slot(
    state_probability: float,
    embodiment_probability: float,
    none_probability: float,
) -> None:
    corrupt = _required_codec_helper("_corrupt_conditioning")
    anchor = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
    robot_type = torch.tensor([1, 2])

    actual_anchor, actual_robot_type, embodiment_valid = corrupt(
        anchor,
        robot_type,
        state_dropout_probability=state_probability,
        embodiment_dropout_probability=embodiment_probability,
        embodiment_none_probability=none_probability,
        embodiment_none_id=7,
        generator=torch.Generator().manual_seed(37),
    )

    if state_probability == 1.0:
        assert torch.equal(actual_anchor, torch.zeros_like(anchor))
        assert torch.equal(actual_robot_type, robot_type)
        assert embodiment_valid.all()
    elif embodiment_probability == 1.0:
        assert torch.equal(actual_anchor, anchor)
        assert torch.equal(actual_robot_type, robot_type)
        assert not embodiment_valid.any()
    else:
        assert torch.equal(actual_anchor, anchor)
        assert torch.equal(actual_robot_type, torch.full_like(robot_type, 7))
        assert embodiment_valid.all()
