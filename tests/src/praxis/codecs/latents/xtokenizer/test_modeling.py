# SPDX-FileCopyrightText: 2026 Xiaoshen Han
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the X-Tokenizer action encoder and decoder."""

from __future__ import annotations

import pytest
import torch

from praxis.codecs.latents.xtokenizer.config import XTokenizerModelConfig
from praxis.codecs.latents.xtokenizer.modeling import (
    ActionDecoder,
    DeltaActionEncoder,
)


def _config() -> XTokenizerModelConfig:
    return XTokenizerModelConfig(
        action_dim=6,
        action_horizon=32,
        compression_ratio=4,
        hidden_dim=32,
        latent_dim=16,
        encoder_num_heads=4,
        encoder_depth=2,
        decoder_num_heads=4,
        decoder_depth=2,
        mam_num_heads=4,
        mam_depth=2,
        prediction_num_heads=4,
        prediction_depth=2,
        dropout=0.0,
        embodiment_vocab_size=8,
    )


def test_model_uses_paper_stem_rope_and_independent_depths() -> None:
    config = _config()
    encoder = DeltaActionEncoder(config)
    decoder = ActionDecoder(config)

    assert [type(module) for module in encoder.action_projection] == [
        torch.nn.Linear,
        torch.nn.LayerNorm,
        torch.nn.GELU,
        torch.nn.Dropout,
    ]
    assert len(encoder.temporal_encoder) == config.encoder_depth
    assert len(decoder.temporal_decoder) == config.decoder_depth
    assert encoder.temporal_encoder[0].attention.num_heads == config.encoder_num_heads
    assert decoder.temporal_decoder[0].attention.num_heads == config.decoder_num_heads
    assert encoder.action_rope.base == pytest.approx(10_000.0)
    assert encoder.latent_rope.base == pytest.approx(10_000.0)
    assert decoder.latent_rope.base == pytest.approx(10_000.0)
    assert decoder.action_rope.base == pytest.approx(10_000.0)


def test_encoder_compresses_32_timesteps_to_8_latent_slots() -> None:
    encoder = DeltaActionEncoder(_config())
    actions = torch.randn(2, 32, 6)
    anchor = torch.randn(2, 1, 6)
    robot_type = torch.tensor([1, 3])

    output = encoder(actions, anchor, robot_type, action_mask=None)

    assert output.latents.shape == (2, 8, 16)
    assert output.latent_mask.shape == (2, 8)
    assert output.latent_mask.dtype is torch.bool
    assert torch.equal(output.latent_mask, torch.ones(2, 8, dtype=torch.bool))


def test_encoder_propagates_padding_and_ignores_padded_action_values() -> None:
    torch.manual_seed(0)
    encoder = DeltaActionEncoder(_config()).eval()
    actions = torch.randn(1, 32, 6)
    changed_padding = actions.clone()
    changed_padding[:, 18:] = torch.randn_like(changed_padding[:, 18:]) * 1_000
    anchor = torch.randn(1, 1, 6)
    robot_type = torch.tensor([2])
    action_mask = torch.arange(32).unsqueeze(0) < 18

    original = encoder(actions, anchor, robot_type, action_mask)
    changed = encoder(changed_padding, anchor, robot_type, action_mask)

    expected_mask = torch.tensor([[True, True, True, True, True, False, False, False]])
    assert torch.equal(original.latent_mask, expected_mask)
    assert torch.equal(changed.latent_mask, expected_mask)
    assert torch.allclose(
        original.latents[original.latent_mask],
        changed.latents[changed.latent_mask],
    )
    assert torch.count_nonzero(original.latents[~original.latent_mask]) == 0


def test_encoder_uses_embodiment_embeddings() -> None:
    torch.manual_seed(1)
    encoder = DeltaActionEncoder(_config()).eval()
    actions = torch.randn(1, 32, 6).expand(2, -1, -1).clone()
    anchor = torch.randn(1, 1, 6).expand(2, -1, -1).clone()

    output = encoder(actions, anchor, torch.tensor([1, 2]), action_mask=None)

    assert not torch.allclose(output.latents[0], output.latents[1])


def test_encoder_can_remove_embodiment_embedding_for_cfg_dropout() -> None:
    torch.manual_seed(101)
    encoder = DeltaActionEncoder(_config()).eval()
    actions = torch.randn(1, 32, 6).expand(2, -1, -1).clone()
    anchor = torch.randn(1, 1, 6).expand(2, -1, -1).clone()

    output = encoder(
        actions,
        anchor,
        torch.tensor([1, 2]),
        action_mask=None,
        embodiment_valid=torch.zeros(2, dtype=torch.bool),
    )

    torch.testing.assert_close(output.latents[0], output.latents[1])


def test_encoder_uses_state_anchor_conditioning() -> None:
    torch.manual_seed(2)
    encoder = DeltaActionEncoder(_config()).eval()
    actions = torch.randn(1, 32, 6).expand(2, -1, -1).clone()
    anchors = torch.stack(
        [torch.zeros(1, 6), torch.ones(1, 6)],
        dim=0,
    )
    robot_type = torch.tensor([1, 1])

    output = encoder(actions, anchors, robot_type, action_mask=None)

    assert not torch.allclose(output.latents[0], output.latents[1])


def test_decoder_reconstructs_timestep_shape_and_masks_padding() -> None:
    torch.manual_seed(3)
    decoder = ActionDecoder(_config()).eval()
    latents = torch.randn(1, 8, 16)
    changed_padding = latents.clone()
    changed_padding[:, 5:] = torch.randn_like(changed_padding[:, 5:]) * 1_000
    anchor = torch.randn(1, 1, 6)
    robot_type = torch.tensor([4])
    action_mask = torch.arange(32).unsqueeze(0) < 18

    reconstructed = decoder(latents, anchor, robot_type, action_mask)
    changed = decoder(changed_padding, anchor, robot_type, action_mask)

    assert reconstructed.shape == (1, 32, 6)
    assert torch.allclose(
        reconstructed[action_mask],
        changed[action_mask],
    )
    assert torch.count_nonzero(reconstructed[~action_mask]) == 0


def test_decoder_uses_state_anchor_conditioning() -> None:
    torch.manual_seed(4)
    decoder = ActionDecoder(_config()).eval()
    latents = torch.randn(1, 8, 16).expand(2, -1, -1).clone()
    anchors = torch.stack(
        [torch.zeros(1, 6), torch.ones(1, 6)],
        dim=0,
    )
    robot_type = torch.tensor([1, 1])

    reconstructed = decoder(latents, anchors, robot_type, action_mask=None)

    assert not torch.allclose(reconstructed[0], reconstructed[1])


def test_gradients_reach_encoder_queries_and_decoder_output() -> None:
    torch.manual_seed(5)
    config = _config()
    encoder = DeltaActionEncoder(config)
    decoder = ActionDecoder(config)
    actions = torch.randn(2, 32, 6)
    anchor = torch.randn(2, 1, 6)
    robot_type = torch.tensor([0, 7])
    action_mask = torch.ones(2, 32, dtype=torch.bool)

    encoded = encoder(actions, anchor, robot_type, action_mask)
    reconstructed = decoder(
        encoded.latents,
        anchor,
        robot_type,
        action_mask,
    )
    reconstructed.square().mean().backward()

    assert encoder.action_projection.weight.grad is not None
    assert torch.count_nonzero(encoder.action_projection.weight.grad) > 0
    assert encoder.latent_queries.grad is not None
    assert torch.count_nonzero(encoder.latent_queries.grad) > 0
    assert decoder.output_projection.weight.grad is not None
    assert torch.count_nonzero(decoder.output_projection.weight.grad) > 0


@pytest.mark.parametrize("module_type", [DeltaActionEncoder, ActionDecoder])
def test_model_rejects_non_divisible_horizon(
    module_type: type[torch.nn.Module],
) -> None:
    config = _config()
    config.action_horizon = 30

    with pytest.raises(ValueError, match="divisible"):
        module_type(config)
