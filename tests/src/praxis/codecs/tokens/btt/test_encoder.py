# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from praxis.codecs.tokens.btt import BTTEncoder, BTTEncoderConfig
from praxis.nn.embeddings.positional import SinusoidPosEmbed


def _small_encoder() -> BTTEncoder:
    return BTTEncoder(BTTEncoderConfig(model_dim=32, num_heads=4, dropout=0.0))


def test_encoder_shapes_and_continuous_token_budget() -> None:
    encoder = _small_encoder()
    actions = torch.randn(2, 32, 14)

    left, right = encoder(actions)

    assert left.shape == (2, 8, 32)
    assert right.shape == (2, 8, 32)
    assert encoder.encode_concatenated(actions).shape == (2, 16, 32)
    assert encoder.latent_horizon == 16


def test_encoder_uses_the_same_fixed_sinusoid_positions_as_quest() -> None:
    encoder = _small_encoder()
    quest_position = SinusoidPosEmbed(32, max_shape=[32])

    assert isinstance(encoder.temporal_position, SinusoidPosEmbed)
    assert not any(
        name.startswith("temporal_position") for name, _ in encoder.named_parameters()
    )
    torch.testing.assert_close(
        encoder.temporal_position([8]),
        quest_position([8]),
    )
    assert "temporal_position.pos_embed" in encoder.state_dict()


def test_right_arm_convention_is_a_persistent_self_inverse_buffer() -> None:
    encoder = _small_encoder()
    right = torch.randn(2, 32, 7)

    canonical = encoder.canonicalize_right_arm(right)

    expected_sign = torch.tensor([-1, 1, 1, 1, -1, -1, 1])
    torch.testing.assert_close(canonical, right * expected_sign)
    torch.testing.assert_close(encoder.decanonicalize_right_arm(canonical), right)
    assert "right_arm_sign" in dict(encoder.named_buffers())
    assert "right_arm_sign" in encoder.state_dict()


def test_action_normalization_canonicalizes_then_shares_pooled_bounds() -> None:
    encoder = BTTEncoder(
        BTTEncoderConfig(
            action_dim=4,
            arm_dim=2,
            right_arm_sign=(-1.0, 1.0),
            shared_action_min=(0.0, 10.0),
            shared_action_max=(5.0, 24.0),
            model_dim=32,
            num_heads=4,
            dropout=0.0,
        )
    )
    torch.testing.assert_close(encoder.shared_action_min, torch.tensor([0.0, 10.0]))
    torch.testing.assert_close(encoder.shared_action_max, torch.tensor([5.0, 24.0]))

    left = torch.tensor([[[1.0, 12.0]]]).expand(1, 32, 2)
    # Physical right joint 0 is negated before normalization.
    physical_right = torch.tensor([[[-1.0, 12.0]]]).expand(1, 32, 2)
    norm_left, norm_right = encoder.split_canonicalize_and_normalize(
        torch.cat((left, physical_right), dim=-1)
    )

    torch.testing.assert_close(norm_left, norm_right)
    restored = encoder.unnormalize_and_decanonicalize(norm_left, norm_right)
    torch.testing.assert_close(restored, torch.cat((left, physical_right), dim=-1))
    assert "shared_action_min" in encoder.state_dict()
    assert "shared_action_max" in encoder.state_dict()


def test_default_extrema_map_to_endpoints_without_quantile_clamping() -> None:
    encoder = _small_encoder()
    torch.testing.assert_close(
        encoder.shared_action_min,
        torch.tensor(
            [
                -1.3947148,
                -0.5742835,
                -0.1190902,
                -1.9893787,
                -1.4216828,
                -3.4687879,
                0.0,
            ]
        ),
    )
    torch.testing.assert_close(
        encoder.shared_action_max,
        torch.tensor(
            [
                2.9455960,
                3.3889313,
                3.4514999,
                1.9530925,
                1.6339467,
                3.0108311,
                1.0,
            ]
        ),
    )
    lower = encoder.shared_action_min.view(1, 1, -1).expand(1, 32, -1)
    upper = encoder.shared_action_max.view(1, 1, -1).expand(1, 32, -1)

    torch.testing.assert_close(
        encoder.normalize_arm_actions(lower), -torch.ones_like(lower)
    )
    torch.testing.assert_close(
        encoder.normalize_arm_actions(upper), torch.ones_like(upper)
    )
    assert torch.all(encoder.normalize_arm_actions(lower - 1.0) < -1.0)
    assert torch.all(encoder.normalize_arm_actions(upper + 1.0) > 1.0)


def test_shared_paths_are_equivariant_to_swapping_canonical_arms() -> None:
    encoder = _small_encoder().eval()
    actions = torch.randn(2, 32, 14)
    left, physical_right = actions.split(7, dim=-1)
    sign = encoder.right_arm_sign.view(1, 1, -1)

    original_left, original_right = encoder(actions)
    swapped_physical = torch.cat((physical_right * sign, left * sign), dim=-1)
    swapped_left, swapped_right = encoder(swapped_physical)

    torch.testing.assert_close(swapped_left, original_right)
    torch.testing.assert_close(swapped_right, original_left)


def test_future_actions_cannot_change_earlier_coordinated_latents() -> None:
    encoder = _small_encoder().eval()
    actions = torch.randn(2, 32, 14)
    changed = actions.clone()
    # Latent positions 0, 1, 2 align with physical endpoints 3, 7, 11.
    changed[:, 12:] = torch.randn_like(changed[:, 12:]) * 10.0

    left, right = encoder(actions)
    changed_left, changed_right = encoder(changed)

    torch.testing.assert_close(left[:, :3], changed_left[:, :3])
    torch.testing.assert_close(right[:, :3], changed_right[:, :3])
    assert not torch.allclose(left[:, 3:], changed_left[:, 3:])
    assert not torch.allclose(right[:, 3:], changed_right[:, 3:])


def test_invalid_input_shape_is_rejected() -> None:
    encoder = _small_encoder()

    with pytest.raises(ValueError, match="bimanual actions"):
        encoder(torch.randn(2, 31, 14))
