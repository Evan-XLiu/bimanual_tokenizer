# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from praxis.codecs.tokens.btt_independent import (
    BTTIndependentConfig,
    BTTIndependentEncoder,
)


def _small_encoder() -> BTTIndependentEncoder:
    return BTTIndependentEncoder(
        BTTIndependentConfig(model_dim=32, num_heads=4, dropout=0.0)
    )


def _parameter_ids(module: torch.nn.Module) -> set[int]:
    return {id(parameter) for parameter in module.parameters()}


def test_encoder_shapes_and_original_right_arm_convention() -> None:
    encoder = _small_encoder().eval()
    actions = torch.randn(2, 32, 14)

    left, right = encoder(actions)

    assert left.shape == (2, 8, 32)
    assert right.shape == (2, 8, 32)
    assert encoder.encode_concatenated(actions).shape == (2, 16, 32)
    assert encoder.latent_horizon == 16
    expected_sign = torch.tensor([-1, 1, 1, 1, -1, -1, 1])
    physical_right = actions[..., 7:]
    torch.testing.assert_close(
        encoder.canonicalize_right_arm(physical_right),
        physical_right * expected_sign,
    )


def test_each_arm_uses_separate_normalization_bounds() -> None:
    encoder = BTTIndependentEncoder(
        BTTIndependentConfig(
            left_action_min=(0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            left_action_max=(2.0, 14.0, 2.0, 2.0, 2.0, 2.0, 2.0),
            right_action_min=(-4.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            right_action_max=(0.0, 28.0, 2.0, 2.0, 2.0, 2.0, 2.0),
            model_dim=32,
            num_heads=4,
            dropout=0.0,
        )
    )
    left = torch.tensor([[[1.0, 12.0, 1.0, 1.0, 1.0, 1.0, 1.0]]]).expand(1, 32, 7)
    canonical_right = torch.tensor([[[-2.0, 24.0, 1.0, 1.0, 1.0, 1.0, 1.0]]]).expand(
        1, 32, 7
    )
    physical_right = canonical_right * encoder.right_arm_sign

    norm_left, norm_right = encoder.split_canonicalize_and_normalize(
        torch.cat((left, physical_right), dim=-1)
    )

    torch.testing.assert_close(norm_left, torch.zeros_like(norm_left))
    torch.testing.assert_close(norm_right, torch.zeros_like(norm_right))
    restored = encoder.unnormalize_and_decanonicalize(norm_left, norm_right)
    torch.testing.assert_close(restored, torch.cat((left, physical_right), dim=-1))
    state = encoder.state_dict()
    assert "left_action_min" in state
    assert "left_action_max" in state
    assert "right_action_min" in state
    assert "right_action_max" in state
    assert "shared_action_min" not in state


def test_encoder_self_attention_and_cross_directions_share_no_parameters() -> None:
    encoder = _small_encoder()

    assert not (
        _parameter_ids(encoder.left_arm_encoder)
        & _parameter_ids(encoder.right_arm_encoder)
    )
    assert not (
        _parameter_ids(encoder.left_self_attention)
        & _parameter_ids(encoder.right_self_attention)
    )
    assert not (
        _parameter_ids(encoder.cross_attention.left_from_right)
        & _parameter_ids(encoder.cross_attention.right_from_left)
    )


def test_swapping_canonical_arms_is_not_structurally_forced_equivariant() -> None:
    encoder = _small_encoder().eval()
    actions = torch.randn(2, 32, 14)
    left, physical_right = actions.split(7, dim=-1)
    sign = encoder.right_arm_sign.view(1, 1, -1)

    original_left, original_right = encoder(actions)
    swapped_physical = torch.cat((physical_right * sign, left * sign), dim=-1)
    swapped_left, swapped_right = encoder(swapped_physical)

    assert not torch.allclose(swapped_left, original_right)
    assert not torch.allclose(swapped_right, original_left)


def test_future_actions_cannot_change_earlier_coordinated_latents() -> None:
    encoder = _small_encoder().eval()
    actions = torch.randn(2, 32, 14)
    changed = actions.clone()
    changed[:, 12:] = torch.randn_like(changed[:, 12:]) * 10.0

    left, right = encoder(actions)
    changed_left, changed_right = encoder(changed)

    torch.testing.assert_close(left[:, :3], changed_left[:, :3])
    torch.testing.assert_close(right[:, :3], changed_right[:, :3])
    assert not torch.allclose(left[:, 3:], changed_left[:, 3:])
    assert not torch.allclose(right[:, 3:], changed_right[:, 3:])
