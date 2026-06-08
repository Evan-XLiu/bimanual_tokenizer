# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import pytest
import torch

from praxis.nn.embeddings.positional import (
    LearnedPosEmbed,
    RotaryPosEmbed,
    SinusoidPosEmbed,
    build_1d_sinusoid_pos_embed,
    build_2d_sinusoid_pos_embed,
)
from praxis.nn.embeddings.time import (
    SinusoidalScalarEmbedding,
    geometric_frequency_grid,
    sinusoidal_scalar_features,
)


class TestSinusoidalScalarFeatures:
    def test_frequency_grid_endpoint_and_direction(self) -> None:
        ascending = geometric_frequency_grid(3, min_frequency=1.0, max_frequency=100.0)
        descending = geometric_frequency_grid(
            3, min_frequency=1.0, max_frequency=100.0, descending=True
        )

        assert torch.allclose(ascending, torch.tensor([1.0, 10.0, 100.0]))
        assert torch.allclose(descending, torch.tensor([100.0, 10.0, 1.0]))

    def test_features_shape_and_order(self) -> None:
        values = torch.tensor([0.0, 0.25])
        features = sinusoidal_scalar_features(
            values,
            dim=4,
            min_frequency=1.0,
            max_frequency=1.0,
            angular_scale=2.0 * math.pi,
            channel_order="sin_cos",
        )

        assert features.shape == (2, 4)
        assert torch.allclose(features[0, :2], torch.zeros(2))
        assert torch.allclose(features[0, 2:], torch.ones(2))
        assert torch.allclose(features[1, :2], torch.ones(2))
        assert torch.allclose(features[1, 2:], torch.zeros(2), atol=1e-6)

    def test_can_represent_legacy_timestep_formula(self) -> None:
        values = torch.tensor([0.0, 10.0])
        features = sinusoidal_scalar_features(
            values,
            dim=8,
            min_frequency=1.0 / 256.0,
            max_frequency=1.0,
            endpoint=False,
            descending=True,
            channel_order="cos_sin",
            pad_odd_dim=True,
        )
        frequencies = torch.tensor([1.0, 0.25, 0.0625, 0.015625])
        angles = values[:, None] * frequencies[None, :]
        expected = torch.cat([angles.cos(), angles.sin()], dim=-1)

        assert torch.allclose(features, expected)

    def test_can_represent_period_formula(self) -> None:
        values = torch.tensor([0.0, 0.5], dtype=torch.float64)
        min_period = 4e-3
        max_period = 4.0
        features = sinusoidal_scalar_features(
            values,
            dim=8,
            min_frequency=1.0 / max_period,
            max_frequency=1.0 / min_period,
            angular_scale=2.0 * math.pi,
            endpoint=True,
            descending=True,
            channel_order="sin_cos",
        )
        periods = min_period * (max_period / min_period) ** torch.linspace(
            0.0, 1.0, 4, dtype=torch.float64
        )
        angles = values[:, None] / periods[None, :] * (2.0 * math.pi)
        expected = torch.cat([angles.sin(), angles.cos()], dim=-1)

        assert torch.allclose(features, expected)

    def test_validation_rank_and_even_dim(self) -> None:
        with pytest.raises(ValueError, match="rank-1"):
            sinusoidal_scalar_features(
                torch.zeros(2, 1),
                dim=8,
                min_frequency=1.0,
                max_frequency=1.0,
            )
        with pytest.raises(ValueError, match="dim must be even"):
            sinusoidal_scalar_features(
                torch.zeros(2),
                dim=7,
                min_frequency=1.0,
                max_frequency=1.0,
            )

    def test_odd_dim_padding(self) -> None:
        out = sinusoidal_scalar_features(
            torch.zeros(2),
            dim=7,
            min_frequency=1.0,
            max_frequency=1.0,
            pad_odd_dim=True,
        )
        assert out.shape == (2, 7)
        assert torch.allclose(out[:, -1], torch.zeros(2))


class TestSinusoidalScalarEmbedding:
    def test_forward_shape(self) -> None:
        module = SinusoidalScalarEmbedding(
            dim=32,
            feature_dim=16,
            hidden_dim=64,
            min_frequency=1.0,
            max_frequency=128.0,
        )
        out = module(torch.tensor([0.0, 0.3, 0.7]))
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 32)

    def test_forward_tokens_output(self) -> None:
        module = SinusoidalScalarEmbedding(
            dim=16,
            min_frequency=1.0,
            max_frequency=128.0,
            as_tokens=True,
        )
        out = module(torch.tensor([0.0, 1.0]))
        assert isinstance(out, tuple)
        assert len(out) == 2
        assert out[0].shape == (1, 1, 16)
        assert out[1].shape == (1, 1, 16)

    def test_invalid_activation(self) -> None:
        with pytest.raises(ValueError, match="Unsupported activation"):
            SinusoidalScalarEmbedding(  # type: ignore[arg-type]
                dim=16,
                min_frequency=1.0,
                max_frequency=128.0,
                activation="relu",
            )


class TestSinusoidPosEmbed:
    def test_1d_sincos_interleaved_at_zero(self) -> None:
        emb = build_1d_sinusoid_pos_embed(length=4, dim=8)
        # Shape: (1, L, D); at position 0 => sin=0 on even channels, cos=1 on odd channels.
        at_zero = emb[0, 0, :]
        assert torch.allclose(at_zero[0::2], torch.zeros(4))
        assert torch.allclose(at_zero[1::2], torch.ones(4))

    def test_2d_builder_shape_and_dim_guard(self) -> None:
        emb = build_2d_sinusoid_pos_embed(height=3, width=5, dim=8)
        assert emb.shape == (1, 3, 5, 8)
        with pytest.raises(ValueError, match="divisible by 4"):
            build_2d_sinusoid_pos_embed(height=2, width=2, dim=6)

    def test_builder_dtype(self) -> None:
        emb = build_1d_sinusoid_pos_embed(length=4, dim=8, dtype=torch.float64)
        assert emb.dtype == torch.float64
        with pytest.raises(ValueError, match="floating point"):
            build_1d_sinusoid_pos_embed(length=4, dim=8, dtype=torch.int64)

    def test_absolute_and_interpolate_scaling(self) -> None:
        absolute = SinusoidPosEmbed(dim=8, max_shape=[8], scaling="absolute")
        interp = SinusoidPosEmbed(dim=8, max_shape=[8], scaling="interpolate")

        out_abs = absolute([4])
        out_interp = interp([10])
        assert out_abs.shape == (1, 4, 8)
        assert out_interp.shape == (1, 10, 8)
        with pytest.raises(ValueError, match="exceeds max_shape"):
            absolute([10])

    def test_sincos_posemb_is_buffer_not_parameter(self) -> None:
        module = SinusoidPosEmbed(dim=8, max_shape=[8])
        assert "pos_embed" in dict(module.named_buffers())
        assert "pos_embed" not in dict(module.named_parameters())

    def test_manual_add_dtype_and_shape(self) -> None:
        pos_embed = SinusoidPosEmbed(dim=8, max_shape=[6])
        x = torch.randn(2, 6, 8, dtype=torch.float32)
        y = x + pos_embed([6], device=x.device, dtype=x.dtype)
        assert y.shape == x.shape
        assert y.dtype == x.dtype


class TestLearnedPosEmbed:
    def test_learned_posemb_is_parameter(self) -> None:
        module = LearnedPosEmbed(dim=8, max_shape=[6])
        out = module([4])
        assert out.shape == (1, 4, 8)
        assert "pos_embed" in dict(module.named_parameters())
        assert "pos_embed" not in dict(module.named_buffers())

    def test_learned_posemb_supports_multidim_absolute_slicing(self) -> None:
        module = LearnedPosEmbed(dim=6, max_shape=[2, 3, 4])
        out = module([1, 2, 3])
        assert out.shape == (1, 1, 2, 3, 6)

    def test_learned_posemb_interpolates_1d_and_2d(self) -> None:
        one_d = LearnedPosEmbed(dim=8, max_shape=[4], scaling="interpolate")
        two_d = LearnedPosEmbed(dim=8, max_shape=[2, 3], scaling="interpolate")
        assert one_d([7]).shape == (1, 7, 8)
        assert two_d([4, 5]).shape == (1, 4, 5, 8)


class TestRotaryPosEmbed:
    def test_zero_positions_are_identity(self) -> None:
        rope = RotaryPosEmbed(dim=4)
        x = torch.randn(2, 3, 5, 4)
        y = rope(x, positions=torch.zeros(5))
        assert torch.allclose(y, x)

    def test_rotation_preserves_pair_norms(self) -> None:
        rope = RotaryPosEmbed(dim=8)
        x = torch.randn(2, 3, 5, 8)
        y = rope(x)
        assert y.shape == x.shape
        assert torch.allclose(
            y.float().norm(dim=-1),
            x.float().norm(dim=-1),
            atol=1e-5,
        )

    def test_partial_rotation_leaves_tail_unchanged(self) -> None:
        rope = RotaryPosEmbed(dim=8, rotary_dim=4)
        x = torch.randn(2, 3, 5, 8)
        y = rope(x)
        assert y.shape == x.shape
        assert torch.allclose(y[..., 4:], x[..., 4:])
        assert torch.allclose(
            y[..., :4].float().norm(dim=-1),
            x[..., :4].float().norm(dim=-1),
            atol=1e-5,
        )

    def test_split_pairing_matches_gemma_style_formula(self) -> None:
        rope = RotaryPosEmbed(dim=4, pairing="split")
        x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
        y = rope(x, positions=torch.ones(1))
        inv_freq = torch.tensor([1.0, 0.01])
        cos = inv_freq.cos()
        sin = inv_freq.sin()
        expected = torch.empty_like(x)
        expected[..., :2] = x[..., :2] * cos - x[..., 2:] * sin
        expected[..., 2:] = x[..., :2] * sin + x[..., 2:] * cos

        assert torch.allclose(y, expected)

    def test_per_batch_positions(self) -> None:
        rope = RotaryPosEmbed(dim=4)
        x = torch.randn(2, 3, 5, 4)
        positions = torch.stack([torch.zeros(5), torch.arange(5)])
        y = rope(x, positions=positions)
        assert y.shape == x.shape
        assert torch.allclose(y[0], x[0])

    def test_dim_and_position_validation(self) -> None:
        with pytest.raises(ValueError, match="even"):
            RotaryPosEmbed(dim=4, rotary_dim=3)
        rope = RotaryPosEmbed(dim=4)
        with pytest.raises(ValueError, match="positions length"):
            rope(torch.randn(2, 3, 4), positions=torch.arange(2))
        with pytest.raises(ValueError, match="batch size"):
            rope(torch.randn(2, 3, 4), positions=torch.zeros(3, 3))
