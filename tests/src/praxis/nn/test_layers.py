# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from praxis.nn.layers.drop_path import DropPath, drop_path
from praxis.nn.layers.norm import Fp32LayerNorm, Fp32RMSNorm


def test_drop_path_is_noop_when_disabled_or_eval() -> None:
    x = torch.randn(4, 3, 2)

    assert torch.equal(drop_path(x, drop_prob=0.5, training=False), x)
    assert torch.equal(drop_path(x, drop_prob=0.0, training=True), x)

    module = DropPath(0.5)
    module.eval()
    assert torch.equal(module(x), x)


def test_drop_path_drops_per_sample_and_preserves_expectation_scale() -> None:
    x = torch.ones(8, 3, 2)
    torch.manual_seed(4)

    out = drop_path(x, drop_prob=0.5, training=True)
    per_sample = out.flatten(1).mean(dim=1)

    assert set(per_sample.tolist()).issubset({0.0, 2.0})
    assert 0.0 in per_sample
    assert 2.0 in per_sample


def test_fp32_layer_norm_restores_input_dtype() -> None:
    norm = Fp32LayerNorm(4)
    x = torch.randn(2, 3, 4, dtype=torch.bfloat16)

    out = norm(x)

    assert out.dtype == x.dtype
    assert out.shape == x.shape


def test_fp32_rms_norm_matches_manual_float32_math() -> None:
    norm = Fp32RMSNorm(3, eps=0.0, elementwise_affine=True)
    norm.weight.data = torch.tensor([1.0, 2.0, 0.5])
    x = torch.tensor([[[3.0, 4.0, 0.0]]], dtype=torch.float16)

    out = norm(x)

    rms = torch.sqrt(torch.mean(x.float().pow(2), dim=-1, keepdim=True))
    expected = (x.float() / rms * norm.weight.float()).to(dtype=x.dtype)
    assert out.dtype == x.dtype
    assert torch.allclose(out, expected)


def test_fp32_rms_norm_without_affine_has_no_weight_parameter() -> None:
    norm = Fp32RMSNorm((2, 3), elementwise_affine=False)
    x = torch.randn(4, 2, 3)

    assert norm.weight is None
    assert norm(x).shape == x.shape
