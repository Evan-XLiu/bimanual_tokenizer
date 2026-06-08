# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from praxis.nn.architectures.transformer import Transformer, TransformerDecoder


def test_transformer_zero_init_query_proj_zeros_fused_self_attention_query() -> None:
    model = Transformer(
        dim=8,
        depth=1,
        head_dim=4,
        qkv_bias=True,
        zero_init_query_proj=True,
    )
    qkv = model.blocks[0].attn.qkv

    assert torch.count_nonzero(qkv.weight[:8]) == 0
    assert torch.count_nonzero(qkv.bias[:8]) == 0
    assert torch.count_nonzero(qkv.weight[8:]) > 0


def test_transformer_decoder_zero_init_query_proj_zeros_self_and_cross_queries() -> (
    None
):
    model = TransformerDecoder(
        dim=8,
        depth=1,
        head_dim=4,
        qkv_bias=True,
        zero_init_query_proj=True,
    )
    block = model.blocks[0]
    assert block.self_attn is not None

    assert torch.count_nonzero(block.self_attn.qkv.weight[:8]) == 0
    assert torch.count_nonzero(block.self_attn.qkv.bias[:8]) == 0
    assert torch.count_nonzero(block.self_attn.qkv.weight[8:]) > 0
    assert torch.count_nonzero(block.cross_attn.q_proj.weight) == 0
    assert torch.count_nonzero(block.cross_attn.q_proj.bias) == 0
    assert torch.count_nonzero(block.cross_attn.kv_proj.weight) > 0


def test_transformer_forward_preserves_shape() -> None:
    model = Transformer(dim=8, depth=2, head_dim=4, drop=0.0)
    x = torch.randn(2, 5, 8)

    assert model(x).shape == x.shape


def test_transformer_adaln_forward_is_identity_at_initialization() -> None:
    model = Transformer(dim=8, depth=1, head_dim=4, use_adaLN=True)
    x = torch.randn(2, 3, 8)
    emb = torch.randn(2, 3, 8)

    out = model(x, adaLN_emb=emb, adaLN_packing_fn=lambda value: value)

    assert torch.allclose(out, x)


def test_transformer_decoder_forward_preserves_shape_without_self_attention() -> None:
    model = TransformerDecoder(dim=8, depth=2, head_dim=4, enable_self_attn=False)
    x = torch.randn(2, 5, 8)
    context = torch.randn(2, 7, 8)

    out = model(x, context)

    assert out.shape == x.shape
    assert all(block.self_attn is None for block in model.blocks)


def test_invalid_transformer_head_dim_raises() -> None:
    with pytest.raises(ValueError, match="head_dim"):
        Transformer(dim=10, depth=1, head_dim=4)


def test_invalid_transformer_init_style_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported weight init"):
        Transformer(dim=8, depth=1, head_dim=4, weight_init_style="invalid")
