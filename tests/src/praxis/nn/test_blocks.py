# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from praxis.nn.blocks.attention import CrossAttention, SelfAttention
from praxis.nn.blocks.mlp import GatedMlp, Mlp
from praxis.nn.blocks.transformer_block import (
    Block,
    BlockAdaLN,
    DecoderBlock,
    DecoderBlockAdaLN,
    expand_to_padded_seq,
)


def _identity_pack(x: torch.Tensor) -> torch.Tensor:
    return x


def test_self_attention_preserves_shape_and_public_projection_name() -> None:
    attn = SelfAttention(dim=12, num_heads=3, qkv_bias=True, qk_norm=True)
    x = torch.randn(2, 5, 12)

    out = attn(x)

    assert out.shape == x.shape
    assert attn.qkv.out_features == 36


def test_cross_attention_uses_separate_query_and_key_value_projections() -> None:
    attn = CrossAttention(dim=12, num_heads=3, qkv_bias=True, qk_norm=False)
    x = torch.randn(2, 4, 12)
    context = torch.randn(2, 7, 12)

    out = attn(x, context)

    assert out.shape == x.shape
    assert attn.q_proj.out_features == 12
    assert attn.kv_proj.out_features == 24


@pytest.mark.parametrize("mlp_cls", [Mlp, GatedMlp])
def test_mlp_variants_preserve_batch_sequence_shape(
    mlp_cls: type[torch.nn.Module],
) -> None:
    mlp = mlp_cls(in_features=8, hidden_features=16, out_features=6, drop=0.0)
    x = torch.randn(2, 3, 8)

    assert mlp(x).shape == (2, 3, 6)


def test_block_forward_preserves_shape_with_head_dim() -> None:
    block = Block(dim=16, head_dim=4, mlp_ratio=2.0, gated_mlp=True)
    x = torch.randn(2, 5, 16)

    assert block(x).shape == x.shape


def test_block_adaln_zero_init_is_identity_and_requires_conditioning() -> None:
    block = BlockAdaLN(dim=8, head_dim=4, adaLN_expansion=1)
    x = torch.randn(2, 3, 8)
    emb = torch.randn(2, 3, 8)

    with pytest.raises(ValueError, match="adaLN_emb"):
        block(x)

    out = block(x, adaLN_emb=emb, adaLN_packing_fn=_identity_pack)
    assert torch.allclose(out, x)


def test_decoder_block_cross_attention_shape_and_self_attn_toggle() -> None:
    block = DecoderBlock(dim=16, head_dim=4, enable_self_attn=False)
    x = torch.randn(2, 5, 16)
    context = torch.randn(2, 7, 16)

    assert block.self_attn is None
    assert block(x, context).shape == x.shape


def test_decoder_block_adaln_zero_init_is_identity_and_requires_conditioning() -> None:
    block = DecoderBlockAdaLN(dim=8, head_dim=4, enable_self_attn=True)
    x = torch.randn(2, 3, 8)
    context = torch.randn(2, 3, 8)
    emb = torch.randn(2, 3, 8)

    with pytest.raises(ValueError, match="adaLN_emb"):
        block(x, context)

    out = block(x, context, adaLN_emb=emb, adaLN_packing_fn=_identity_pack)
    assert torch.allclose(out, x)


def test_expand_to_padded_seq_extends_to_target_sequence_length() -> None:
    emb = torch.ones(2, 3, 4)
    padded = torch.zeros(2, 5, 4)

    out = expand_to_padded_seq(emb, padded)

    assert out.shape == padded.shape
    assert torch.equal(out[:, :3], emb)
    assert torch.count_nonzero(out[:, 3:]) == 0
