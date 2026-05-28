from __future__ import annotations

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
