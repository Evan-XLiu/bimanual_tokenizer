# --------------------------------------------------------------------
# Copyright (C) 2024-2025 EPFL & Apple Inc.
# Licensed under the EPFL-Apple Sample Code License (Non-Commercial)
#
# Modifications by Chaoqi Liu for research purposes.
# --------------------------------------------------------------------

from collections.abc import Callable
from functools import partial

import torch
import torch.nn as nn

from praxis.nn.blocks.attention import CrossAttention, SelfAttention
from praxis.nn.blocks.transformer_block import (
    Block,
    BlockAdaLN,
    DecoderBlock,
    DecoderBlockAdaLN,
)
from praxis.nn.layers.norm import Fp32RMSNorm

__all__ = ["Transformer", "TransformerDecoder"]

_DEFAULT_NORM_LAYER = partial(Fp32RMSNorm, elementwise_affine=False)


def _zero_query_projection(module: nn.Module) -> None:
    if isinstance(module, SelfAttention):
        query_dim = module.qkv.out_features // 3
        nn.init.zeros_(module.qkv.weight[:query_dim])
        if module.qkv.bias is not None:
            nn.init.zeros_(module.qkv.bias[:query_dim])
        return
    if isinstance(module, CrossAttention):
        nn.init.zeros_(module.q_proj.weight)
        if module.q_proj.bias is not None:
            nn.init.zeros_(module.q_proj.bias)


class Transformer(nn.Module):
    """
    Transformer.

    Args:
        dim: Dimension of the input and output features.
        depth: Number of Transformer blocks in the model.
        head_dim: Dimension of each attention head.
        mlp_ratio: Ratio of the hidden dimension size to the input dimension size in the MLP layers.
        qkv_bias: Whether to use bias in the Q, K, V projections of the attention layers.
        proj_bias: Whether to use bias in the projection layers of the attention.
        mlp_bias: Whether to use bias in the MLP layers.
        drop: Dropout rate applied to attention and MLP layers.
        drop_path_rate: Dropout rate for stochastic depth (drop path).
        act_layer: Activation layer used in the MLPs.
        norm_layer: Normalization layer used before attention and MLP layers.
        gated_mlp: Whether to use gated MLP layers in the transformer blocks.
        qk_norm: Whether to apply normalization to the Q and K projections.
        weight_init_style: Style of weight initialization ('xavier', 'trunc_normal').
        zero_init_query_proj: Whether to zero-initialize the query projection layer.
        adaLN_expansion: Expansion factor for adaLN modulation, e.g. for learning separate
            shift and scale parameters for patches and registers.
    """

    def __init__(
        self,
        dim: int = 768,
        depth: int = 12,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        mlp_bias: bool = False,
        drop: float = 0.0,
        drop_path_rate: float = 0.0,
        act_layer: type[nn.Module] = nn.SiLU,
        norm_layer: Callable[[int], nn.Module] = _DEFAULT_NORM_LAYER,
        gated_mlp: bool = True,
        qk_norm: bool = True,
        weight_init_style: str = "xavier",
        zero_init_query_proj: bool = False,
        use_adaLN: bool = False,
        adaLN_expansion: int = 1,
    ):
        super().__init__()

        block_fn = BlockAdaLN if use_adaLN else Block

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                block_fn(
                    dim=dim,
                    head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    mlp_bias=mlp_bias,
                    drop=drop,
                    drop_path=dpr[i],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    gated_mlp=gated_mlp,
                    qk_norm=qk_norm,
                    adaLN_expansion=adaLN_expansion,
                )
                for i in range(depth)
            ]
        )

        # Weight init
        self.weight_init_style = weight_init_style
        self.zero_init_query_proj = zero_init_query_proj
        self.init_weights_sp()

    def init_weights_sp(self):
        """SP weight initialization scheme"""
        for name, m in self.named_modules():
            # Linear
            if isinstance(m, nn.Linear):
                # Weight
                if "adaLN_modulation" in name:
                    nn.init.constant_(m.weight, 0)
                elif self.weight_init_style == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif self.weight_init_style == "trunc_normal":
                    nn.init.trunc_normal_(m.weight, std=0.02)
                else:
                    raise ValueError(
                        f"Unsupported weight init: {self.weight_init_style}"
                    )
                # Bias
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # LayerNorm
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        if self.zero_init_query_proj:
            for module in self.modules():
                _zero_query_projection(module)

    def forward(
        self,
        x: torch.Tensor,
        block_mask: torch.Tensor | None = None,
        adaLN_emb: torch.Tensor | None = None,
        adaLN_packing_fn: Callable | None = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(
                x,
                block_mask=block_mask,
                adaLN_emb=adaLN_emb,
                adaLN_packing_fn=adaLN_packing_fn,
            )
        return x


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder.

    Args:
        dim: Dimension of the input and output features.
        depth: Number of Transformer blocks in the model.
        head_dim: Dimension of each attention head.
        mlp_ratio: Ratio of the hidden dimension size to the input dimension size in the MLP layers.
        qkv_bias: Whether to use bias in the Q, K, V projections of the attention layers.
        proj_bias: Whether to use bias in the projection layers of the attention.
        mlp_bias: Whether to use bias in the MLP layers.
        drop: Dropout rate applied to attention and MLP layers.
        drop_path_rate: Dropout rate for stochastic depth (drop path).
        act_layer: Activation layer used in the MLPs.
        norm_layer: Normalization layer used before attention and MLP layers.
        gated_mlp: Whether to use gated MLP layers in the transformer blocks.
        qk_norm: Whether to apply normalization to the Q and K projections.
        weight_init_style: Style of weight initialization ('xavier', 'trunc_normal').
        zero_init_query_proj: Whether to zero-initialize the query projection layer.
        adaLN_expansion: Expansion factor for adaLN modulation, e.g. for learning separate
            shift and scale parameters for patches and registers.
    """

    def __init__(
        self,
        dim: int = 768,
        depth: int = 12,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        mlp_bias: bool = False,
        drop: float = 0.0,
        drop_path_rate: float = 0.0,
        act_layer: type[nn.Module] = nn.SiLU,
        norm_layer: Callable[[int], nn.Module] = _DEFAULT_NORM_LAYER,
        gated_mlp: bool = True,
        qk_norm: bool = True,
        weight_init_style: str = "xavier",
        zero_init_query_proj: bool = False,
        use_adaLN: bool = False,
        adaLN_expansion: int = 1,
        enable_self_attn: bool = True,
    ):
        super().__init__()

        block_fn = DecoderBlockAdaLN if use_adaLN else DecoderBlock

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                block_fn(
                    dim=dim,
                    head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    mlp_bias=mlp_bias,
                    drop=drop,
                    drop_path=dpr[i],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    gated_mlp=gated_mlp,
                    qk_norm=qk_norm,
                    enable_self_attn=enable_self_attn,
                    adaLN_expansion=adaLN_expansion,
                )
                for i in range(depth)
            ]
        )

        # Weight init
        self.weight_init_style = weight_init_style
        self.zero_init_query_proj = zero_init_query_proj
        self.init_weights_sp()

    def init_weights_sp(self):
        """SP weight initialization scheme"""
        for name, m in self.named_modules():
            # Linear
            if isinstance(m, nn.Linear):
                # Weight
                if "adaLN_modulation" in name:
                    nn.init.constant_(m.weight, 0)
                elif self.weight_init_style == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif self.weight_init_style == "trunc_normal":
                    nn.init.trunc_normal_(m.weight, std=0.02)
                else:
                    raise ValueError(
                        f"Unsupported weight init: {self.weight_init_style}"
                    )
                # Bias
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # LayerNorm
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        if self.zero_init_query_proj:
            for module in self.modules():
                _zero_query_projection(module)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        block_mask: torch.Tensor | None = None,
        context_block_mask: torch.Tensor | None = None,
        adaLN_emb: torch.Tensor | None = None,
        adaLN_packing_fn: Callable | None = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(
                x,
                context=context,
                block_mask=block_mask,
                context_block_mask=context_block_mask,
                adaLN_emb=adaLN_emb,
                adaLN_packing_fn=adaLN_packing_fn,
            )
        return x
