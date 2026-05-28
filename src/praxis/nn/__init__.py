"""Shared neural network building blocks for Praxis."""

from praxis.nn.architectures.transformer import Transformer, TransformerDecoder
from praxis.nn.blocks.attention import Attention, CrossAttention, SelfAttention
from praxis.nn.blocks.mlp import GatedMlp, Mlp
from praxis.nn.blocks.transformer_block import (
    Block,
    BlockAdaLN,
    DecoderBlock,
    DecoderBlockAdaLN,
)
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
from praxis.nn.layers.drop_path import DropPath, drop_path
from praxis.nn.layers.norm import Fp32LayerNorm, Fp32RMSNorm
from praxis.nn.vision.rgb_encoder import (
    ResNetSpatialSoftmaxEncoder,
    SpatialSoftmax,
    replace_bn_with_gn,
)

__all__ = [
    "Attention",
    "Block",
    "BlockAdaLN",
    "CrossAttention",
    "DecoderBlock",
    "DecoderBlockAdaLN",
    "DropPath",
    "Fp32LayerNorm",
    "Fp32RMSNorm",
    "GatedMlp",
    "Mlp",
    "LearnedPosEmbed",
    "SinusoidalScalarEmbedding",
    "RotaryPosEmbed",
    "SelfAttention",
    "SpatialSoftmax",
    "SinusoidPosEmbed",
    "Transformer",
    "TransformerDecoder",
    "build_1d_sinusoid_pos_embed",
    "build_2d_sinusoid_pos_embed",
    "ResNetSpatialSoftmaxEncoder",
    "geometric_frequency_grid",
    "replace_bn_with_gn",
    "sinusoidal_scalar_features",
    "drop_path",
]
