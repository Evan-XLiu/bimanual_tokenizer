# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn as nn

from praxis.codecs.latents.oat.modeling import SinglePassDecoder
from praxis.nn.architectures.transformer import TransformerDecoder


class _SpyTransformerDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, object]] = []

    def forward(
        self, tgt: torch.Tensor, memory: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        self.calls.append(
            {
                "kwargs": dict(kwargs),
                "tgt_shape": tuple(tgt.shape),
                "memory_shape": tuple(memory.shape),
            }
        )
        return tgt


def test_boat_single_pass_decoder_uses_latent_cross_attention() -> None:
    decoder = SinglePassDecoder(
        sample_dim=3,
        sample_horizon=8,
        emb_dim=32,
        head_dim=8,
        depth=1,
        pdropout=0.0,
        latent_dropout_mode="pow2",
        latent_dim=2,
        latent_horizon=4,
    )
    spy = _SpyTransformerDecoder()
    decoder.decoder = spy

    out = decoder(torch.randn(2, 4, 2), decode_keep_k=[2, 4])

    assert tuple(out.shape) == (2, 8, 3)
    assert spy.calls == [
        {
            "kwargs": {},
            "tgt_shape": (2, 8, 32),
            "memory_shape": (2, 4, 32),
        }
    ]


def test_boat_decoder_disables_action_self_attention() -> None:
    decoder = SinglePassDecoder(
        sample_dim=3,
        sample_horizon=8,
        emb_dim=32,
        head_dim=8,
        depth=2,
        pdropout=0.0,
        latent_dropout_mode="pow2",
        latent_dim=2,
        latent_horizon=4,
    )

    assert isinstance(decoder.decoder, TransformerDecoder)
    assert all(block.self_attn is None for block in decoder.decoder.blocks)
