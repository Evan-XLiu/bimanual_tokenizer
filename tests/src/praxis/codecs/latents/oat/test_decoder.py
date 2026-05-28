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
                "memory": memory.detach().clone(),
                "tgt_shape": tuple(tgt.shape),
                "memory_shape": tuple(memory.shape),
            }
        )
        return tgt


def test_oat_single_pass_decoder_uses_latent_cross_attention() -> None:
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
    assert len(spy.calls) == 1
    assert spy.calls[0]["kwargs"] == {}
    assert spy.calls[0]["tgt_shape"] == (2, 8, 32)
    assert spy.calls[0]["memory_shape"] == (2, 4, 32)


def test_oat_decoder_disables_action_self_attention() -> None:
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


def test_oat_decoder_adds_latent_position_after_prefix_masking() -> None:
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
    decoder.eval()

    latents = torch.randn(2, 4, 2)
    _ = decoder(latents, decode_keep_k=[2, 4])

    memory = spy.calls[0]["memory"]
    assert isinstance(memory, torch.Tensor)
    latent_pos = decoder.latent_pos_emb([4], device=memory.device, dtype=memory.dtype)
    mask_token = decoder.nested_dropout.dropout_mask_token.to(
        device=memory.device, dtype=memory.dtype
    )
    expected_masked_suffix = mask_token.view(1, 1, -1) + latent_pos[:, 2:, :]
    assert torch.allclose(memory[:1, 2:, :], expected_masked_suffix)
