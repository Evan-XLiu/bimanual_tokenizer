"""Smoke tests for OATAutoencoder."""

from __future__ import annotations

import pytest
import torch

from praxis.codecs.latents.oat import OATAutoencoder, OATAutoencoderConfig


class _SpyEncoderTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, torch.Tensor]] = []

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        self.calls.append(
            {
                "x": x.detach().clone(),
                "context": context.detach().clone(),
            }
        )
        return x


def _tiny_config(**overrides) -> OATAutoencoderConfig:
    base = {
        "action_dim": 3,
        "action_horizon": 8,
        "latent_horizon": 4,
        "latent_dim": 5,
        "emb_dim": 32,
        "head_dim": 8,
        "encoder_depth": 1,
        "decoder_depth": 1,
        "dropout": 0.0,
        "latent_dropout_mode": "pow2",
    }
    base.update(overrides)
    return OATAutoencoderConfig(**base)


def test_oat_autoencoder_forward_encode_decode() -> None:
    codec = OATAutoencoder(config=_tiny_config())
    actions = torch.randn(2, 8, 3)

    assert codec.max_code_seq_len == 4
    assert codec.supports_prefix_decode is True
    assert not hasattr(codec, "block_schedule")

    loss, metrics = codec({"action": actions})
    assert loss.ndim == 0
    assert "latent_mean_loss" in metrics
    assert "latent_variance_loss" in metrics

    latents = codec.encode(actions)
    assert latents.shape == (2, 4, 5)
    assert codec.decode(latents).shape == actions.shape


def test_oat_encoder_adds_fixed_register_position() -> None:
    codec = OATAutoencoder(config=_tiny_config())
    spy = _SpyEncoderTransformer()
    codec.encoder.transformer = spy
    actions = torch.randn(2, 8, 3)

    _ = codec.encode(actions)

    assert len(spy.calls) == 1
    encoder_input = spy.calls[0]["x"]
    expected_pos = codec.encoder.register_pos_emb(
        [codec.latent_horizon],
        device=encoder_input.device,
        dtype=encoder_input.dtype,
    )
    expected = codec.encoder.registers.detach().unsqueeze(0).expand(2, -1, -1)
    expected = expected + expected_pos
    assert torch.allclose(encoder_input, expected)


def test_oat_autoencoder_decodes_latent_prefixes() -> None:
    codec = OATAutoencoder(config=_tiny_config())
    actions = torch.randn(2, 8, 3)
    latents = codec.encode(actions)

    prefix = latents[:, :2]
    assert codec.decode(prefix).shape == actions.shape
    assert codec.decode(latents, decode_keep_k=[2, 3]).shape == actions.shape
    padded = torch.cat([prefix, torch.zeros_like(latents[:, 2:])], dim=1)
    assert torch.allclose(
        codec.decode(prefix),
        codec.decode(padded, decode_keep_k=[2, 2]),
    )


def test_oat_autoencoder_full_decode_keeps_implicit_training_dropout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec = OATAutoencoder(config=_tiny_config())
    latents = torch.randn(2, 4, 5)
    seen_keep_k: list[list[int] | None] = []
    original_forward = codec.decoder.forward

    def wrapped_forward(
        decoder_latents: torch.Tensor,
        *,
        decode_keep_k: list[int] | None = None,
    ) -> torch.Tensor:
        seen_keep_k.append(decode_keep_k)
        return original_forward(decoder_latents, decode_keep_k=decode_keep_k)

    monkeypatch.setattr(codec.decoder, "forward", wrapped_forward)

    codec.decode(latents)

    assert seen_keep_k == [None]


def test_oat_autoencoder_rejects_invalid_prefix_decode_budget() -> None:
    codec = OATAutoencoder(config=_tiny_config())
    prefix = torch.randn(2, 2, 5)

    with pytest.raises(ValueError, match="provided latent length"):
        codec.decode(prefix, decode_keep_k=[3, 2])

    too_long = torch.randn(2, 5, 5)
    with pytest.raises(ValueError, match="latent length mismatch"):
        codec.decode(too_long)


def test_oat_autoencoder_rejects_prefix_decode_when_dropout_disabled() -> None:
    codec = OATAutoencoder(config=_tiny_config(latent_dropout_mode="disable"))
    latents = torch.randn(2, 4, 5)

    assert codec.supports_prefix_decode is False
    assert codec.decode(latents).shape == (2, 8, 3)
    assert codec.decode(latents, decode_keep_k=[4, 4]).shape == (2, 8, 3)
    with pytest.raises(ValueError, match="does not support prefix decode"):
        codec.decode(latents[:, :2])
    with pytest.raises(ValueError, match="does not support prefix decode"):
        codec.decode(latents, decode_keep_k=[2, 2])


def test_oat_autoencoder_uses_configured_latent_dim() -> None:
    codec = OATAutoencoder(config=_tiny_config(latent_dim=2))
    actions = torch.randn(2, 8, 3)
    assert codec.encode(actions).shape == (2, 4, 2)


def test_oat_autoencoder_requires_action_dim() -> None:
    with pytest.raises(ValueError, match="requires action_dim"):
        OATAutoencoder(config=_tiny_config(action_dim=None))
