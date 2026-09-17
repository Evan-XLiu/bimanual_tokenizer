# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from praxis.codecs.factory import available_codecs, make_codec
from praxis.codecs.tokens.btt import BTTEncoderConfig, BTTTokenizer


def _tiny_tokenizer() -> BTTTokenizer:
    return BTTTokenizer(
        BTTEncoderConfig(
            model_dim=32,
            num_heads=4,
            decoder_nhead=4,
            decoder_nlayer=1,
            dropout=0.0,
        )
    )


def test_factory_and_default_decoder_geometry(monkeypatch) -> None:
    assert "tokens/btt" in available_codecs()
    tokenizer = make_codec(
        {
            "kind": "tokens",
            "name": "btt",
            "model_dim": 32,
            "num_heads": 4,
            "decoder_nhead": 4,
            "decoder_nlayer": 1,
            "dropout": 0.0,
        }
    )
    assert isinstance(tokenizer, BTTTokenizer)
    assert tokenizer.codebook_size == 1920
    assert tokenizer.max_code_seq_len == 16
    assert tokenizer.block_schedule == (8, 8)

    default = BTTTokenizer()
    assert default.decoder.post_quant.in_features == 8
    assert default.decoder.post_quant.out_features == 256
    assert default.decoder.action_head.in_features == 256
    assert default.decoder.action_head.out_features == 14
    assert len(default.decoder.decoder.layers) == 6

    left_tokens = torch.zeros(2, 8, dtype=torch.long)
    right_tokens = torch.ones(2, 8, dtype=torch.long)
    left_latents, right_latents = default._tokens_to_latents(left_tokens, right_tokens)
    paired = default.decoder.pair_latents(left_latents, right_latents)
    memory = default.decoder.build_memory(left_latents, right_latents)
    assert left_latents.shape == (2, 8, 4)
    assert right_latents.shape == (2, 8, 4)
    assert paired.shape == (2, 8, 8)
    assert memory.shape == (2, 8, 256)

    # Preserve the default 6-layer construction above while isolating this
    # contract test from the cost of executing all six layers on CPU.
    monkeypatch.setattr(
        default.decoder.decoder,
        "forward",
        lambda tgt, _memory, **_kwargs: tgt,
    )
    decoded = default.decoder.decode_features(left_latents, right_latents)
    norm_actions = default.decoder(left_latents, right_latents)
    assert decoded.shape == (2, 32, 256)
    assert norm_actions.shape == (2, 32, 14)


def test_separate_arm_tokens_and_post_dequantization_shapes() -> None:
    tokenizer = _tiny_tokenizer().eval()
    actions = torch.randn(2, 32, 14)

    encoded = tokenizer.tokenize_with_latents(actions)
    assert encoded.left_tokens.shape == (2, 8)
    assert encoded.right_tokens.shape == (2, 8)
    assert encoded.left_latents.shape == (2, 8, 4)
    assert encoded.right_latents.shape == (2, 8, 4)
    assert encoded.token_ids.shape == (2, 16)

    left_tokens, right_tokens = tokenizer.encode(actions)
    assert left_tokens.shape == (2, 8)
    assert right_tokens.shape == (2, 8)

    left_latents, right_latents = tokenizer._tokens_to_latents(
        left_tokens, right_tokens
    )
    assert left_latents.shape == (2, 8, 4)
    assert right_latents.shape == (2, 8, 4)

    paired = tokenizer.decoder.pair_latents(left_latents, right_latents)
    memory = tokenizer.decoder.build_memory(left_latents, right_latents)
    decoded = tokenizer.decoder.decode_features(left_latents, right_latents)
    norm_actions = tokenizer.decoder(left_latents, right_latents)
    assert paired.shape == (2, 8, 8)
    assert memory.shape == (2, 8, 32)
    assert decoded.shape == (2, 32, 32)
    assert norm_actions.shape == (2, 32, 14)
    assert tokenizer.decode(left_tokens, right_tokens).shape == (2, 32, 14)
    assert tokenizer.detokenize(encoded.token_ids).shape == (2, 32, 14)


def test_pairing_preserves_left_then_right_at_every_timestep() -> None:
    tokenizer = _tiny_tokenizer()
    left = torch.arange(2 * 8 * 4, dtype=torch.float32).reshape(2, 8, 4)
    right = left + 1000.0

    paired = tokenizer.decoder.pair_latents(left, right)

    torch.testing.assert_close(paired[..., :4], left)
    torch.testing.assert_close(paired[..., 4:], right)
    for timestep in range(8):
        torch.testing.assert_close(paired[:, timestep, :4], left[:, timestep])
        torch.testing.assert_close(paired[:, timestep, 4:], right[:, timestep])
    assert not any("arm_embedding" in name for name, _ in tokenizer.named_parameters())


def test_decoder_uses_causal_query_mask_and_no_memory_mask(monkeypatch) -> None:
    tokenizer = _tiny_tokenizer()
    captured: dict[str, object] = {}

    def fake_decoder(tgt, memory, **kwargs):
        captured["tgt_shape"] = tuple(tgt.shape)
        captured["memory_shape"] = tuple(memory.shape)
        captured.update(kwargs)
        return tgt

    monkeypatch.setattr(tokenizer.decoder.decoder, "forward", fake_decoder)
    left = torch.randn(2, 8, 4)
    right = torch.randn(2, 8, 4)
    decoded = tokenizer.decoder.decode_features(left, right)

    assert decoded.shape == (2, 32, 32)
    assert captured["tgt_shape"] == (2, 32, 32)
    assert captured["memory_shape"] == (2, 8, 32)
    tgt_mask = cast(Tensor, captured["tgt_mask"])
    assert isinstance(tgt_mask, torch.Tensor)
    assert tgt_mask.shape == (32, 32)
    assert captured["tgt_is_causal"] is True
    assert "memory_mask" not in captured


def test_tensor_and_training_forward_paths() -> None:
    tokenizer = _tiny_tokenizer()
    actions = torch.randn(2, 32, 14)

    reconstruction = tokenizer(actions)
    assert reconstruction.shape == actions.shape

    loss, metrics = tokenizer({"action": actions})
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert metrics == {}
    loss.backward()
    assert tokenizer.pre_quant.weight.grad is not None
    assert tokenizer.decoder.post_quant.weight.grad is not None
    assert tokenizer.decoder.action_head.weight.grad is not None
