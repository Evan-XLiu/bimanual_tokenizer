# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for BOATTokenizer."""

from __future__ import annotations

import pytest
import torch

from praxis.codecs.latents.oat.modeling import SinglePassDecoder
from praxis.codecs.latents.oat.schedule import (
    create_register_schedule_mask,
    pow2_register_schedule,
)
from praxis.codecs.tokens.boat.config import BOATTokenizerConfig
from praxis.codecs.tokens.boat.tokenizer import BOATTokenizer
from praxis.codecs.tokens.oat.tokenizer import OATTokenizer


def _tiny_config(**overrides) -> BOATTokenizerConfig:
    base = {
        "action_dim": 3,
        "action_horizon": 8,
        "latent_horizon": 4,
        "emb_dim": 32,
        "head_dim": 8,
        "encoder_depth": 1,
        "decoder_depth": 1,
        "dropout": 0.0,
        "quantizer_levels": (8, 8),
        "latent_dropout_mode": "pow2",
    }
    base.update(overrides)
    return BOATTokenizerConfig(**base)


class TestBOATTokenizer:
    def test_missing_action_dim_raises(self):
        with pytest.raises(ValueError, match="requires action_dim"):
            BOATTokenizer(config=_tiny_config(action_dim=None))

    def test_latent_horizon_must_be_power_of_two(self):
        with pytest.raises(ValueError, match="power of 2"):
            BOATTokenizerConfig(action_dim=3, latent_horizon=6)

    def test_block_schedule_is_derived_from_latent_horizon(self):
        cfg = _tiny_config(latent_horizon=16)
        assert cfg.register_schedule == (1, 1, 2, 4, 8)
        assert cfg.block_schedule == (1, 1, 2, 4, 8)

        tok = BOATTokenizer(config=cfg)
        assert tok.block_schedule == (1, 1, 2, 4, 8)
        assert tok.encoder.register_schedule == (1, 1, 2, 4, 8)
        assert isinstance(tok, OATTokenizer)

    def test_block_mask_is_self_only_within_level_and_causal_across_levels(self):
        mask = create_register_schedule_mask(pow2_register_schedule(8), "cpu")
        assert mask.shape == (8, 8)

        # Same-level off-diagonal attention is disabled.
        assert not bool(mask[2, 3])
        assert not bool(mask[3, 2])

        # Self attention remains enabled.
        assert bool(mask[2, 2])
        assert bool(mask[7, 7])

        # Later levels can see earlier levels.
        assert bool(mask[4, 2])

        # Earlier levels cannot see future levels.
        assert not bool(mask[2, 4])
        assert not bool(mask[4, 7])
        assert not bool(mask[7, 4])

    def test_forward_encode_decode_tokenize(self):
        tok = BOATTokenizer(config=_tiny_config())
        assert isinstance(tok.decoder, SinglePassDecoder)
        actions = torch.randn(2, 8, 3)

        loss, metrics = tok({"action": actions})
        assert loss.ndim == 0
        assert metrics == {}

        encoded = tok.tokenize_with_latents(actions)
        latents = encoded.latents
        tokens = encoded.token_ids
        assert latents.shape == (2, 4, 2)
        assert tokens.shape == (2, 4)
        assert tok.variable_length_codes is False
        assert tok.max_code_seq_len == 4

        recons = tok.decode(latents)
        assert recons.shape == actions.shape

        toks = tok.tokenize(actions)
        assert toks.shape == (2, 4)

        detok = tok.detokenize(toks)
        assert detok.shape == actions.shape

    def test_detokenize_variable_length_sequences(self):
        tok = BOATTokenizer(config=_tiny_config())
        assert tok.supports_prefix_decode is True
        detok = tok.detokenize([[1], [3, 4]])
        assert detok.shape == (2, 8, 3)

    def test_detokenize_rejects_non_block_boundary_prefixes(self):
        tok = BOATTokenizer(config=_tiny_config())
        with pytest.raises(ValueError, match="cumulative block boundaries"):
            tok.detokenize([[1, 2, 3]])

        latents = torch.randn(1, 4, 2)
        with pytest.raises(ValueError, match="cumulative block boundaries"):
            tok.decode(latents, decode_keep_k=[3])

    def test_decode_rejects_keep_k_beyond_provided_prefix(self):
        tok = BOATTokenizer(config=_tiny_config())
        latents = torch.randn(2, 4, 2)

        with pytest.raises(ValueError, match="provided latent length"):
            tok.decode(latents[:, :2], decode_keep_k=[4, 4])

    def test_full_decode_keeps_implicit_training_dropout(self, monkeypatch):
        tok = BOATTokenizer(config=_tiny_config())
        latents = torch.randn(2, 4, 2)
        seen_keep_k = []
        original_forward = tok.decoder.forward

        def wrapped_forward(decoder_latents, *, decode_keep_k=None):
            seen_keep_k.append(decode_keep_k)
            return original_forward(decoder_latents, decode_keep_k=decode_keep_k)

        monkeypatch.setattr(tok.decoder, "forward", wrapped_forward)

        tok.decode(latents)

        assert seen_keep_k == [None]
