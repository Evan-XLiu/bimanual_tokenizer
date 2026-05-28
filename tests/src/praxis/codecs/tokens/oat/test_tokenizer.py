"""Smoke tests for OATTokenizer."""

from __future__ import annotations

import pytest
import torch

from praxis.codecs.latents.oat import OATAutoencoder
from praxis.codecs.tokens.oat.config import OATTokenizerConfig
from praxis.codecs.tokens.oat.tokenizer import OATTokenizer


def _tiny_config(**overrides) -> OATTokenizerConfig:
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
    return OATTokenizerConfig(**base)


class TestOATTokenizer:
    def test_missing_action_dim_raises(self):
        with pytest.raises(ValueError, match="requires action_dim"):
            OATTokenizer(config=_tiny_config(action_dim=None))

    def test_latent_dim_is_inferred_from_fsq_levels(self):
        tok = OATTokenizer(config=_tiny_config(quantizer_levels=(8, 8, 8)))
        assert isinstance(tok, OATAutoencoder)
        actions = torch.randn(2, 8, 3)
        assert tok.encode(actions).shape[-1] == 3

    def test_tokenizer_latent_dim_must_match_fsq_levels(self):
        with pytest.raises(ValueError, match="latent_dim must match"):
            OATTokenizer(config=_tiny_config(latent_dim=4, quantizer_levels=(8, 8)))

    def test_max_code_seq_len_exposes_token_horizon(self):
        tok = OATTokenizer(config=_tiny_config())
        assert tok.max_code_seq_len == 4
        assert tok.block_schedule == (1, 1, 1, 1)

    def test_forward_encode_decode_tokenize(self):
        tok = OATTokenizer(config=_tiny_config())
        actions = torch.randn(2, 8, 3)

        loss, metrics = tok({"action": actions})
        assert loss.ndim == 0
        assert all(not key.startswith("latent_") for key in metrics)

        encoded = tok.tokenize_with_latents(actions)
        latents = encoded.latents
        tokens = encoded.token_ids
        assert latents.shape == (2, 4, 2)
        assert tokens.shape == (2, 4)

        recons = tok.decode(latents)
        assert recons.shape == actions.shape

        toks = tok.tokenize(actions)
        assert toks.shape == (2, 4)

        detok = tok.detokenize(toks)
        assert detok.shape == actions.shape

    def test_detokenize_variable_length_sequences(self):
        tok = OATTokenizer(config=_tiny_config())
        assert tok.supports_prefix_decode is True
        detok = tok.detokenize([[1, 2], [3, 4, 5]])
        assert detok.shape == (2, 8, 3)

    def test_detokenize_rejects_prefix_when_dropout_disabled(self):
        tok = OATTokenizer(config=_tiny_config(latent_dropout_mode="disable"))
        assert tok.supports_prefix_decode is False

        full_tokens = torch.zeros((2, tok.latent_horizon), dtype=torch.long)
        assert tok.detokenize(full_tokens).shape == (2, 8, 3)
        with pytest.raises(ValueError, match="does not support prefix decode"):
            tok.detokenize([[1, 2], [3, 4, 5]])

    def test_decode_keep_k_validation(self):
        tok = OATTokenizer(config=_tiny_config())
        latents = torch.randn(2, 4, 2)
        with pytest.raises(ValueError, match="must match batch size"):
            tok.decode(latents, decode_keep_k=[4])
        with pytest.raises(ValueError, match="must be <="):
            tok.decode(latents, decode_keep_k=[4, 5])

    def test_tokenizer_config_rejects_latent_space(self):
        with pytest.raises(ValueError, match="latent_space"):
            OATTokenizerConfig.from_dict(
                {
                    "action_dim": 3,
                    "latent_space": {"decoder_noise_std": 0.1},
                }
            )

    def test_tokenizer_config_loads_legacy_token_dropout_mode(self):
        cfg = OATTokenizerConfig.from_dict(
            {
                "action_dim": 3,
                "token_dropout_mode": "pow2",
                "quantizer_levels": [8, 8],
            }
        )

        assert cfg.latent_dropout_mode == "pow2"
        assert cfg.quantizer_levels == (8, 8)
