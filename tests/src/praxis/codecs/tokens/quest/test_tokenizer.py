"""Smoke tests for QueSTTokenizer."""

from __future__ import annotations

import pytest
import torch

from praxis.codecs.latents.quest import QueSTAutoencoder
from praxis.codecs.tokens.quest.config import QueSTTokenizerConfig
from praxis.codecs.tokens.quest.tokenizer import QueSTTokenizer


def _tiny_config(**overrides) -> QueSTTokenizerConfig:
    base = {
        "action_dim": 3,
        "action_horizon": 8,
        "encoder_dim": 32,
        "encoder_nhead": 4,
        "encoder_nlayer": 1,
        "decoder_dim": 32,
        "decoder_nhead": 4,
        "decoder_nlayer": 1,
        "dropout": 0.0,
        "downsample_factor": 4,
        "use_causal_encoder": True,
        "use_causal_decoder": True,
        "fsq_levels": (8, 8),
    }
    base.update(overrides)
    return QueSTTokenizerConfig(**base)


class TestQueSTTokenizer:
    def test_missing_action_dim_raises(self):
        with pytest.raises(ValueError, match="requires action_dim"):
            QueSTTokenizer(config=_tiny_config(action_dim=None))

    def test_max_code_seq_len_exposes_token_horizon(self):
        tok = QueSTTokenizer(config=_tiny_config())
        assert isinstance(tok, QueSTAutoencoder)
        assert tok.max_code_seq_len == tok.latent_horizon
        assert tok.supports_prefix_decode is False

    def test_forward_encode_decode_tokenize(self):
        tok = QueSTTokenizer(config=_tiny_config())
        actions = torch.randn(2, 8, 3)

        loss, metrics = tok({"action": actions})
        assert loss.ndim == 0
        assert all(not key.startswith("latent_") for key in metrics)

        encoded = tok.tokenize_with_latents(actions)
        latents = encoded.latents
        tokens = encoded.token_ids
        assert latents.shape[0] == 2
        assert tokens.shape == (2, tok.latent_horizon)

        recons = tok.decode(latents)
        assert recons.shape == actions.shape

        toks = tok.tokenize(actions)
        assert toks.shape == (2, tok.latent_horizon)

        detok = tok.detokenize(toks)
        assert detok.shape == actions.shape

    def test_autoencode_shape(self):
        tok = QueSTTokenizer(config=_tiny_config())
        actions = torch.randn(1, 8, 3)
        out = tok.autoencode(actions)
        assert out.shape == actions.shape

    def test_downsample_factor_validation(self):
        with pytest.raises(ValueError, match="power of two"):
            QueSTTokenizer(config=_tiny_config(downsample_factor=3))

    def test_horizon_divisibility_validation(self):
        with pytest.raises(ValueError, match="divisible"):
            QueSTTokenizer(config=_tiny_config(action_horizon=10, downsample_factor=4))

    def test_detokenize_token_length_validation(self):
        tok = QueSTTokenizer(config=_tiny_config())
        bad = torch.zeros((2, tok.latent_horizon + 1), dtype=torch.long)
        with pytest.raises(ValueError, match="token length mismatch"):
            tok.detokenize(bad)

    def test_tokenizer_config_rejects_latent_space(self):
        with pytest.raises(ValueError, match="latent_space"):
            QueSTTokenizerConfig.from_dict(
                {
                    "action_dim": 3,
                    "latent_space": {"decoder_noise_std": 0.1},
                }
            )
