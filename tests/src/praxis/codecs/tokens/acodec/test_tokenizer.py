"""Smoke tests for ActionCodecTokenizer."""

from __future__ import annotations

import pytest
import torch

from praxis.codecs.latents.acodec import ActionCodecAutoencoder
from praxis.codecs.tokens.acodec.config import ActionCodecTokenizerConfig
from praxis.codecs.tokens.acodec.tokenizer import ActionCodecTokenizer


def _tiny_config(**overrides) -> ActionCodecTokenizerConfig:
    base = {
        "action_dim": 3,
        "action_horizon": 8,
        "latent_horizon": 4,
        "emb_dim": 32,
        "latent_dim": 16,
        "head_dim": 8,
        "encoder_depth": 1,
        "decoder_depth": 1,
        "dropout": 0.0,
        "vq_codebook_size": 32,
        "vq_kmeans_init": False,
    }
    base.update(overrides)
    return ActionCodecTokenizerConfig(**base)


class TestActionCodecTokenizer:
    def test_missing_action_dim_raises(self):
        with pytest.raises(ValueError, match="requires action_dim"):
            ActionCodecTokenizer(config=_tiny_config(action_dim=None))

    def test_max_code_seq_len_exposes_token_horizon(self):
        tok = ActionCodecTokenizer(config=_tiny_config())
        assert isinstance(tok, ActionCodecAutoencoder)
        assert tok.max_code_seq_len == 4
        assert tok.block_schedule == (4,)
        assert tok.supports_prefix_decode is False

    def test_forward_encode_decode_tokenize(self):
        tok = ActionCodecTokenizer(config=_tiny_config())
        actions = torch.randn(2, 8, 3)

        loss, metrics = tok({"action": actions})
        assert loss.ndim == 0
        assert "recon_loss" not in metrics
        assert all(not key.startswith("latent_") for key in metrics)
        assert "vq_commitment_loss" in metrics
        assert "vq_codebook_loss" in metrics

        encoded = tok.tokenize_with_latents(actions)
        latents = encoded.latents
        tokens = encoded.token_ids
        assert latents.shape == (2, 4, 16)
        assert tokens.shape == (2, 4)
        assert tok.codebook_size == 32

        recons = tok.decode(latents)
        assert recons.shape == actions.shape

        toks = tok.tokenize(actions)
        assert toks.shape == (2, 4)

        detok = tok.detokenize(toks)
        assert detok.shape == actions.shape

    def test_detokenize_length_validation(self):
        tok = ActionCodecTokenizer(config=_tiny_config())
        with pytest.raises(ValueError, match="token length mismatch"):
            tok.detokenize(torch.zeros((2, 3), dtype=torch.long))

    def test_tokenizer_config_rejects_latent_space(self):
        with pytest.raises(ValueError, match="latent_space"):
            ActionCodecTokenizerConfig.from_dict(
                {
                    "action_dim": 3,
                    "latent_space": {"decoder_noise_std": 0.1},
                }
            )
