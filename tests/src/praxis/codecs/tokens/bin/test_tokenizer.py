"""Unit tests for BinTokenizer."""

from __future__ import annotations

import pytest
import torch

from praxis.codecs.tokens.bin.config import BinTokenizerConfig
from praxis.codecs.tokens.bin.tokenizer import BinTokenizer


def _tiny_config(**overrides) -> BinTokenizerConfig:
    base = {
        "action_dim": 3,
        "action_horizon": 4,
        "num_bins": 8,
        "min_val": -1.0,
        "max_val": 1.0,
    }
    base.update(overrides)
    return BinTokenizerConfig(**base)


class TestBinTokenizer:
    def test_missing_action_dim_raises(self):
        with pytest.raises(ValueError, match="requires action_dim"):
            BinTokenizer(config=_tiny_config(action_dim=None))

    def test_max_code_seq_len_exposes_token_horizon(self):
        tok = BinTokenizer(config=_tiny_config())
        assert tok.max_code_seq_len == 12

    def test_codebook_size_matches_detokenizable_centers(self):
        tok = BinTokenizer(config=_tiny_config(num_bins=8))
        assert tok.codebook_size == 8
        assert tok.codebook_size == tok.bin_centers.shape[0]

    def test_block_schedule_matches_action_horizon_and_dim(self):
        tok = BinTokenizer(config=_tiny_config(action_dim=3, action_horizon=4))
        assert tok.block_schedule == (3, 3, 3, 3)

    def test_tokenize_returns_flat_long_tensor(self):
        tok = BinTokenizer(config=_tiny_config())
        actions = torch.linspace(-1.5, 1.5, steps=24, dtype=torch.float32).reshape(
            2, 4, 3
        )
        tokens = tok.tokenize(actions)

        assert tokens.dtype == torch.long
        assert tokens.shape == (2, 12)
        assert int(tokens.min()) >= 0
        assert int(tokens.max()) < tok.codebook_size

    def test_tokenize_matches_reference_bucketize_logic(self):
        tok = BinTokenizer(config=_tiny_config())
        actions = torch.randn(2, 4, 3)
        tokens = tok.tokenize(actions)

        clamped = torch.clamp(actions, tok.min_val, tok.max_val)
        expected = torch.bucketize(clamped, tok.bins, right=False)
        expected = torch.clamp(expected, 0, tok.codebook_size - 1).long().reshape(2, -1)
        assert torch.equal(tokens, expected)

    def test_detokenize_unflattens_to_action_shape(self):
        tok = BinTokenizer(config=_tiny_config())
        tokens = torch.randint(low=0, high=tok.codebook_size, size=(2, 12))
        actions = tok.detokenize(tokens)
        assert actions.shape == (2, 4, 3)

    def test_normalization_is_applied(self):
        tok = BinTokenizer(config=_tiny_config())
        stats = {
            "action": {
                "min": torch.tensor([0.0, 0.0, 0.0]),
                "max": torch.tensor([10.0, 10.0, 10.0]),
                "mean": torch.tensor([5.0, 5.0, 5.0]),
                "std": torch.tensor([2.0, 2.0, 2.0]),
            }
        }
        tok.set_normalization_stats(stats)
        actions = torch.zeros(1, 4, 3)
        tokens = tok.tokenize(actions)
        # Normalized zeros map to -1, which becomes the first bin after clamping/bucketize.
        assert torch.all(tokens == 0)

    def test_detokenize_accepts_grid_tokens(self):
        tok = BinTokenizer(config=_tiny_config())
        grid_tokens = torch.randint(low=0, high=tok.codebook_size, size=(2, 4, 3))
        actions = tok.detokenize(grid_tokens)
        assert actions.shape == (2, 4, 3)

    def test_detokenize_accepts_last_advertised_token(self):
        tok = BinTokenizer(config=_tiny_config())
        tokens = torch.full(
            (1, tok.max_code_seq_len), tok.codebook_size - 1, dtype=torch.long
        )
        actions = tok.detokenize(tokens)
        expected = tok.bin_centers[-1].expand(1, tok.action_horizon, tok.action_dim)
        assert torch.equal(actions, expected)

    def test_tokenize_max_action_uses_last_advertised_token(self):
        tok = BinTokenizer(config=_tiny_config())
        actions = torch.full(
            (1, tok.action_horizon, tok.action_dim), tok.max_val, dtype=torch.float32
        )
        tokens = tok.tokenize(actions)
        assert torch.all(tokens == tok.codebook_size - 1)
