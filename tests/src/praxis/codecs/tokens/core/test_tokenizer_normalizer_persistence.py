# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for tokenizer normalization persistence."""

from __future__ import annotations

import torch

from praxis.core.types import NormMode


class TestTokenizerNormPersistence:
    """BaseTokenizer persists action norm stats via external file."""

    def test_tokenizer_norm_save_load(self):
        from praxis.codecs.tokens.bin.config import BinTokenizerConfig
        from praxis.codecs.tokens.bin.tokenizer import BinTokenizer

        tok = BinTokenizer(
            config=BinTokenizerConfig(action_dim=3, action_horizon=4, num_bins=8)
        )
        stats = {
            "action": {
                "min": torch.tensor([0.0, 0.0, 0.0]),
                "max": torch.tensor([10.0, 10.0, 10.0]),
            }
        }
        tok.set_normalization_stats(stats)

        actions = torch.ones(1, 4, 3) * 5.0
        t1 = tok.tokenize(actions)

        tok2 = BinTokenizer(
            config=BinTokenizerConfig(action_dim=3, action_horizon=4, num_bins=8)
        )
        tok2.set_normalization_stats(stats)
        t2 = tok2.tokenize(actions)
        assert torch.equal(t1, t2)

    def test_tokenizer_no_norm(self):
        from praxis.codecs.tokens.bin.config import BinTokenizerConfig
        from praxis.codecs.tokens.bin.tokenizer import BinTokenizer

        tok = BinTokenizer(
            config=BinTokenizerConfig(action_dim=3, action_horizon=4, num_bins=8)
        )
        assert tok.normalize_action is None
        assert tok.unnormalize_action is None

    def test_tokenizer_honors_explicit_mode(self):
        from praxis.codecs.tokens.bin.config import BinTokenizerConfig
        from praxis.codecs.tokens.bin.tokenizer import BinTokenizer

        tok = BinTokenizer(
            config=BinTokenizerConfig(action_dim=3, action_horizon=4, num_bins=8)
        )
        stats = {
            "action": {
                "min": torch.tensor([0.0, 0.0, 0.0]),
                "max": torch.tensor([10.0, 10.0, 10.0]),
                "mean": torch.tensor([5.0, 5.0, 5.0]),
                "std": torch.tensor([2.0, 2.0, 2.0]),
            }
        }
        tok.set_normalization_stats(stats, modes={"action": NormMode.MEAN_STD})
        x = torch.tensor([[[7.0, 5.0, 3.0]]], dtype=torch.float32)

        out = tok.normalize_action({"action": x})["action"]
        expected = (x - stats["action"]["mean"]) / stats["action"]["std"]
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
