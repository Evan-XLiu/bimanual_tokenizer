"""Deterministic action binning tokenizer."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor

from praxis.codecs.tokens.bin.config import BinTokenizerConfig
from praxis.codecs.tokens.core import BaseTokenizer


class BinTokenizer(BaseTokenizer):
    """OpenVLA-style action binning tokenizer (adapted for Praxis)."""

    bins: Tensor
    bin_centers: Tensor

    def __init__(
        self, config: dict[str, Any] | BinTokenizerConfig | None = None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        if config is None:
            cfg = BinTokenizerConfig()
        elif isinstance(config, BinTokenizerConfig):
            cfg = config
        else:
            cfg = BinTokenizerConfig.from_dict(config)

        if cfg.num_bins < 2:
            raise ValueError(f"num_bins must be >= 2, got {cfg.num_bins}.")
        if cfg.max_val <= cfg.min_val:
            raise ValueError(
                f"max_val must be greater than min_val, got min_val={cfg.min_val}, max_val={cfg.max_val}."
            )

        self.config = cfg
        if cfg.action_dim is None:
            raise ValueError(
                "BinTokenizer requires action_dim at construction time. "
                "Set tokenizer.action_dim explicitly or build via dataset/PolicyIO-aware path."
            )
        self.action_dim = int(cfg.action_dim)
        self.action_horizon = int(cfg.action_horizon)
        self.num_bins = int(cfg.num_bins)
        self.min_val = float(cfg.min_val)
        self.max_val = float(cfg.max_val)

        bin_centers = torch.linspace(self.min_val, self.max_val, self.num_bins)
        bins = (bin_centers[:-1] + bin_centers[1:]) / 2.0
        self.register_buffer("bins", bins)
        self.register_buffer("bin_centers", bin_centers)

    @property
    def codebook_size(self) -> int:
        return int(cast(Tensor, self.bin_centers).shape[0])

    @property
    def max_code_seq_len(self) -> int:
        return self.action_horizon * self.action_dim

    @property
    def block_schedule(self) -> tuple[int, ...]:
        return (self.action_dim,) * self.action_horizon

    def _tokens_to_grid(self, tokens: Tensor) -> Tensor:
        if tokens.ndim == 3:
            return tokens
        if tokens.ndim != 2:
            raise ValueError(
                f"tokens must be rank-2 or rank-3, got shape={tuple(tokens.shape)}."
            )

        expected = self.action_horizon * self.action_dim
        if tokens.shape[1] != expected:
            raise ValueError(
                f"Flattened token length mismatch: expected {expected} "
                f"(action_horizon={self.action_horizon} * action_dim={self.action_dim}), "
                f"got {tokens.shape[1]}."
            )
        return tokens.view(tokens.shape[0], self.action_horizon, self.action_dim)

    def tokenize(self, actions: Tensor) -> Tensor:
        # actions: (B, T, D) -> tokens: (B, T*D)
        norm_actions = self._normalize_action_tensor(actions)
        norm_actions = torch.clamp(norm_actions, self.min_val, self.max_val)
        bins = cast(Tensor, self.bins)
        token_ids = torch.bucketize(norm_actions, bins, right=False)
        token_ids = torch.clamp(token_ids, 0, self.codebook_size - 1)
        return token_ids.long().reshape(actions.shape[0], -1)

    def detokenize(self, tokens: Tensor, **kwargs) -> Tensor:
        # tokens: (B, T*D) or (B, T, D) -> actions: (B, T, D)
        del kwargs
        token_grid = self._tokens_to_grid(tokens.long())
        bin_centers = cast(Tensor, self.bin_centers)
        token_grid = torch.clamp(token_grid, 0, bin_centers.shape[0] - 1)
        norm_actions = bin_centers[token_grid]
        return self._unnormalize_action_tensor(norm_actions)
