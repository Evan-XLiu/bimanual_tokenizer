# SPDX-License-Identifier: Apache-2.0

"""Metadata-only codec for token targets produced by an offline encoder."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from praxis.codecs.tokens.core import BaseTokenizer


class OfflineTargetTokenizer(BaseTokenizer):
    def __init__(self, config: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        config = dict(config or {})
        self.config = config
        self._codebook_size = int(config.get("codebook_size", 2048))
        self._block_schedule = tuple(
            int(value) for value in config.get("block_schedule", (1,) * 32)
        )
        if (
            self._codebook_size <= 0
            or not self._block_schedule
            or any(value <= 0 for value in self._block_schedule)
        ):
            raise ValueError(
                "Offline target codec requires a positive codebook and block schedule."
            )

    @property
    def codebook_size(self) -> int:
        return self._codebook_size

    @property
    def block_schedule(self) -> tuple[int, ...]:
        return self._block_schedule

    @property
    def max_code_seq_len(self) -> int:
        return sum(self._block_schedule)

    def tokenize(self, actions: Tensor) -> Tensor:
        del actions
        raise RuntimeError(
            "Offline target codec cannot tokenize actions; provide cached action_token_ids."
        )

    def detokenize(self, tokens: Tensor, **kwargs: Any) -> Tensor:
        del tokens, kwargs
        raise RuntimeError(
            "Offline target codec cannot decode tokens; KI inference uses its flow branch."
        )
