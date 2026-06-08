# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Base class for discrete action-token codecs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor

from praxis.codecs.core import BaseCodec


class BaseTokenizer(BaseCodec, ABC):
    """Base class for discrete action-token codecs.

    Maps between continuous action sequences and discrete token sequences.
    Shared codec lifecycle lives in :class:`praxis.codecs.core.BaseCodec`;
    this class owns token-only methods. Generic code-sequence capabilities live
    on :class:`praxis.codecs.core.BaseCodec`.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @property
    def block_schedule(self) -> tuple[int, ...] | None:
        """Optional block-generation schedule for token policies.

        ``None`` means the tokenizer does not declare a block schedule and
        policy generation should fall back to standard autoregressive decoding.
        Explicit schedules opt into token-level BAR or PD behavior.
        """
        return None

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        del batch
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement forward()."
        )

    @abstractmethod
    def tokenize(self, actions: Tensor) -> Tensor:
        """Encode actions to discrete token indices.

        Args:
            actions: ``(B, T, D)`` continuous action sequences.

        Returns:
            Token indices ``(B, L)`` of dtype int64.
        """

    @abstractmethod
    def detokenize(self, tokens: Tensor, **kwargs: Any) -> Tensor:
        """Decode discrete tokens back to continuous actions.

        Args:
            tokens: ``(B, L)`` integer token indices.

        Returns:
            Reconstructed actions ``(B, T, D)``.
        """

    @property
    @abstractmethod
    def codebook_size(self) -> int:
        """Vocabulary size of the tokenizer."""
