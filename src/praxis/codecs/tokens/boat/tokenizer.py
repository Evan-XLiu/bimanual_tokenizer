"""BOAT tokenizer implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from torch import Tensor

from praxis.codecs.latents.oat.schedule import (
    register_schedule_prefix_boundaries,
)
from praxis.codecs.tokens.boat.config import BOATTokenizerConfig
from praxis.codecs.tokens.oat.tokenizer import OATTokenizer


class BOATTokenizer(OATTokenizer):
    """BOAT is an OAT tokenizer with a coarse-to-fine register schedule."""

    config: BOATTokenizerConfig

    def __init__(
        self, config: dict[str, Any] | BOATTokenizerConfig | None = None, **kwargs
    ) -> None:
        if config is None:
            cfg = BOATTokenizerConfig()
        elif isinstance(config, BOATTokenizerConfig):
            cfg = config
        else:
            cfg = BOATTokenizerConfig.from_dict(config)
        super().__init__(config=cfg, **kwargs)

    def _decode_to_norm_actions(self, latents: Tensor, **kwargs: Any) -> Tensor:
        latents, decode_keep_k = self._prepare_decode_latents(
            latents,
            decode_keep_k=kwargs.get("decode_keep_k"),
        )
        if decode_keep_k is not None:
            self._validate_decode_keep_k_boundaries(decode_keep_k)
        return self.decoder(latents, decode_keep_k=decode_keep_k)

    def _validate_decode_keep_k_boundaries(self, decode_keep_k: Sequence[int]) -> None:
        valid = set(register_schedule_prefix_boundaries(self.block_schedule))
        invalid = [int(k) for k in decode_keep_k if int(k) not in valid]
        if invalid:
            raise ValueError(
                "BOAT decode_keep_k values must be cumulative block boundaries "
                f"{sorted(valid)}, got {invalid}."
            )
