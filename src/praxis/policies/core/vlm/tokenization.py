"""Action-token codec helpers for VLM CE policies."""

from __future__ import annotations

from typing import Protocol, cast

import torch
from torch import Tensor

from praxis.codecs.tokens.core import BaseTokenizer


class _TokenizeCallable(Protocol):
    def tokenize(self, actions: Tensor) -> Tensor | list[Tensor] | list[list[int]]: ...


def require_token_codec(codec: object, *, owner_name: str) -> BaseTokenizer:
    """Validate that a constructed codec has the token-codec runtime contract."""
    missing = [
        name
        for name in ("tokenize", "detokenize")
        if not callable(getattr(codec, name, None))
    ]
    if not hasattr(codec, "codebook_size"):
        missing.append("codebook_size")
    if missing:
        raise TypeError(
            f"{owner_name}.action_tokenizer must resolve to a token codec; "
            f"{type(codec).__name__} is missing {', '.join(missing)}."
        )
    return cast(BaseTokenizer, codec)


def tokenize_actions(action_tokenizer: object, actions: Tensor) -> list[list[int]]:
    """Tokenize actions into Python token-id rows for CE label rendering."""
    tokenizer = cast(_TokenizeCallable, action_tokenizer)
    with torch.inference_mode():
        raw = tokenizer.tokenize(actions)
    if isinstance(raw, Tensor):
        return raw.long().tolist()
    return [
        row.long().tolist() if isinstance(row, Tensor) else [int(item) for item in row]
        for row in raw
    ]
