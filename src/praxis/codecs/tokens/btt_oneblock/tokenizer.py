# SPDX-FileCopyrightText: 2026 Chaoqi Liu
# SPDX-License-Identifier: Apache-2.0

"""Experiment 2: unchanged BTT autoencoder, one VLA generation block."""

from typing import Any

from praxis.codecs.tokens.btt.tokenizer import BTTTokenizer
from praxis.codecs.tokens.btt_oneblock.config import BTTOneBlockConfig


class BTTOneBlockTokenizer(BTTTokenizer):
    """Keep [left_0..7, right_0..7] IDs and decode jointly as in BTT."""

    config: BTTOneBlockConfig

    def __init__(
        self, config: dict[str, Any] | BTTOneBlockConfig | None = None, **kwargs: Any
    ) -> None:
        if config is None:
            cfg = BTTOneBlockConfig()
        elif isinstance(config, BTTOneBlockConfig):
            cfg = config
        else:
            cfg = BTTOneBlockConfig.from_dict(config)
        super().__init__(config=cfg, **kwargs)

    @property
    def block_schedule(self) -> tuple[int]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Only policy generation changes: 16 parallel tokens, not 8 then 8."""
        return (self.max_code_seq_len,)

    @classmethod
    def _from_pretrained(cls, **kwargs: Any):
        if kwargs.get("strict", True) is not True:
            raise ValueError("btt_oneblock requires strict checkpoint loading.")
        kwargs["strict"] = True
        return super()._from_pretrained(**kwargs)
