# SPDX-FileCopyrightText: 2026 Chaoqi Liu
# SPDX-License-Identifier: Apache-2.0

"""BTT configuration with a distinct single-block policy interface."""

from dataclasses import dataclass

from praxis.codecs.tokens.btt.config import BTTEncoderConfig


@dataclass
class BTTOneBlockConfig(BTTEncoderConfig):
    """Preserve BTT geometry and training; expose all tokens in one block."""

    name: str = "btt_oneblock"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.name != "btt_oneblock":
            raise ValueError(
                "Expected a btt_oneblock config/checkpoint. A standard BTT "
                "checkpoint must be explicitly converted before using this interface."
            )
