# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared config helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from praxis.core.config import (
    config_to_dict,
    config_values_equal,
    require_config_dict,
)


@dataclass
class _DataclassConfig:
    name: str
    values: list[int]


class _ToDictConfig:
    def to_dict(self) -> dict[str, object]:
        return {"name": "from_to_dict", "values": [1, 2]}


def test_config_to_dict_supports_common_config_shapes() -> None:
    assert config_to_dict({"name": "dict"}) == {"name": "dict"}
    assert config_to_dict(_DataclassConfig(name="dc", values=[1])) == {
        "name": "dc",
        "values": [1],
    }
    assert config_to_dict(_ToDictConfig()) == {
        "name": "from_to_dict",
        "values": [1, 2],
    }


def test_config_values_equal_ignores_mapping_key_and_sequence_container_noise() -> None:
    assert config_values_equal(
        {"1": [1, {"nested": [2, 3]}]},
        {1: (1, {"nested": (2, 3)})},
    )


def test_require_config_dict_rejects_unsupported_shapes() -> None:
    with pytest.raises(TypeError, match="Unsupported config object type"):
        require_config_dict(object())
