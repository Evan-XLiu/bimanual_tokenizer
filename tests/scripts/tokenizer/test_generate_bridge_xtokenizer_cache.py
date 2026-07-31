# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from scripts.tokenizer.generate_bridge_xtokenizer_cache import (
    _canonicalize_native_normalization,
)


def _exported_normalization() -> dict[str, object]:
    return {
        "kind": "xtokenizer_dual_role_v1",
        "roles": {},
        "modes": {"action": "min_max", "observation.state": "min_max"},
        "identity": {"format_version": 3},
        "stats": {
            "action": {
                "min": [float(np.float32(-1.25))],
                "max": [float(np.float32(1.25))],
            },
            "observation.state": {
                "min": [float(np.float32(-2.5))],
                "max": [float(np.float32(2.5))],
            },
        },
    }


def _training_data_identity() -> dict[str, object]:
    return {
        "canonical_stats_identity": {
            "normalization": {
                "predict_action": {"q001": [-1.25000001], "q999": [1.25000001]},
                "observation_state": {
                    "q001": [-2.50000001],
                    "q999": [2.50000001],
                },
            }
        }
    }


def test_native_normalization_uses_canonical_json_bounds_without_changing_runtime() -> (
    None
):
    exported = _exported_normalization()

    corrected = _canonicalize_native_normalization(
        exported,
        _training_data_identity(),
    )

    assert corrected is not exported
    assert corrected["stats"] == {
        "action": {"min": [-1.25000001], "max": [1.25000001]},
        "observation.state": {"min": [-2.50000001], "max": [2.50000001]},
    }
    assert exported["stats"]["action"]["min"] == [float(np.float32(-1.25))]


def test_native_normalization_rejects_canonical_bounds_that_change_runtime_values() -> (
    None
):
    identity = _training_data_identity()
    identity["canonical_stats_identity"]["normalization"]["predict_action"]["q001"] = [
        -1.0
    ]

    with pytest.raises(ValueError, match="would change runtime values"):
        _canonicalize_native_normalization(_exported_normalization(), identity)
