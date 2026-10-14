"""Protocol checks for the HTTP FM box bridge without loading model weights."""

import json

import numpy as np
import pytest

from scripts.serve_fm_box_http import (
    _decode_numpy_object,
    _encode_numpy_object,
    parse_observation,
)


def test_json_numpy_camera_and_state_request() -> None:
    cameras = {
        name: np.zeros((8, 12, 3), dtype=np.uint8)
        for name in ("top", "left", "right")
    }
    payload = {
        **cameras,
        "state": np.arange(14, dtype=np.float32),
        "instruction": "Put all blocks into the box.",
        "num_steps": 10,
    }
    decoded = json.loads(
        json.dumps(payload, default=_encode_numpy_object),
        object_hook=_decode_numpy_object,
    )
    observation, steps = parse_observation(decoded)
    assert steps == 10
    assert observation["state"].shape == (14,)
    assert observation["top"].dtype == np.uint8
    np.testing.assert_array_equal(observation["state"], payload["state"])


def test_eef_state_and_missing_camera_are_rejected() -> None:
    payload = {
        "top": np.zeros((4, 4, 3), dtype=np.uint8),
        "left": np.zeros((4, 4, 3), dtype=np.uint8),
        "right": np.zeros((4, 4, 3), dtype=np.uint8),
        "state": np.zeros(16, dtype=np.float32),
        "instruction": "box",
    }
    with pytest.raises(ValueError, match="14-dimensional"):
        parse_observation(payload)
    payload["state"] = np.zeros(14, dtype=np.float32)
    del payload["right"]
    with pytest.raises(ValueError, match="right_cam"):
        parse_observation(payload)
