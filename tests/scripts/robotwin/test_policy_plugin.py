# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.robotwin.policy.praxis_ar.deploy_policy import (
    PraxisRobotWinPolicy,
    close_model,
    get_model,
)


class _Client:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.reset_calls: list[list[str]] = []
        self.close_calls = 0

    def health_check(self):
        return True, '{"checkpoint": {"artifact_sha256": "abc"}}'

    def predict_action_from_parts(self, **kwargs):
        self.calls.append(kwargs)
        return np.arange(14, dtype=np.float32)[None, :]

    def reset(self, *, episode_ids):
        self.reset_calls.append(list(episode_ids))
        return True

    def close(self):
        self.close_calls += 1


class _TaskEnv:
    def __init__(self) -> None:
        self.action = None
        self.action_type = None

    def get_instruction(self):
        return "put the apple in the bowl"

    def take_action(self, action, *, action_type):
        self.action = np.asarray(action)
        self.action_type = action_type


def _observation() -> dict:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    return {
        "observation": {
            "head_camera": {"rgb": image + 1},
            "left_camera": {"rgb": image + 2},
            "right_camera": {"rgb": image + 3},
        },
        "joint_action": {"vector": np.arange(14, dtype=np.float32)},
    }


def test_policy_sends_three_chw_images_and_executes_one_qpos_action() -> None:
    client = _Client()
    policy = PraxisRobotWinPolicy(client=client)
    env = _TaskEnv()

    policy.step(env, _observation())

    assert env.action_type == "qpos"
    assert env.action is not None
    assert env.action.shape == (14,)
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["state"].shape == (14,)
    assert set(call["images"]) == {"cam_high", "cam_left_wrist", "cam_right_wrist"}
    assert all(image.shape == (3, 480, 640) for image in call["images"].values())
    assert all(image.dtype == np.uint8 for image in call["images"].values())
    assert call["task_description"] == "put the apple in the bowl"
    assert call["episode_id"] == "robotwin:000000"


def test_policy_initial_reset_keeps_first_episode_index_zero() -> None:
    client = _Client()
    policy = PraxisRobotWinPolicy(client=client)

    policy.reset()

    assert client.reset_calls == [["robotwin:000000"]]
    assert policy.episode_index == 0


def test_policy_reset_advances_after_an_episode_was_observed() -> None:
    client = _Client()
    policy = PraxisRobotWinPolicy(client=client)
    policy.step(_TaskEnv(), _observation())

    policy.reset()
    policy.reset()

    assert client.reset_calls == [["robotwin:000000"], ["robotwin:000001"]]
    assert policy.episode_index == 1


def test_task_namespace_isolates_shared_server_episode_state() -> None:
    first_client = _Client()
    second_client = _Client()
    first = PraxisRobotWinPolicy(
        client=first_client, episode_namespace="adjust_bottle"
    )
    second = PraxisRobotWinPolicy(client=second_client, episode_namespace="click_bell")

    first.step(_TaskEnv(), _observation())
    second.step(_TaskEnv(), _observation())
    first.reset()

    assert first_client.calls[0]["episode_id"] == "adjust_bottle:000000"
    assert second_client.calls[0]["episode_id"] == "click_bell:000000"
    assert first_client.reset_calls == [["adjust_bottle:000000"]]


def test_get_model_preserves_structured_server_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _Client()
    monkeypatch.setattr(
        "scripts.robotwin.policy.praxis_ar.deploy_policy.PolicyClient",
        lambda **_kwargs: client,
    )

    policy = get_model({"task_name": "adjust_bottle"})

    assert policy.server_model_info == {
        "checkpoint": {"artifact_sha256": "abc"}
    }
    assert policy.episode_namespace == "adjust_bottle"


@pytest.mark.parametrize(
    "overrides",
    [
        {"camera_map": "not-json"},
        {"action_dim": "not-an-integer"},
    ],
)
def test_get_model_closes_client_when_plugin_arguments_are_invalid(
    monkeypatch: pytest.MonkeyPatch, overrides: dict[str, str]
) -> None:
    client = _Client()
    monkeypatch.setattr(
        "scripts.robotwin.policy.praxis_ar.deploy_policy.PolicyClient",
        lambda **_kwargs: client,
    )

    with pytest.raises((RuntimeError, ValueError)):
        get_model(overrides)

    assert client.close_calls == 1


def test_close_model_closes_policy_client() -> None:
    client = _Client()
    policy = PraxisRobotWinPolicy(client=client)

    close_model(policy)

    assert client.close_calls == 1


def test_rollout_profile_combines_client_and_server_chunk_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server_trace = tmp_path / "server-profile.jsonl"
    monkeypatch.setenv("ROLLOUT_PROFILE", "1")
    monkeypatch.setenv("ROLLOUT_PROFILE_SERVER_TRACE", str(server_trace))
    events = [
        {
            "episode_id": "robotwin:000000",
            "step": 0,
            "new_inference": True,
            "chunk_remaining_before": 0,
            "predicted_chunk": 2,
            "executed_chunk_length": 2,
            "praxis_preprocess_seconds": 0.01,
            "inference_seconds": 0.02,
            "postprocess_seconds": 0.003,
        },
        {
            "episode_id": "robotwin:000000",
            "step": 1,
            "new_inference": False,
            "chunk_remaining_before": 1,
            "predicted_chunk": None,
            "executed_chunk_length": None,
            "praxis_preprocess_seconds": 0.01,
            "inference_seconds": 0.0,
            "postprocess_seconds": 0.0,
        },
    ]
    policy = PraxisRobotWinPolicy(client=_Client())
    env = _TaskEnv()
    policy.profile_episode_start(reset_seconds=0.1)
    server_trace.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    policy.profile_observation(0.005)
    policy.step(env, _observation())
    policy.profile_observation(0.005)
    policy.step(env, _observation())
    policy.profile_episode_end()

    output = capsys.readouterr().out
    assert "episode_steps:                2" in output
    assert "policy_inference_calls:       1" in output
    assert "returned_chunk_lengths:       [2]" in output
    assert "actions_consumed_per_chunk:   [2]" in output
    assert "step=0  new_inference=True" in output
    assert "step=1  new_inference=False" in output


def test_official_dynamic_importer_finds_package_level_policy_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(Path("scripts/robotwin/policy").resolve()))
    sys.modules.pop("praxis_ar", None)

    module = importlib.import_module("praxis_ar")

    assert callable(module.get_model)
    assert callable(module.eval)
    assert callable(module.reset_model)
    assert callable(module.close_model)
