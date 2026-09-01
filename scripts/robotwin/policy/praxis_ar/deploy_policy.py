# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Official RoboTwin policy plugin backed by a Praxis gRPC server."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from praxis_remote import PolicyClient

DEFAULT_CAMERA_MAP = {
    "cam_high": "head_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}


class _PolicyClient(Protocol):
    def predict_action_from_parts(
        self,
        *,
        state: np.ndarray,
        images: Mapping[str, np.ndarray],
        task_description: str,
        episode_id: str,
    ) -> Any: ...

    def reset(self, *, episode_ids: list[str]) -> bool: ...

    def close(self) -> None: ...


def _parse_camera_map(value: Any) -> dict[str, str]:
    if value is None:
        return dict(DEFAULT_CAMERA_MAP)
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict) or not value:
        raise ValueError("camera_map must be a non-empty mapping or JSON object.")
    return {str(key): str(camera) for key, camera in value.items()}


def _as_chw_uint8(image: Any, *, label: str) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"{label} RGB image must be rank 3, got {array.shape}.")
    if array.shape[-1] in (1, 3, 4):
        array = np.moveaxis(array, -1, 0)
    elif array.shape[0] not in (1, 3, 4):
        raise ValueError(
            f"{label} RGB image has no recognizable channel axis: {array.shape}."
        )
    if array.shape[0] == 4:
        array = array[:3]
    if array.shape[0] != 3:
        raise ValueError(
            f"{label} RGB image must have three channels, got {array.shape}."
        )
    if array.dtype != np.uint8:
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"{label} image dtype must be numeric, got {array.dtype}.")
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


class PraxisRobotWinPolicy:
    """Send each RoboTwin observation to a stateful Praxis AR policy server."""

    def __init__(
        self,
        *,
        client: _PolicyClient,
        camera_map: dict[str, str] | None = None,
        action_dim: int = 14,
        episode_namespace: str = "robotwin",
        server_model_info: dict[str, Any] | None = None,
        trace_path: str | Path | None = None,
    ) -> None:
        self.client = client
        self.camera_map = dict(camera_map or DEFAULT_CAMERA_MAP)
        self.action_dim = int(action_dim)
        self.episode_namespace = str(episode_namespace)
        self.server_model_info = dict(server_model_info or {})
        self.episode_index = 0
        self._episode_recorded = False
        self._trace_step = 0
        self._trace_handle = None
        self._profile_enabled = os.environ.get("ROLLOUT_PROFILE", "0") == "1"
        self._profile_server_trace = os.environ.get("ROLLOUT_PROFILE_SERVER_TRACE")
        self._profile_server_trace_offset = 0
        self._profile_reset()
        if trace_path is not None:
            resolved_trace = Path(trace_path).expanduser().resolve()
            resolved_trace.parent.mkdir(parents=True, exist_ok=True)
            self._trace_handle = resolved_trace.open("a", encoding="utf-8", buffering=1)

    def step(self, task_env: Any, observation: dict[str, Any]) -> None:
        if observation is None:
            raise ValueError(
                "RoboTwin observation must be provided on every policy step."
            )
        instruction = str(task_env.get_instruction())
        self._episode_recorded = True

        prepare_started = time.perf_counter() if self._profile_enabled else 0.0
        observation_data = observation.get("observation")
        if not isinstance(observation_data, dict):
            raise KeyError("RoboTwin observation is missing observation camera data.")
        images: dict[str, np.ndarray] = {}
        for policy_camera, simulator_camera in self.camera_map.items():
            try:
                rgb = observation_data[simulator_camera]["rgb"]
            except (KeyError, TypeError):
                raise KeyError(
                    f"RoboTwin observation is missing {simulator_camera!r} RGB data."
                ) from None
            images[policy_camera] = _as_chw_uint8(rgb, label=simulator_camera)

        try:
            state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        except (KeyError, TypeError):
            raise KeyError(
                "RoboTwin observation is missing joint_action.vector."
            ) from None
        if state.shape != (self.action_dim,):
            raise ValueError(
                f"RoboTwin joint state must have shape ({self.action_dim},), got {state.shape}."
            )

        if self._profile_enabled:
            self._profile["xpolicy_preprocess_seconds"] += (
                time.perf_counter() - prepare_started
            )
        rpc_started = time.perf_counter() if self._profile_enabled else 0.0
        action_batch = np.asarray(
            self.client.predict_action_from_parts(
                state=state,
                images=images,
                task_description=instruction,
                episode_id=self._episode_id,
            ),
            dtype=np.float32,
        )
        if self._profile_enabled:
            self._profile["rpc_seconds"] += time.perf_counter() - rpc_started
        if action_batch.shape == (1, self.action_dim):
            action = action_batch[0]
        elif action_batch.shape == (self.action_dim,):
            action = action_batch
        else:
            raise ValueError(
                "Praxis server returned an invalid RoboTwin action shape: "
                f"{action_batch.shape}; expected ({self.action_dim},) or "
                f"(1, {self.action_dim})."
            )
        if self._trace_handle is not None:
            trace_record = {
                "episode_id": self._episode_id,
                "step": self._trace_step,
                "instruction": instruction,
                "state": state.tolist(),
                "action": action.tolist(),
                "state_parts": {
                    "left_arm": state[:6].tolist(),
                    "left_gripper": float(state[6]),
                    "right_arm": state[7:13].tolist(),
                    "right_gripper": float(state[13]),
                },
                "action_parts": {
                    "left_arm": action[:6].tolist(),
                    "left_gripper": float(action[6]),
                    "right_arm": action[7:13].tolist(),
                    "right_gripper": float(action[13]),
                },
            }
            self._trace_handle.write(json.dumps(trace_record, sort_keys=True) + "\n")
            self._trace_step += 1
        env_step_started = time.perf_counter() if self._profile_enabled else 0.0
        task_env.take_action(action, action_type="qpos")
        if self._profile_enabled:
            self._profile["env_step_seconds"] += time.perf_counter() - env_step_started
            self._profile["episode_steps"] += 1

    def _profile_reset(self) -> None:
        self._profile: dict[str, float | int] = {
            "episode_steps": 0,
            "reset_seconds": 0.0,
            "obs_render_seconds": 0.0,
            "xpolicy_preprocess_seconds": 0.0,
            "rpc_seconds": 0.0,
            "env_step_seconds": 0.0,
        }
        self._profile_episode_started: float | None = None

    def profile_episode_start(self, *, reset_seconds: float = 0.0) -> None:
        if not self._profile_enabled:
            return
        self._profile_reset()
        self._profile["reset_seconds"] = float(reset_seconds)
        self._profile_episode_started = time.perf_counter() - float(reset_seconds)
        if self._profile_server_trace:
            path = Path(self._profile_server_trace)
            self._profile_server_trace_offset = path.stat().st_size if path.is_file() else 0

    def profile_observation(self, elapsed_seconds: float) -> None:
        if self._profile_enabled:
            self._profile["obs_render_seconds"] += float(elapsed_seconds)

    def _server_profile_events(self) -> list[dict[str, Any]]:
        if not self._profile_server_trace:
            return []
        path = Path(self._profile_server_trace)
        if not path.is_file():
            return []
        events: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            handle.seek(self._profile_server_trace_offset)
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("episode_id") == self._episode_id:
                    events.append(event)
        return sorted(events, key=lambda item: int(item.get("step", 0)))

    def profile_episode_end(self) -> None:
        if not self._profile_enabled or self._profile_episode_started is None:
            return
        wall = time.perf_counter() - self._profile_episode_started
        events = self._server_profile_events()
        inference_calls = sum(bool(event.get("new_inference")) for event in events)
        preprocess = sum(float(event.get("praxis_preprocess_seconds", 0.0)) for event in events)
        inference = sum(float(event.get("inference_seconds", 0.0)) for event in events)
        postprocess = sum(float(event.get("postprocess_seconds", 0.0)) for event in events)
        chunks: list[dict[str, int]] = []
        for event in events:
            if event.get("new_inference"):
                chunks.append(
                    {
                        "predicted": int(event.get("predicted_chunk") or 0),
                        "executed": int(event.get("executed_chunk_length") or 0),
                        "consumed": 0,
                    }
                )
            if chunks:
                chunks[-1]["consumed"] += 1
        steps = int(self._profile["episode_steps"])
        rpc = float(self._profile["rpc_seconds"])
        transport = max(0.0, rpc - preprocess - inference - postprocess)
        env_step = float(self._profile["env_step_seconds"])
        non_inference = max(0.0, wall - inference)
        avg_actions = steps / inference_calls if inference_calls else 0.0
        expected = chunks[0]["executed"] if chunks else 0

        print("===== Rollout Timing Summary =====", flush=True)
        print(f"episode_steps:                {steps}", flush=True)
        print(f"episode_wall_time:            {wall:.3f} s", flush=True)
        print(f"steps_per_second:             {steps / wall if wall else 0.0:.3f}", flush=True)
        print(f"policy_inference_calls:       {inference_calls}", flush=True)
        print(f"avg_actions_per_inference:    {avg_actions:.3f}", flush=True)
        print(f"expected_chunk_execution:     {expected}", flush=True)
        print(f"returned_chunk_lengths:       {[c['predicted'] for c in chunks]}", flush=True)
        print(f"actions_consumed_per_chunk:   {[c['consumed'] for c in chunks]}", flush=True)
        self._print_profile_stage("environment reset", float(self._profile["reset_seconds"]), wall)
        self._print_profile_stage("obs/render total", float(self._profile["obs_render_seconds"]), wall)
        self._print_profile_stage("XPolicy preprocess total", float(self._profile["xpolicy_preprocess_seconds"]), wall)
        self._print_profile_stage("serialization/network total", transport, wall)
        self._print_profile_stage("praxis preprocess total", preprocess, wall)
        self._print_profile_stage("policy inference total", inference, wall)
        self._print_profile_stage("postprocess total", postprocess, wall)
        self._print_profile_stage("env.step total", env_step, wall)
        print(f"avg policy inference:         {1000 * inference / inference_calls if inference_calls else 0.0:.3f} ms", flush=True)
        print(f"avg env.step:                 {1000 * env_step / steps if steps else 0.0:.3f} ms", flush=True)
        print(f"avg non-inference step:       {1000 * non_inference / steps if steps else 0.0:.3f} ms", flush=True)
        print("first 30 profiled steps:", flush=True)
        for event in events[:30]:
            detail = (
                f"step={int(event.get('step', 0))}  "
                f"new_inference={bool(event.get('new_inference'))}  "
                f"chunk_remaining_before={int(event.get('chunk_remaining_before', 0))}"
            )
            if event.get("new_inference"):
                detail += f"  predicted_chunk={int(event.get('predicted_chunk') or 0)}"
            print(detail, flush=True)
        print("=================================", flush=True)

    @staticmethod
    def _print_profile_stage(label: str, seconds: float, wall: float) -> None:
        print(f"{label + ':':28s} {seconds:9.3f} s   ({100 * seconds / wall if wall else 0.0:5.1f}%)", flush=True)

    def reset(self) -> None:
        if not self.client.reset(episode_ids=[self._episode_id]):
            raise RuntimeError("Praxis server rejected the episode reset request.")
        if self._episode_recorded:
            self.episode_index += 1
        self._episode_recorded = False
        self._trace_step = 0

    @property
    def _episode_id(self) -> str:
        return f"{self.episode_namespace}:{self.episode_index:06d}"

    def close(self) -> None:
        self.client.close()
        if self._trace_handle is not None:
            self._trace_handle.close()
            self._trace_handle = None


def get_model(usr_args: dict[str, Any]) -> PraxisRobotWinPolicy:
    host = str(usr_args.get("host", "127.0.0.1"))
    port = int(usr_args.get("port", 50051))
    timeout_value = usr_args.get("timeout", 120.0)
    timeout = None if timeout_value is None else float(timeout_value)
    client = PolicyClient(host=host, port=port, timeout=timeout)
    try:
        ready, model_info = client.health_check()
        if not ready:
            raise RuntimeError(f"Praxis policy server is not ready: {model_info}")
        try:
            parsed_model_info = json.loads(model_info)
        except (json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError(
                "Praxis policy server returned non-auditable model info; expected JSON."
            ) from exc
        if not isinstance(parsed_model_info, dict):
            raise RuntimeError("Praxis policy server model info must be a JSON object.")
        camera_map = _parse_camera_map(usr_args.get("camera_map"))
        action_dim = int(usr_args.get("action_dim", 14))
        episode_namespace = str(
            usr_args.get("episode_namespace")
            or usr_args.get("task_name")
            or "robotwin"
        )
        return PraxisRobotWinPolicy(
            client=client,
            camera_map=camera_map,
            action_dim=action_dim,
            episode_namespace=episode_namespace,
            server_model_info=parsed_model_info,
            trace_path=usr_args.get("trace_path") or os.environ.get("ROBOTWIN_POLICY_TRACE_PATH"),
        )
    except BaseException:
        client.close()
        raise


def eval(
    task_env: Any, model: PraxisRobotWinPolicy, observation: dict[str, Any]
) -> None:
    model.step(task_env, observation)


def reset_model(model: PraxisRobotWinPolicy) -> None:
    model.reset()


def close_model(model: PraxisRobotWinPolicy) -> None:
    model.close()
