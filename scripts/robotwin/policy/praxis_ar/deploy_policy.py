# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Official RoboTwin policy plugin backed by a Praxis gRPC server."""

from __future__ import annotations

import json
from collections.abc import Mapping
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
    ) -> None:
        self.client = client
        self.camera_map = dict(camera_map or DEFAULT_CAMERA_MAP)
        self.action_dim = int(action_dim)
        self.episode_namespace = str(episode_namespace)
        self.server_model_info = dict(server_model_info or {})
        self.episode_index = 0
        self._episode_recorded = False

    def step(self, task_env: Any, observation: dict[str, Any]) -> None:
        if observation is None:
            raise ValueError(
                "RoboTwin observation must be provided on every policy step."
            )
        instruction = str(task_env.get_instruction())
        self._episode_recorded = True

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

        action_batch = np.asarray(
            self.client.predict_action_from_parts(
                state=state,
                images=images,
                task_description=instruction,
                episode_id=self._episode_id,
            ),
            dtype=np.float32,
        )
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
        task_env.take_action(action, action_type="qpos")

    def reset(self) -> None:
        if not self.client.reset(episode_ids=[self._episode_id]):
            raise RuntimeError("Praxis server rejected the episode reset request.")
        if self._episode_recorded:
            self.episode_index += 1
        self._episode_recorded = False

    @property
    def _episode_id(self) -> str:
        return f"{self.episode_namespace}:{self.episode_index:06d}"

    def close(self) -> None:
        self.client.close()


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
        episode_namespace = str(usr_args.get("task_name") or "robotwin")
        return PraxisRobotWinPolicy(
            client=client,
            camera_map=camera_map,
            action_dim=action_dim,
            episode_namespace=episode_namespace,
            server_model_info=parsed_model_info,
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
