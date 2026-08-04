# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Adapters between Praxis policies and the standalone ``praxis_eval`` API."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from praxis_eval.contracts import ActionSpec
from praxis_eval.policies.actions import normalize_batched_action
from praxis_eval.types import Observation

from praxis.core.types import PolicyIO
from praxis.policies.core import BasePolicy, PolicyRolloutState
from praxis.serving.runtime import collate_observations_to_torch_batch


class EnvironmentActionBoundsPolicyAdapter:
    """Remote-policy wrapper that leaves action bounds to eval environments.

    ``praxis_eval`` validates action shape/dtype before stepping an environment.
    The environment still owns clipping or rejecting actions against its own
    runtime bounds, so the wrapper strips min/max bounds before forwarding.
    """

    def __init__(self, policy: Any) -> None:
        self.policy = policy

    def act(
        self,
        observations: Sequence[Observation],
        *,
        action_spec: ActionSpec | None = None,
        policy_kwargs: Mapping[str, Any] | None = None,
        episode_ids: Sequence[str] | None = None,
    ) -> np.ndarray:
        return self.policy.act(
            observations,
            action_spec=_action_spec_without_env_bounds(action_spec),
            policy_kwargs=policy_kwargs,
            episode_ids=episode_ids,
        )

    def reset(self, episode_ids: Sequence[str] | None = None) -> None:
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset(episode_ids=episode_ids)

    def close(self) -> None:
        close = getattr(self.policy, "close", None)
        if callable(close):
            close()


class PolicyEvalAdapter:
    """Adapt a local ``BasePolicy`` to the batched ``praxis_eval`` policy API.

    The adapter owns tensor collation, device placement, optional preprocessing,
    and per-episode rollout state. Policy implementations keep their native
    ``predict_action`` interface.
    """

    def __init__(
        self,
        policy: BasePolicy,
        *,
        device: str | torch.device,
        preprocessor=None,
        inference_trace_path: str | Path | None = None,
        sampling_seed: int = 0,
    ) -> None:
        self.policy = policy
        self.device = torch.device(device)
        self.preprocessor = preprocessor if preprocessor is not None else lambda x: x
        self._inference_info: list[dict[str, Any]] = []
        self._inference_trace_path = (
            Path(inference_trace_path) if inference_trace_path is not None else None
        )
        if self._inference_trace_path is not None:
            self._inference_trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._sampling_seed = int(sampling_seed)
        self._episode_states: dict[str, PolicyRolloutState] = {}
        self._episode_generators: dict[str, torch.Generator] = {}
        self._reset_generation = 0
        self.policy.to(self.device)
        if self.policy.training:
            self.policy.eval()

    def reset(self, episode_ids: Sequence[str] | None = None) -> None:
        self._reset_generation += 1
        if episode_ids is not None:
            for episode_id in episode_ids:
                episode_key = str(episode_id)
                self._episode_states.pop(episode_key, None)
                self._episode_generators.pop(episode_key, None)
            return
        self._episode_states.clear()
        self._episode_generators.clear()
        self.policy.reset()

    @torch.inference_mode()
    def act(
        self,
        observations: Sequence[Observation],
        *,
        action_spec: ActionSpec | None = None,
        policy_kwargs: Mapping[str, Any] | None = None,
        episode_ids: Sequence[str] | None = None,
    ) -> np.ndarray:
        if len(observations) < 1:
            raise ValueError("observations must be non-empty")
        validation_action_spec = _action_spec_without_env_bounds(action_spec)
        if episode_ids is not None:
            action_array = self._predict_episode_scoped(
                observations,
                episode_ids=episode_ids,
                action_spec=validation_action_spec,
                policy_kwargs=policy_kwargs,
            )
            return normalize_batched_action(
                action_array,
                batch_size=len(observations),
                action_spec=validation_action_spec,
            )
        action = self._predict_raw_action(
            observations,
            policy_kwargs=policy_kwargs,
        )
        self._consume_policy_inference_info()
        return normalize_batched_action(
            action.detach().cpu().numpy(),
            batch_size=len(observations),
            action_spec=validation_action_spec,
        )

    def _predict_raw_action(
        self,
        observations: Sequence[Observation],
        *,
        policy_kwargs: Mapping[str, Any] | None,
    ) -> torch.Tensor:
        batch = collate_observations_to_torch_batch(
            observations,
            device=self.device,
        )
        batch = resize_policy_io_images(batch, io=self.policy.io)
        batch = self.preprocessor(batch)
        return self.policy.predict_action(batch, **dict(policy_kwargs or {}))

    def _predict_episode_scoped(
        self,
        observations: Sequence[Observation],
        *,
        episode_ids: Sequence[str],
        action_spec: ActionSpec | None,
        policy_kwargs: Mapping[str, Any] | None,
    ) -> np.ndarray:
        if len(episode_ids) != len(observations):
            raise ValueError(
                "episode_ids must match observations length: "
                f"{len(episode_ids)} != {len(observations)}."
            )
        ids = [str(episode_id) for episode_id in episode_ids]
        if len(set(ids)) != len(ids):
            raise ValueError("episode_ids must be unique within one predict call.")

        ambient_state = self.policy.capture_rollout_state()
        actions: list[np.ndarray] = []
        try:
            for observation, episode_id in zip(observations, ids, strict=True):
                # ``BasePolicy`` rollout state is mutable, so episode-scoped
                # eval temporarily installs one episode state at a time and
                # restores the caller-visible state after the batch.
                self.policy.restore_rollout_state(
                    self._episode_states.get(episode_id)
                    or self.policy.new_rollout_state()
                )
                action = self._predict_raw_action(
                    [observation],
                    policy_kwargs={
                        **dict(policy_kwargs or {}),
                        "generator": self._episode_generator(episode_id),
                    },
                )
                self._consume_policy_inference_info(episode_id=episode_id)
                self._episode_states[episode_id] = self.policy.capture_rollout_state()
                batched = normalize_batched_action(
                    action.detach().cpu().numpy(),
                    batch_size=1,
                    action_spec=action_spec,
                )
                actions.append(np.asarray(batched[0]))
        finally:
            self.policy.restore_rollout_state(ambient_state)

        return np.stack(actions, axis=0)

    def _episode_generator(self, episode_id: str) -> torch.Generator:
        generator = self._episode_generators.get(episode_id)
        if generator is None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(_stable_episode_seed(self._sampling_seed, episode_id))
            self._episode_generators[episode_id] = generator
        return generator

    def _consume_policy_inference_info(self, *, episode_id: str | None = None) -> None:
        consume_info = getattr(self.policy, "consume_inference_info", None)
        if callable(consume_info):
            info = consume_info()
            if isinstance(info, Mapping):
                self._record_external_inference_info(dict(info), episode_id=episode_id)
            elif isinstance(info, Iterable) and not isinstance(info, (str, bytes)):
                for item in info:
                    if isinstance(item, Mapping):
                        self._record_external_inference_info(
                            dict(item), episode_id=episode_id
                        )

    def _record_external_inference_info(
        self,
        info: dict[str, Any],
        *,
        episode_id: str | None = None,
    ) -> None:
        record = dict(info)
        if episode_id is not None:
            record["episode_id"] = episode_id
            record["reset_generation"] = self._reset_generation
        self._inference_info.append(record)
        if self._inference_trace_path is not None:
            self._inference_trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self._inference_trace_path.open("a", encoding="utf-8") as trace_file:
                trace_file.write(json.dumps(record, sort_keys=True) + "\n")

    def consume_inference_info(self) -> list[dict[str, Any]]:
        info = list(self._inference_info)
        self._inference_info.clear()
        return info


def resize_policy_io_images(
    batch: dict[str, Any],
    *,
    io: PolicyIO | None,
) -> dict[str, Any]:
    """Resize image observations to the spatial shapes declared by ``PolicyIO``."""
    if io is None:
        return batch

    out = batch
    for feature in io.image_features:
        value = batch.get(feature.key)
        if not isinstance(value, torch.Tensor):
            continue
        if len(feature.shape) != 3:
            raise ValueError(
                f"PolicyIO image feature {feature.key!r} must have shape (C, H, W), "
                f"got {feature.shape}."
            )
        expected_channels, target_height, target_width = feature.shape
        resized = _resize_image_tensor(
            value,
            key=feature.key,
            expected_channels=int(expected_channels),
            target_size=(int(target_height), int(target_width)),
        )
        if resized is value:
            continue
        if out is batch:
            out = dict(batch)
        out[feature.key] = resized
    return out


def _resize_image_tensor(
    image: torch.Tensor,
    *,
    key: str,
    expected_channels: int,
    target_size: tuple[int, int],
) -> torch.Tensor:
    if image.ndim < 3:
        raise ValueError(
            f"Image observation {key!r} must end with (C, H, W), got "
            f"shape {tuple(image.shape)}."
        )
    if int(image.shape[-3]) != expected_channels:
        raise ValueError(
            f"Image observation {key!r} expected {expected_channels} channels in "
            f"(C, H, W) layout, got shape {tuple(image.shape)}."
        )
    if tuple(int(dim) for dim in image.shape[-2:]) == target_size:
        return image

    leading_shape = tuple(int(dim) for dim in image.shape[:-3])
    flat = image.reshape(-1, expected_channels, *image.shape[-2:])
    original_dtype = flat.dtype
    interpolate_input = flat
    if not interpolate_input.is_floating_point() or (
        interpolate_input.device.type == "cpu"
        and interpolate_input.dtype in {torch.float16, torch.bfloat16}
    ):
        interpolate_input = interpolate_input.to(dtype=torch.float32)

    resized = F.interpolate(
        interpolate_input,
        size=target_size,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    if original_dtype.is_floating_point:
        resized = resized.to(dtype=original_dtype)
    else:
        dtype_info = torch.iinfo(original_dtype)
        resized = (
            resized.round().clamp(dtype_info.min, dtype_info.max).to(original_dtype)
        )
    return resized.reshape(*leading_shape, expected_channels, *target_size)


def _stable_episode_seed(base_seed: int, episode_id: str) -> int:
    payload = f"{int(base_seed)}\0{episode_id}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big") & ((1 << 63) - 1)


def _action_spec_without_env_bounds(
    action_spec: ActionSpec | None,
) -> ActionSpec | None:
    """Preserve action shape/dtype checks while leaving bounds to the env."""
    if action_spec is None:
        return None
    if action_spec.minimum is None and action_spec.maximum is None:
        return action_spec
    return replace(action_spec, minimum=None, maximum=None)
