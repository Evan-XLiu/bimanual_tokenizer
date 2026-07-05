# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Adapters between Praxis policies and the standalone ``praxis_eval`` API."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import torch
from praxis_eval.contracts import ActionSpec
from praxis_eval.policies.actions import normalize_batched_action
from praxis_eval.types import Observation

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
    ) -> None:
        self.policy = policy
        self.device = torch.device(device)
        self.preprocessor = preprocessor if preprocessor is not None else lambda x: x
        self._inference_info: list[dict[str, Any]] = []
        self._episode_states: dict[str, PolicyRolloutState] = {}
        self.policy.to(self.device)
        if self.policy.training:
            self.policy.eval()

    def reset(self, episode_ids: Sequence[str] | None = None) -> None:
        if episode_ids is not None:
            for episode_id in episode_ids:
                self._episode_states.pop(str(episode_id), None)
            return
        self._episode_states.clear()
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
                    policy_kwargs=policy_kwargs,
                )
                self._consume_policy_inference_info()
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

    def _consume_policy_inference_info(self) -> None:
        consume_info = getattr(self.policy, "consume_inference_info", None)
        if callable(consume_info):
            info = consume_info()
            if isinstance(info, Mapping):
                self._inference_info.append(dict(info))
            elif isinstance(info, Iterable) and not isinstance(info, (str, bytes)):
                for item in info:
                    if isinstance(item, Mapping):
                        self._inference_info.append(dict(item))

    def consume_inference_info(self) -> list[dict[str, Any]]:
        info = list(self._inference_info)
        self._inference_info.clear()
        return info


def _action_spec_without_env_bounds(
    action_spec: ActionSpec | None,
) -> ActionSpec | None:
    """Preserve action shape/dtype checks while leaving bounds to the env."""
    if action_spec is None:
        return None
    if action_spec.minimum is None and action_spec.maximum is None:
        return action_spec
    return replace(action_spec, minimum=None, maximum=None)
