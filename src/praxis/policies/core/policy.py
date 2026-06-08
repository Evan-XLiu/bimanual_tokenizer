# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Abstract base class for all policies."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from lerobot.policies.utils import populate_queues
from torch import Tensor

from praxis.core.normalizer import Normalize, Unnormalize
from praxis.core.types import FeatureSpec, NormMode, PolicyIO
from praxis.policies.core.normalization import (
    collect_action_norm_stats,
    collect_observation_norm_stats,
)
from praxis.policies.core.pretrained_artifacts import (
    export_policy_norm_stats,
    load_policy_from_pretrained,
    save_policy_pretrained,
    save_policy_weights,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PolicyRolloutState:
    """Snapshot of queue-backed rollout state for episode-scoped inference."""

    obs_queues: dict[str, deque[Tensor]]
    action_queue: deque[Tensor]
    pending_inference_info: dict[str, Any] | None


class BasePolicy(nn.Module, PyTorchModelHubMixin, ABC):
    """Base class for all policies.

    Subclasses must implement forward (training) and predict_action (inference).

    Inherits from ``PyTorchModelHubMixin`` which provides:
    - ``save_pretrained(save_directory)`` — saves weights + config.json
    - ``from_pretrained(path_or_hub_id)`` — loads from local dir or HF Hub
    - ``push_to_hub(repo_id)`` — uploads to HF Hub

    **HubMixin contract**: subclass ``__init__`` must accept flat kwargs
    (no complex objects) so they can be serialized to ``config.json``.
    Runtime objects like ``PolicyIO`` should be set after construction
    via ``set_io()``.
    """

    config: Any

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.io: PolicyIO | None = None
        self.normalize: Normalize | None = None
        self.unnormalize_action: Unnormalize | None = None
        self._obs_queues: dict[str, deque[Tensor]] = {}
        self._action_queue: deque[Tensor] = deque()
        self._pending_inference_info: dict[str, Any] | None = None

    def _record_inference_info(self, info: dict[str, Any] | None) -> None:
        """Store one chunk's inference metadata for external rollout collectors."""
        self._pending_inference_info = dict(info) if info else None

    def consume_inference_info(self) -> dict[str, Any] | None:
        """Return and clear inference metadata emitted by the latest chunk decode."""
        info = self._pending_inference_info
        self._pending_inference_info = None
        return info

    def capture_rollout_state(self) -> PolicyRolloutState:
        """Return a copy of queue-backed rollout state."""
        return PolicyRolloutState(
            obs_queues={
                str(key): deque(queue, maxlen=queue.maxlen)
                for key, queue in self._obs_queues.items()
            },
            action_queue=deque(self._action_queue, maxlen=self._action_queue.maxlen),
            pending_inference_info=(
                dict(self._pending_inference_info)
                if self._pending_inference_info is not None
                else None
            ),
        )

    def new_rollout_state(self) -> PolicyRolloutState:
        """Return an empty rollout state matching this policy's queue shape."""
        return PolicyRolloutState(
            obs_queues={},
            action_queue=deque(maxlen=self._action_queue.maxlen),
            pending_inference_info=None,
        )

    def restore_rollout_state(self, state: PolicyRolloutState) -> None:
        """Restore queue-backed rollout state from ``capture_rollout_state``."""
        self._obs_queues = {
            str(key): deque(queue, maxlen=queue.maxlen)
            for key, queue in state.obs_queues.items()
        }
        self._action_queue = deque(
            state.action_queue,
            maxlen=state.action_queue.maxlen,
        )
        self._pending_inference_info = (
            dict(state.pending_inference_info)
            if state.pending_inference_info is not None
            else None
        )

    def set_io(self, io: PolicyIO) -> None:
        """Attach runtime PolicyIO (obs/action specs from dataset metadata)."""
        self.io = io

    def refresh_after_config_update(self, updated_keys: frozenset[str]) -> None:
        """Refresh runtime state after checkpoint config overrides are applied."""
        del updated_keys
        if self.io is not None:
            self.set_io(self.io)

    @staticmethod
    def _resolve_action_dim_for_from_io(
        *,
        configured_action_dim: Any,
        io: PolicyIO,
        owner: str,
    ) -> int:
        io_action_dim = int(io.action_dim)
        if configured_action_dim is None:
            return io_action_dim
        resolved = int(configured_action_dim)
        if resolved != io_action_dim:
            raise ValueError(
                f"{owner} action_dim mismatch: configured action_dim={resolved}, "
                f"dataset action_dim={io_action_dim}."
            )
        return resolved

    def save_pretrained(
        self,
        save_directory,
        *,
        state_dict: dict[str, Any] | None = None,
        max_shard_size: int | str | None = None,
        config: Any | None = None,
        repo_id: str | None = None,
        push_to_hub: bool = False,
        model_card_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Save weights/config plus PolicyIO and norm sidecars."""
        return save_policy_pretrained(
            policy=self,
            save_model=super().save_pretrained,
            save_directory=save_directory,
            state_dict=state_dict,
            max_shard_size=max_shard_size,
            config=config,
            repo_id=repo_id,
            push_to_hub=push_to_hub,
            model_card_kwargs=model_card_kwargs,
            extra_kwargs=kwargs,
        )

    def _save_pretrained(self, save_directory: Path) -> None:
        save_policy_weights(self, save_directory)

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str | Path,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        force_download: bool = False,
        proxies: dict | None = None,
        resume_download: bool | None = None,
        local_files_only: bool = False,
        token: str | bool | None = None,
        map_location: str = "cpu",
        strict: bool = False,
        **model_kwargs,
    ):
        """Load weights/config and required sidecars from local dir or HF repo id."""
        return load_policy_from_pretrained(
            policy_cls=cls,
            model_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
            map_location=map_location,
            strict=strict,
            model_kwargs=model_kwargs,
        )

    def _export_norm_stats_for_checkpoint(
        self,
    ) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, str]]:
        """Return checkpoint-owned norm stats/modes for this policy."""
        return export_policy_norm_stats(self)

    @abstractmethod
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        """Training forward pass.

        Args:
            batch: Dict of tensors from the DataLoader.

        Returns:
            (loss, info_dict) where info_dict contains optional extra metrics.
        """

    @abstractmethod
    def predict_action(self, obs: dict[str, Tensor], **kwargs) -> Tensor:
        """Inference: predict action from observations.

        Handles observation history and action chunking internally.

        Args:
            obs: Dict of observation tensors (unbatched or batch=1).

        Returns:
            Single action tensor of shape (action_dim,).
        """

    def predict_action_chunk(
        self, obs: dict[str, Tensor], **kwargs
    ) -> tuple[Tensor, dict[str, Any]]:
        """Inference: predict an action chunk from stacked observations.

        Args:
            obs: Dict of observation tensors, typically shaped ``(B, T_obs, ...)``.

        Returns:
            Tuple ``(action_chunk, info)`` where:
            - ``action_chunk`` is typically shaped ``(B, horizon, action_dim)``
            - ``info`` contains optional inference diagnostics.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement predict_action_chunk()."
        )

    def _configure_action_queue(self, *, n_action_steps: int) -> None:
        """Configure the rollout action FIFO for chunked-action policies."""
        self._action_queue = deque(maxlen=int(n_action_steps))

    def _predict_action_from_chunk_queue(
        self,
        obs: dict[str, Tensor],
        *,
        predict_action_chunk: Callable[..., tuple[Tensor, dict[str, Any]]],
        postprocess_action: Callable[[Tensor], Tensor] | None = None,
        missing_history_error: Callable[[list[str], dict[str, Tensor]], str]
        | None = None,
        preserve_language_key: bool = True,
        **kwargs,
    ) -> Tensor:
        """Serve one action from a cached action chunk.

        This is the generic rollout path for policies that:
        1. keep per-feature observation history,
        2. predict a full action chunk from stacked observations, and
        3. expose one action at a time to the caller.

        Family-specific behavior stays in the injected callbacks:
        ``predict_action_chunk`` produces the chunk, ``postprocess_action``
        can apply per-family output transforms, and ``missing_history_error``
        can customize the user-facing error message.
        """
        if self.io is None:
            raise RuntimeError(
                "PolicyIO is not set. Call set_io(...) before inference."
            )

        obs = self._prepare_current_observations(
            obs,
            normalize=self.normalize is not None,
            require_features=False,
        )

        for feat in self.io.obs_features:
            key = feat.key
            if key not in self._obs_queues:
                self._obs_queues[key] = deque(maxlen=self.config.n_obs_steps)
        self._obs_queues = populate_queues(self._obs_queues, obs)

        if len(self._action_queue) == 0:
            missing = [
                feat.key
                for feat in self.io.obs_features
                if len(self._obs_queues.get(feat.key, ())) == 0
            ]
            if missing:
                if missing_history_error is not None:
                    raise RuntimeError(missing_history_error(missing, obs))
                detail = self._format_missing_observation_keys_error(missing, obs)
                raise RuntimeError(
                    f"Missing required observation history for keys {missing}. {detail}"
                )

            stacked_obs = {
                feat.key: torch.stack(list(self._obs_queues[feat.key]), dim=1)
                for feat in self.io.obs_features
            }
            if (
                preserve_language_key
                and self.io.language_key is not None
                and self.io.language_key in obs
            ):
                stacked_obs[self.io.language_key] = obs[self.io.language_key]

            chunk_result = predict_action_chunk(stacked_obs, **kwargs)
            if not isinstance(chunk_result, tuple) or len(chunk_result) != 2:
                raise TypeError(
                    "predict_action_chunk must return (actions, info). Return "
                    "an empty dict when no inference info is available."
                )
            actions, inference_info = chunk_result
            self._record_inference_info(inference_info)
            if actions.dim() < 3:
                raise RuntimeError(
                    f"Expected chunked actions with shape (B, T, D), got {tuple(actions.shape)}."
                )
            n_chunk_steps = min(self.config.n_action_steps, actions.shape[1])
            if n_chunk_steps <= 0:
                raise RuntimeError("Model returned an empty action chunk.")
            for i in range(n_chunk_steps):
                self._action_queue.append(actions[:, i])

        action = self._action_queue.popleft()
        if postprocess_action is not None:
            action = postprocess_action(action)
        return action.squeeze(0)

    def reset(self) -> None:
        """Reset internal state at episode boundaries (e.g., action queues)."""
        self._obs_queues.clear()
        self._action_queue.clear()
        self._pending_inference_info = None

    def get_optim_params(self) -> list[dict]:
        """Return parameter groups for the optimizer.

        Override to set per-group learning rates, weight decay, etc.
        """
        return [{"params": self.parameters()}]

    def set_normalization_stats(
        self,
        stats: dict[str, dict[str, Tensor]],
        modes: dict[str, NormMode] | None = None,
    ) -> None:
        """Load normalization statistics into the policy's normalizer buffers.

        Args:
            stats: {feature_key: {"min", "max", "mean", "std"} -> Tensor}.
            modes: optional per-key normalization mode map.
        """
        self.normalize = None
        self.unnormalize_action = None
        if self.io is None:
            return

        relevant_stats, resolved_modes = self._collect_observation_norm_stats(
            stats,
            feature_modes=modes,
        )
        action_stats_for_unnorm, action_mode_for_unnorm = (
            self._collect_action_norm_stats(
                stats=stats,
                feature_modes=modes,
            )
        )
        if self.io is not None and self.io.action_feature is not None:
            action_key = self.io.action_feature.key
            if (
                action_stats_for_unnorm is not None
                and action_mode_for_unnorm is not None
            ):
                relevant_stats[action_key] = action_stats_for_unnorm
                resolved_modes[action_key] = action_mode_for_unnorm

        self.normalize = (
            Normalize(relevant_stats, resolved_modes) if resolved_modes else None
        )

        self.unnormalize_action = None
        if (
            self.io.action_feature is not None
            and action_stats_for_unnorm is not None
            and action_mode_for_unnorm is not None
        ):
            action_key = self.io.action_feature.key
            self.unnormalize_action = Unnormalize(
                {action_key: action_stats_for_unnorm},
                {action_key: action_mode_for_unnorm},
            )

    def _collect_observation_norm_stats(
        self,
        stats: dict[str, dict[str, Tensor]],
        *,
        feature_modes: dict[str, NormMode] | None = None,
        features: list[FeatureSpec] | None = None,
        use_image_defaults: bool = True,
        missing_stats_warning: str = (
            "Missing normalization stats for %s; skipping normalization for this key."
        ),
    ) -> tuple[dict[str, dict[str, Tensor]], dict[str, NormMode]]:
        return collect_observation_norm_stats(
            io=self.io,
            stats=stats,
            feature_modes=feature_modes,
            features=features,
            use_image_defaults=use_image_defaults,
            missing_stats_warning=missing_stats_warning,
        )

    def _collect_action_norm_stats(
        self,
        *,
        stats: dict[str, dict[str, Tensor]],
        feature_modes: dict[str, NormMode] | None = None,
    ) -> tuple[dict[str, Tensor] | None, NormMode | None]:
        return collect_action_norm_stats(
            io=self.io,
            stats=stats,
            feature_modes=feature_modes,
        )

    def _prepare_chunk_observations(
        self,
        obs: dict[str, Tensor],
        *,
        normalize: bool,
        missing_key_error_template: str = (
            "Missing required observation key for predict_action_chunk: {key}"
        ),
    ) -> dict[str, Tensor]:
        """Canonicalize chunk observations to ``(B, T, ...)`` and optionally normalize."""
        if self.io is None:
            raise RuntimeError(
                "PolicyIO is not set. Call set_io(...) before inference."
            )
        out = dict(obs)

        for feat in self.io.obs_features:
            key = feat.key
            if key not in out:
                raise RuntimeError(
                    self._format_missing_observation_key_error(
                        key=key,
                        obs=out,
                        base_message=missing_key_error_template.format(key=key),
                    )
                )
            value = out[key]
            if not isinstance(value, Tensor):
                continue
            expected_without_time = len(feat.shape) + 1
            expected_with_time = expected_without_time + 1
            if value.dim() == len(feat.shape):
                value = value.unsqueeze(0)
            if value.dim() == expected_without_time:
                value = value.unsqueeze(1)
            elif value.dim() != expected_with_time:
                raise RuntimeError(
                    f"Unexpected observation shape for {key}: got {tuple(value.shape)}"
                )
            out[key] = value

        if normalize and self.normalize is not None:
            out = self.normalize(out)
        return out

    def _prepare_current_observations(
        self,
        obs: dict[str, Tensor],
        *,
        normalize: bool,
        require_features: bool,
        missing_key_error_template: str = (
            "Missing required observation key for predict_action: {key}"
        ),
    ) -> dict[str, Tensor]:
        """Canonicalize current-step observations to ``(B, ...)`` by PolicyIO."""
        if self.io is None:
            raise RuntimeError(
                "PolicyIO is not set. Call set_io(...) before inference."
            )
        out = dict(obs)

        for feat in self.io.obs_features:
            key = feat.key
            if key not in out:
                if require_features:
                    raise RuntimeError(
                        self._format_missing_observation_key_error(
                            key=key,
                            obs=out,
                            base_message=missing_key_error_template.format(key=key),
                        )
                    )
                continue
            value = out[key]
            if not isinstance(value, Tensor):
                continue
            expected_without_batch = len(feat.shape)
            expected_with_batch = expected_without_batch + 1
            if value.dim() == expected_without_batch:
                value = value.unsqueeze(0)
            elif value.dim() != expected_with_batch:
                raise RuntimeError(
                    f"Unexpected observation shape for {key}: got {tuple(value.shape)}"
                )
            out[key] = value

        if normalize and self.normalize is not None:
            out = self.normalize(out)
        return out

    def _format_missing_observation_key_error(
        self,
        *,
        key: str,
        obs: dict[str, Any],
        base_message: str,
    ) -> str:
        if not key.startswith("observation.images."):
            return base_message

        available_image_keys = [
            obs_key
            for obs_key in obs
            if isinstance(obs_key, str) and obs_key.startswith("observation.images.")
        ]
        if not available_image_keys:
            return base_message

        available = ", ".join(sorted(available_image_keys))
        hint = (
            f"{base_message}. Available image keys: [{available}]. "
            "Observation-key aliasing has been removed; env image keys must match PolicyIO exactly."
        )
        hint += (
            " If env image names differ from dataset image names, align the env-side "
            "camera/image naming to the PolicyIO contract before inference."
        )
        return hint

    def _format_missing_observation_keys_error(
        self,
        missing_keys: list[str],
        obs: dict[str, Any],
    ) -> str:
        image_missing = [
            key
            for key in missing_keys
            if isinstance(key, str) and key.startswith("observation.images.")
        ]
        if len(image_missing) == 1:
            return self._format_missing_observation_key_error(
                key=image_missing[0],
                obs=obs,
                base_message=f"Missing required observation key: {image_missing[0]}",
            )
        return "Provide all observation keys required by PolicyIO."
