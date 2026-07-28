# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Base class for action codecs."""

from __future__ import annotations

import json
from abc import ABC
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor

from praxis.core.checkpoint_artifacts import (
    NORM_STATS_FILENAME,
    POLICY_IO_FILENAME,
    HubDownloadOptions,
    collect_normalizer_stats,
    normalize_push_to_hub_kwargs,
    push_saved_pretrained_dir,
    resolve_sidecar_path,
)
from praxis.core.config import config_to_dict
from praxis.core.normalizer import Normalize, Unnormalize
from praxis.core.types import CANONICAL_ACTION_KEY, NormMode, PolicyIO


class BaseCodec(nn.Module, PyTorchModelHubMixin, ABC):
    """Base class for action codecs.

    A codec maps between normalized continuous action chunks and another action
    representation, for example discrete tokens or continuous latents.  This
    base owns the shared checkpoint, PolicyIO, action-key, normalization
    lifecycle, and representation-sequence metadata that applies to both
    tokens and continuous latents. Representation-specific contracts belong in
    narrower bases such as ``BaseTokenizer``.

    ``PyTorchModelHubMixin`` requires subclass ``__init__`` methods to accept
    flat serializable kwargs so they can be saved to ``config.json``.
    """

    def __init__(self, **kwargs: Any) -> None:
        del kwargs
        super().__init__()
        self.io: PolicyIO | None = None
        self.action_key = CANONICAL_ACTION_KEY
        self.normalize_action: Normalize | None = None
        self.unnormalize_action: Unnormalize | None = None

    @property
    def variable_length_codes(self) -> bool:
        """Whether encoding emits variable-length code sequences."""
        return False

    @property
    def max_code_seq_len(self) -> int | None:
        """Maximum code-sequence length this codec can emit/consume.

        ``None`` means the codec does not declare a bounded code horizon.
        Fixed-length codecs should return their exact code horizon here.
        Variable-length codecs can still expose an explicit decode budget.
        """
        return None

    @property
    def supports_prefix_decode(self) -> bool:
        """Whether decode can consume a code prefix shorter than the horizon."""
        return False

    def save_pretrained(
        self,
        save_directory,
        *,
        config: Any | None = None,
        repo_id: str | None = None,
        push_to_hub: bool = False,
        model_card_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Save codec weights/config plus PolicyIO and norm sidecars."""
        if self.io is None:
            raise RuntimeError(
                f"{self.__class__.__name__} has no PolicyIO. Call set_io(...) before save_pretrained()."
            )
        push_kwargs = normalize_push_to_hub_kwargs(kwargs) if push_to_hub else None
        if config is None:
            config = config_to_dict(getattr(self, "config", None))

        super().save_pretrained(
            save_directory,
            config=config,
            repo_id=repo_id,
            push_to_hub=False,
            model_card_kwargs=model_card_kwargs,
            **kwargs,
        )
        save_dir = Path(save_directory)
        (save_dir / POLICY_IO_FILENAME).write_text(
            json.dumps(self.io.to_dict(), indent=2) + "\n"
        )
        torch.save(
            self._normalization_checkpoint_payload(),
            save_dir / NORM_STATS_FILENAME,
        )
        if push_kwargs is not None:
            return push_saved_pretrained_dir(
                save_dir=save_dir,
                repo_id=repo_id,
                push_kwargs=push_kwargs,
            )
        return None

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
        **model_kwargs,
    ):
        """Load codec and required sidecars from local dir or HF repo id."""
        hub_options = HubDownloadOptions(
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
        )
        model = super()._from_pretrained(
            model_id=str(model_id),
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
            **model_kwargs,
        )

        io_path = resolve_sidecar_path(
            model_id,
            POLICY_IO_FILENAME,
            hub_options=hub_options,
        )
        if io_path is None:
            raise FileNotFoundError(
                f"Codec checkpoint at {model_id} is missing {POLICY_IO_FILENAME}."
            )
        model.set_io(PolicyIO.from_dict(json.loads(io_path.read_text())))

        norm_path = resolve_sidecar_path(
            model_id,
            NORM_STATS_FILENAME,
            hub_options=hub_options,
        )
        if norm_path is None:
            raise FileNotFoundError(
                f"Codec checkpoint at {model_id} is missing {NORM_STATS_FILENAME}."
            )
        data = torch.load(norm_path, map_location="cpu", weights_only=True)
        model._load_normalization_checkpoint_payload(data)
        return model

    def _normalization_checkpoint_payload(self) -> dict[str, Any]:
        """Return the existing generic normalization sidecar payload."""
        stats, modes = self._export_norm_stats_for_checkpoint()
        return {"stats": stats, "modes": modes}

    def _load_normalization_checkpoint_payload(self, data: dict[str, Any]) -> None:
        """Restore the existing generic normalization sidecar payload."""
        stats = data.get("stats", {})
        modes = {key: NormMode(value) for key, value in data.get("modes", {}).items()}
        self.set_normalization_stats(stats, modes=modes)

    def _export_norm_stats_for_checkpoint(
        self,
    ) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, str]]:
        """Return checkpoint-owned norm stats/modes for this codec."""
        stats: dict[str, dict[str, torch.Tensor]] = {}
        modes: dict[str, str] = {}
        collect_normalizer_stats(self.normalize_action, stats, modes)
        return stats, modes

    def set_io(self, io: PolicyIO) -> None:
        self.io = io
        old_key = self.action_key
        if io.action_feature is not None:
            self.action_key = io.action_feature.key
        if old_key == self.action_key:
            return
        if self.normalize_action is None:
            return
        old_stats = self.normalize_action.stats
        old_modes = self.normalize_action.modes
        source_key = (
            old_key if old_key in old_stats else next(iter(old_stats.keys()), None)
        )
        if source_key is None:
            return
        mode_map = (
            {self.action_key: old_modes[source_key]}
            if source_key in old_modes
            else None
        )
        self.set_normalization_stats(
            {self.action_key: old_stats[source_key]}, modes=mode_map
        )

    def _normalize_action_tensor(self, actions: Tensor) -> Tensor:
        if self.normalize_action is None:
            return actions
        return self.normalize_action({self.action_key: actions})[self.action_key]

    def _unnormalize_action_tensor(self, actions: Tensor) -> Tensor:
        if self.unnormalize_action is None:
            return actions
        return self.unnormalize_action({self.action_key: actions})[self.action_key]

    def get_optim_params(self) -> list[dict]:
        """Return optimizer parameter groups.

        Non-learning codecs can use the default empty list. Learning-based
        codecs can override for custom parameter groups.
        """
        params = [p for p in self.parameters() if p.requires_grad]
        if not params:
            return []
        return [{"params": params}]

    def set_normalization_stats(
        self,
        stats: dict[str, dict[str, Tensor]],
        modes: dict[str, NormMode] | None = None,
    ) -> None:
        """Load action normalization statistics into normalizer buffers."""
        source_key: str | None = None
        action_stats = stats.get(self.action_key)
        if action_stats is not None:
            source_key = self.action_key
        elif len(stats) == 1:
            source_key = next(iter(stats.keys()))
            action_stats = stats[source_key]

        if action_stats is None:
            self.normalize_action = None
            self.unnormalize_action = None
            return

        action_stats = {
            k: torch.as_tensor(v, dtype=torch.float32) for k, v in action_stats.items()
        }

        has_minmax = "min" in action_stats and "max" in action_stats
        has_meanstd = "mean" in action_stats and "std" in action_stats

        preferred_mode: NormMode | None = None
        if modes is not None:
            preferred_mode = modes.get(self.action_key)
            if preferred_mode is None and source_key is not None:
                preferred_mode = modes.get(source_key)
        if (
            preferred_mode is None
            and self.io is not None
            and self.io.action_feature is not None
        ):
            preferred_mode = self.io.action_feature.norm_mode
        if preferred_mode is None:
            preferred_mode = NormMode.MIN_MAX

        if preferred_mode == NormMode.NONE:
            self.normalize_action = None
            self.unnormalize_action = None
            return

        mode: NormMode | None
        if preferred_mode == NormMode.MIN_MAX and has_minmax:
            mode = NormMode.MIN_MAX
        elif preferred_mode == NormMode.MEAN_STD and has_meanstd:
            mode = NormMode.MEAN_STD
        elif has_minmax:
            mode = NormMode.MIN_MAX
        elif has_meanstd:
            mode = NormMode.MEAN_STD
        else:
            self.normalize_action = None
            self.unnormalize_action = None
            return

        self.normalize_action = Normalize(
            {self.action_key: action_stats}, {self.action_key: mode}
        )
        self.unnormalize_action = Unnormalize(
            {self.action_key: action_stats}, {self.action_key: mode}
        )
