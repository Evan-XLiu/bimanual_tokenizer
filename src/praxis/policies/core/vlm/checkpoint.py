# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint filtering helpers for VLM-backed policies."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast


def filter_frozen_vlm_state_dict(
    state_dict: dict,
    *,
    trainable_param_names: set[str],
    is_vlm_state_key: Callable[[str], bool],
) -> dict:
    """Drop frozen VLM tensors from a state dict while preserving other entries."""
    out = dict(state_dict)
    for key in list(out.keys()):
        if is_vlm_state_key(key) and key not in trainable_param_names:
            out.pop(key)
    return out


def validate_filtered_load_state(
    incompatible,
    *,
    strict: bool,
    trainable_param_names: set[str],
    is_vlm_state_key: Callable[[str], bool],
    is_tokenizer_state_key: Callable[[str], bool],
    ignore_tokenizer_state_keys: bool,
    owner_name: str,
) -> None:
    """Validate a non-strict load result under filtered checkpoint semantics."""
    if not strict:
        return

    missing = [
        key
        for key in incompatible.missing_keys
        if not (ignore_tokenizer_state_keys and is_tokenizer_state_key(key))
        and not (is_vlm_state_key(key) and key not in trainable_param_names)
    ]
    unexpected = [
        key
        for key in incompatible.unexpected_keys
        if not (ignore_tokenizer_state_keys and is_tokenizer_state_key(key))
    ]
    if missing or unexpected:
        raise RuntimeError(
            f"Error(s) in loading state_dict for {owner_name}: "
            f"missing_keys={missing}, unexpected_keys={unexpected}"
        )


class VlmCheckpointFilterMixin:
    """Checkpoint lifecycle for VLM-backed policies with frozen base weights."""

    config: Any

    def _exclude_frozen_vlm_from_checkpoint(self) -> bool:
        return bool(getattr(self.config, "use_lora", False))

    @staticmethod
    def _is_vlm_state_key(key: str) -> bool:
        return key.startswith("vlm.")

    def _trainable_param_names(self) -> set[str]:
        return {
            name
            for name, param in cast(Any, self).named_parameters()
            if param.requires_grad
        }

    def filter_checkpoint_export_state_dict(self, state_dict: dict) -> dict:
        super_obj = cast(Any, super())
        super_filter = getattr(super_obj, "filter_checkpoint_export_state_dict", None)
        out = (
            cast(dict, super_filter(state_dict))
            if callable(super_filter)
            else dict(state_dict)
        )
        if not self._exclude_frozen_vlm_from_checkpoint():
            return out
        return filter_frozen_vlm_state_dict(
            out,
            trainable_param_names=self._trainable_param_names(),
            is_vlm_state_key=self._is_vlm_state_key,
        )

    def filter_checkpoint_incompatible_keys(
        self,
        *,
        missing: list[str] | set[str] | tuple[str, ...],
        unexpected: list[str] | set[str] | tuple[str, ...],
    ) -> tuple[list[str], list[str]]:
        super_obj = cast(Any, super())
        super_filter = getattr(super_obj, "filter_checkpoint_incompatible_keys", None)
        if callable(super_filter):
            filtered_missing, filtered_unexpected = cast(
                tuple[list[str], list[str]],
                super_filter(
                    missing=missing,
                    unexpected=unexpected,
                ),
            )
        else:
            filtered_missing = list(missing)
            filtered_unexpected = list(unexpected)
        if not self._exclude_frozen_vlm_from_checkpoint():
            return filtered_missing, filtered_unexpected
        trainable = self._trainable_param_names()
        filtered_missing = [
            key
            for key in filtered_missing
            if not (self._is_vlm_state_key(key) and key not in trainable)
        ]
        filtered_unexpected = [
            key
            for key in filtered_unexpected
            if not (self._is_vlm_state_key(key) and key not in trainable)
        ]
        return filtered_missing, filtered_unexpected

    def state_dict(self, *args, **kwargs):
        out = cast(Any, super()).state_dict(*args, **kwargs)
        if not self._exclude_frozen_vlm_from_checkpoint():
            return out
        return filter_frozen_vlm_state_dict(
            out,
            trainable_param_names=self._trainable_param_names(),
            is_vlm_state_key=self._is_vlm_state_key,
        )

    def load_state_dict(
        self,
        state_dict,
        strict: bool = True,
        assign: bool = False,
    ):
        if not self._exclude_frozen_vlm_from_checkpoint():
            return cast(Any, super()).load_state_dict(
                state_dict, strict=strict, assign=assign
            )

        incompatible = cast(Any, super()).load_state_dict(
            state_dict,
            strict=False,
            assign=assign,
        )
        self_any = cast(Any, self)
        validate_filtered_load_state(
            incompatible,
            strict=strict,
            trainable_param_names=self._trainable_param_names(),
            is_vlm_state_key=self._is_vlm_state_key,
            is_tokenizer_state_key=self_any._is_tokenizer_state_key,
            ignore_tokenizer_state_keys=self_any._exclude_frozen_tokenizer_from_checkpoint(),
            owner_name=self.__class__.__name__,
        )
        return incompatible
