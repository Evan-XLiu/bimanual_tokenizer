"""Helpers for checkpoint-compatible pretrained model artifacts."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, cast

import torch
from huggingface_hub import constants, load_torch_model
from safetensors.torch import load_model as load_safetensors_model

from praxis.core.checkpoint_artifacts import is_complete_safetensors_dir
from praxis.core.config import config_to_dict


def _coerce_string_key_iterable(value: object, *, label: str) -> list[str]:
    if not isinstance(value, Iterable) or isinstance(value, str | bytes):
        raise TypeError(f"{label} must be an iterable of strings.")
    keys = list(value)
    if not all(isinstance(key, str) for key in keys):
        raise TypeError(f"{label} must contain only strings.")
    return keys


def filter_checkpoint_incompatible_keys(
    model: torch.nn.Module,
    *,
    missing: list[str] | set[str] | tuple[str, ...],
    unexpected: list[str] | set[str] | tuple[str, ...],
) -> tuple[list[str], list[str]]:
    """Let models discard expected save/load key mismatches."""
    filter_keys = getattr(model, "filter_checkpoint_incompatible_keys", None)
    if callable(filter_keys):
        filtered = filter_keys(missing=missing, unexpected=unexpected)
        if not isinstance(filtered, list | tuple) or len(filtered) != 2:
            raise TypeError(
                "filter_checkpoint_incompatible_keys() must return "
                "(missing, unexpected)."
            )
        filtered_missing, filtered_unexpected = filtered
        return (
            _coerce_string_key_iterable(
                filtered_missing,
                label="filtered missing keys",
            ),
            _coerce_string_key_iterable(
                filtered_unexpected,
                label="filtered unexpected keys",
            ),
        )
    return list(missing), list(unexpected)


def filter_model_export_state_dict(
    model: torch.nn.Module,
    state_dict: dict[str, Any],
) -> dict[str, Any]:
    """Apply model-owned checkpoint filtering to an already gathered state dict."""
    filter_state = getattr(model, "filter_checkpoint_export_state_dict", None)
    if callable(filter_state):
        filtered = filter_state(dict(state_dict))
        if not isinstance(filtered, dict):
            raise TypeError("filter_checkpoint_export_state_dict() must return a dict.")
        if not all(isinstance(key, str) for key in filtered):
            raise TypeError(
                "filter_checkpoint_export_state_dict() must return string keys."
            )
        return cast(dict[str, Any], dict(filtered))
    return dict(state_dict)


def serialize_model_config(
    model: torch.nn.Module,
    *,
    serializer: Callable[[torch.nn.Module], dict[str, Any] | None] | None = None,
) -> dict[str, Any] | None:
    """Serialize model config payload for save_pretrained-style exports."""
    if serializer is not None:
        payload = serializer(model)
        if payload is None or isinstance(payload, dict):
            return cast(dict[str, Any] | None, payload)
        raise TypeError("_serialize_model_config() must return a dict or None.")

    config_obj = getattr(model, "config", None)
    return config_to_dict(config_obj)


def save_pretrained_model(
    model: torch.nn.Module,
    model_dir: Path,
    *,
    config: dict[str, Any] | None = None,
    state_dict: dict[str, Any] | None = None,
) -> None:
    """Save an HF-style pretrained model artifact through the model boundary."""
    kwargs: dict[str, Any] = {}
    if config is not None:
        kwargs["config"] = config
    if state_dict is not None:
        kwargs["state_dict"] = state_dict
    save_pretrained = getattr(model, "save_pretrained", None)
    if not callable(save_pretrained):
        raise TypeError(
            f"{model.__class__.__name__} does not implement save_pretrained()."
        )
    save_pretrained(model_dir, **kwargs)


def load_pretrained_model_weights(
    model: torch.nn.Module,
    model_dir: Path,
    *,
    strict: bool = False,
    map_location: str = "cpu",
) -> tuple[list[str], list[str]]:
    """Load single-file or sharded HF-native safetensors into ``model``."""
    index_file = model_dir / constants.SAFETENSORS_INDEX_FILE
    if index_file.exists():
        if not is_complete_safetensors_dir(model_dir):
            raise FileNotFoundError(
                f"Sharded safetensors checkpoint under {model_dir} is incomplete."
            )
        incompatible = cast(
            Any,
            load_torch_model(
                model,
                model_dir,
                strict=strict,
                safe=True,
                map_location=map_location,
            ),
        )
        if str(map_location) != "cpu":
            model.to(map_location)
        missing = _filter_sharded_safetensors_tied_alias_missing_keys(
            model,
            model_dir,
            missing=incompatible.missing_keys,
        )
        return missing, list(incompatible.unexpected_keys)

    single_file = model_dir / constants.SAFETENSORS_SINGLE_FILE
    if single_file.exists():
        return load_safetensors_model(
            model,
            str(single_file),
            strict=strict,
            device=map_location,
        )

    raise FileNotFoundError(
        f"No pretrained safetensors found under {model_dir}. Expected "
        f"{constants.SAFETENSORS_SINGLE_FILE} or {constants.SAFETENSORS_INDEX_FILE}."
    )


def _filter_sharded_safetensors_tied_alias_missing_keys(
    model: torch.nn.Module,
    model_dir: Path,
    *,
    missing: Iterable[str],
) -> list[str]:
    missing_list = list(missing)
    missing_set = set(missing_list)
    index = json.loads((model_dir / constants.SAFETENSORS_INDEX_FILE).read_text())
    metadata = index.get("metadata", {})
    weight_map = index.get("weight_map", {})
    if not isinstance(metadata, dict) or not isinstance(weight_map, dict):
        return missing_list

    loaded_keys = set(weight_map)
    model_state = model.state_dict()
    out: list[str] = []
    for key in missing_list:
        tied_to = metadata.get(key)
        if (
            isinstance(tied_to, str)
            and tied_to in loaded_keys
            and tied_to not in missing_set
            and _state_dict_entries_share_storage(model_state, key, tied_to)
        ):
            continue
        out.append(key)
    return out


def _state_dict_entries_share_storage(
    state_dict: dict[str, torch.Tensor],
    left_key: str,
    right_key: str,
) -> bool:
    left = state_dict.get(left_key)
    right = state_dict.get(right_key)
    if left is None or right is None:
        return False
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.device.type == "meta" or right.device.type == "meta":
        return False
    try:
        left_storage = left.untyped_storage()
        right_storage = right.untyped_storage()
    except Exception:
        return False
    return (
        left_storage.data_ptr() == right_storage.data_ptr()
        and left_storage.nbytes() == right_storage.nbytes()
        and left.storage_offset() == right.storage_offset()
    )
