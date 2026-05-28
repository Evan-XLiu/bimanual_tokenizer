"""Pretrained artifact helpers for policy save/load."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import load_torch_model, save_torch_model, save_torch_state_dict

from praxis.core.checkpoint_artifacts import (
    NORM_STATS_FILENAME,
    POLICY_IO_FILENAME,
    HubDownloadOptions,
    collect_normalizer_stats,
    normalize_push_to_hub_kwargs,
    push_saved_pretrained_dir,
    resolve_pretrained_weight_file,
    resolve_sidecar_path,
)
from praxis.core.types import NormMode, PolicyIO

_STATE_DICT_EXPORT_ATTR = "_praxis_save_pretrained_state_dict"
_MAX_SHARD_SIZE_EXPORT_ATTR = "_praxis_save_pretrained_max_shard_size"
_STATE_DICT_EXPORT_SENTINEL = object()


def save_policy_pretrained(
    *,
    policy: Any,
    save_model: Callable[..., Any],
    save_directory: str | Path,
    state_dict: dict[str, Any] | None,
    max_shard_size: int | str | None,
    config: Any | None,
    repo_id: str | None,
    push_to_hub: bool,
    model_card_kwargs: dict[str, Any] | None,
    extra_kwargs: dict[str, Any],
) -> Any:
    """Save policy weights/config plus Praxis sidecars and optional Hub upload."""
    if getattr(policy, "io", None) is None:
        raise RuntimeError(
            f"{policy.__class__.__name__} has no PolicyIO. Call set_io(...) before save_pretrained()."
        )
    push_kwargs = normalize_push_to_hub_kwargs(extra_kwargs) if push_to_hub else None

    with temporary_save_pretrained_options(
        policy,
        state_dict=state_dict,
        max_shard_size=max_shard_size,
    ):
        save_model(
            save_directory,
            config=config,
            repo_id=repo_id,
            push_to_hub=False,
            model_card_kwargs=model_card_kwargs,
            **extra_kwargs,
        )

    save_dir = Path(save_directory)
    write_policy_pretrained_sidecars(policy, save_dir)
    if push_kwargs is not None:
        return push_saved_pretrained_dir(
            save_dir=save_dir,
            repo_id=repo_id,
            push_kwargs=push_kwargs,
        )
    return None


@contextmanager
def temporary_save_pretrained_options(
    policy: Any,
    *,
    state_dict: dict[str, Any] | None,
    max_shard_size: int | str | None,
) -> Iterator[None]:
    """Expose Praxis-only save options to the HF mixin callback."""
    previous_state_dict = getattr(
        policy,
        _STATE_DICT_EXPORT_ATTR,
        _STATE_DICT_EXPORT_SENTINEL,
    )
    previous_max_shard_size = getattr(
        policy,
        _MAX_SHARD_SIZE_EXPORT_ATTR,
        _STATE_DICT_EXPORT_SENTINEL,
    )
    if state_dict is not None:
        setattr(policy, _STATE_DICT_EXPORT_ATTR, state_dict)
    if max_shard_size is not None:
        setattr(policy, _MAX_SHARD_SIZE_EXPORT_ATTR, max_shard_size)
    try:
        yield
    finally:
        restore_temporary_save_pretrained_attr(
            policy,
            _STATE_DICT_EXPORT_ATTR,
            previous_state_dict,
            enabled=state_dict is not None,
        )
        restore_temporary_save_pretrained_attr(
            policy,
            _MAX_SHARD_SIZE_EXPORT_ATTR,
            previous_max_shard_size,
            enabled=max_shard_size is not None,
        )


def restore_temporary_save_pretrained_attr(
    policy: Any,
    name: str,
    previous_value: object,
    *,
    enabled: bool,
) -> None:
    if not enabled:
        return
    if previous_value is _STATE_DICT_EXPORT_SENTINEL:
        delattr(policy, name)
    else:
        setattr(policy, name, previous_value)


def write_policy_pretrained_sidecars(policy: Any, save_dir: Path) -> None:
    """Write Praxis policy sidecars next to HF-native model artifacts."""
    io = getattr(policy, "io", None)
    assert io is not None
    (save_dir / POLICY_IO_FILENAME).write_text(
        json.dumps(io.to_dict(), indent=2) + "\n"
    )
    stats, modes = export_policy_norm_stats(policy)
    torch.save({"stats": stats, "modes": modes}, save_dir / NORM_STATS_FILENAME)


def save_policy_weights(policy: Any, save_directory: Path) -> None:
    """HF mixin callback that writes policy weights, with optional Praxis overrides."""
    state_dict = getattr(policy, _STATE_DICT_EXPORT_ATTR, None)
    max_shard_size = getattr(policy, _MAX_SHARD_SIZE_EXPORT_ATTR, "15GB")
    if state_dict is None:
        save_torch_model(
            policy,
            save_directory,
            max_shard_size=max_shard_size,
        )
        return
    save_torch_state_dict(
        state_dict,
        save_directory,
        max_shard_size=max_shard_size,
    )


def load_policy_from_pretrained(
    *,
    policy_cls: type,
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
    model_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Load a policy with checkpoint PolicyIO and normalization sidecars."""
    hub_options = HubDownloadOptions(
        revision=revision,
        cache_dir=cache_dir,
        force_download=force_download,
        proxies=proxies,
        resume_download=resume_download,
        local_files_only=local_files_only,
        token=token,
    )

    # Some policies create trainable modules from finalized PolicyIO. Attach IO
    # before loading weights so those modules exist when state_dict loads.
    model = policy_cls(**(model_kwargs or {}))
    model.set_io(load_policy_checkpoint_io(model_id, hub_options=hub_options))
    model = load_policy_weights(
        policy_cls=policy_cls,
        model=model,
        model_id=model_id,
        hub_options=hub_options,
        map_location=map_location,
        strict=strict,
    )

    stats, modes = load_policy_checkpoint_norm_stats(
        model_id,
        hub_options=hub_options,
    )
    model.set_normalization_stats(stats, modes=modes)
    return model


def load_policy_checkpoint_io(
    model_id: str | Path,
    *,
    hub_options: HubDownloadOptions,
) -> PolicyIO:
    """Load the checkpoint PolicyIO sidecar before weight materialization."""
    io_path = resolve_sidecar_path(
        model_id,
        POLICY_IO_FILENAME,
        hub_options=hub_options,
    )
    if io_path is None:
        raise FileNotFoundError(
            f"Policy checkpoint at {model_id} is missing {POLICY_IO_FILENAME}."
        )
    return PolicyIO.from_dict(json.loads(io_path.read_text()))


def load_policy_checkpoint_norm_stats(
    model_id: str | Path,
    *,
    hub_options: HubDownloadOptions,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, NormMode]]:
    """Load checkpoint-owned normalization stats and modes."""
    norm_path = resolve_sidecar_path(
        model_id,
        NORM_STATS_FILENAME,
        hub_options=hub_options,
    )
    if norm_path is None:
        raise FileNotFoundError(
            f"Policy checkpoint at {model_id} is missing {NORM_STATS_FILENAME}."
        )
    norm_data = torch.load(norm_path, map_location="cpu", weights_only=True)
    stats = norm_data.get("stats", {})
    modes = {k: NormMode(v) for k, v in norm_data.get("modes", {}).items()}
    return stats, modes


def load_policy_weights(
    *,
    policy_cls: type,
    model: Any,
    model_id: str | Path,
    hub_options: HubDownloadOptions,
    map_location: str,
    strict: bool,
) -> Any:
    """Load checkpoint weights into a policy whose runtime IO is already set."""
    weight_file = resolve_pretrained_weight_file(
        model_id,
        hub_options=hub_options,
    )
    if weight_file.format == "safetensors":
        if weight_file.path.is_dir():
            load_torch_model(
                model,
                weight_file.path,
                strict=strict,
                safe=True,
                map_location=map_location,
            )
            if str(map_location) != "cpu":
                model.to(map_location)
            return model
        return policy_cls._load_as_safetensor(
            model,
            str(weight_file.path),
            map_location,
            strict,
        )
    return policy_cls._load_as_pickle(
        model, str(weight_file.path), map_location, strict
    )


def export_policy_norm_stats(
    policy: Any,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, str]]:
    """Return checkpoint-owned norm stats/modes for a policy."""
    stats: dict[str, dict[str, torch.Tensor]] = {}
    modes: dict[str, str] = {}
    collect_normalizer_stats(getattr(policy, "normalize", None), stats, modes)
    collect_normalizer_stats(getattr(policy, "unnormalize_action", None), stats, modes)
    return stats, modes
