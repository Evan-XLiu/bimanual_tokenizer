# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Policy factory: instantiate policies from config with optional pretrained loading."""

from __future__ import annotations

import importlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from praxis.core.checkpoint_artifacts import (
    NORM_STATS_FILENAME,
    POLICY_IO_FILENAME,
    resolve_sidecar_path,
)
from praxis.core.types import CANONICAL_ACTION_KEY, PolicyIO
from praxis.policies import catalog as policy_catalog
from praxis.policies.core import BasePolicy
from praxis.policies.core.io_metadata import make_io_from_metadata
from praxis.policies.core.runtime_config_overrides import (
    apply_pretrained_config_overrides,
)

__all__ = [
    "available_policies",
    "get_policy_class",
    "make_io_from_metadata",
    "make_policy",
    "register_policy",
]

_POLICY_REGISTRY: dict[str, str] = {
    spec.name: spec.import_path for spec in policy_catalog.iter_policy_specs()
}


def register_policy(name: str, import_path: str) -> None:
    """Register a policy class import path.

    Args:
        name: Registry key used by ``cfg.policy.name``.
        import_path: ``"module.path:ClassName"``.
    """
    key = name.strip().lower()
    if ":" not in import_path:
        raise ValueError(
            f"Invalid import path {import_path!r}. Expected 'module.path:ClassName'."
        )
    _POLICY_REGISTRY[key] = import_path


def available_policies() -> tuple[str, ...]:
    """Return registered policy names."""
    return tuple(sorted(_POLICY_REGISTRY))


def get_policy_class(name: str) -> type[BasePolicy]:
    """Resolve policy class from the factory registry."""
    key = name.strip().lower()
    if key not in _POLICY_REGISTRY:
        raise ValueError(
            f"Unknown policy: {name!r}. Available: {', '.join(available_policies())}"
        )

    module_path, class_name = _POLICY_REGISTRY[key].split(":", 1)
    module = importlib.import_module(module_path)
    policy_cls = getattr(module, class_name, None)
    if policy_cls is None:
        raise ImportError(f"Could not import {class_name!r} from {module_path!r}.")
    if not isinstance(policy_cls, type) or not issubclass(policy_cls, BasePolicy):
        raise TypeError(
            f"Registered policy {name!r} resolved to {policy_cls!r}, expected BasePolicy subclass."
        )
    return policy_cls


def make_policy(policy_cfg: DictConfig, io: PolicyIO | None = None) -> BasePolicy:
    """Instantiate a policy from config.

    Supports two modes:
    1. **From scratch**: builds policy from config + io dimensions.
    2. **Pretrained**: loads weights, PolicyIO, and norm stats from checkpoint.
       The checkpoint MUST contain ``policy_io.json`` and ``norm_stats.pt``.

    Args:
        policy_cfg: Hydra policy config (e.g. ``cfg.policy``).
        io: PolicyIO — required for from-scratch, ignored for pretrained
            (loaded from checkpoint sidecar).

    Returns:
        Initialized BasePolicy subclass with IO and norm stats set.
    """
    pretrained_path = policy_cfg.pretrained_path
    if pretrained_path:
        return _make_policy_from_pretrained(policy_cfg, pretrained_path, io)
    return _make_policy_from_scratch(policy_cfg, io)


def _make_policy_from_pretrained(
    policy_cfg: DictConfig, pretrained_path: str, io: PolicyIO | None
) -> BasePolicy:
    """Load a pretrained policy and apply explicit config overrides when requested."""
    policy = _load_pretrained_policy(policy_cfg.name, pretrained_path)
    _validate_pretrained_action_dim(policy, io)
    cfg_dict = _policy_cfg_to_dict(policy_cfg)
    applied_override = apply_pretrained_config_overrides(
        policy, cfg_dict, explicit_override_keys=_get_explicit_policy_override_keys()
    )
    if applied_override:
        policy.refresh_after_config_update(applied_override)
    return policy


def _make_policy_from_scratch(
    policy_cfg: DictConfig, io: PolicyIO | None
) -> BasePolicy:
    """Construct a fresh policy from the current Hydra policy config."""
    if io is None:
        raise ValueError("PolicyIO is required when building a policy from scratch.")

    cfg_dict = _policy_cfg_to_dict(policy_cfg)
    validate_tokenizer_cfg(policy_name=policy_cfg.name, cfg_dict=cfg_dict)
    policy_cls = get_policy_class(policy_cfg.name)
    cfg_dict.pop("pretrained_path", None)
    return cast(Any, policy_cls).from_io(io, **cfg_dict)


def _policy_cfg_to_dict(policy_cfg: DictConfig) -> dict[str, Any]:
    """Resolve OmegaConf policy config into a plain string-keyed dict."""
    cfg_dict_raw = OmegaConf.to_container(policy_cfg, resolve=True)
    if not isinstance(cfg_dict_raw, dict):
        raise TypeError(
            f"Policy config must resolve to a dict, got {type(cfg_dict_raw)!r}."
        )
    return {
        str(key): value for key, value in cast(dict[Any, Any], cfg_dict_raw).items()
    }


def validate_tokenizer_cfg(*, policy_name: str, cfg_dict: dict[str, Any]) -> None:
    """Require pretrained token-codec checkpoints for policy construction."""
    action_tokenizer_cfg = cfg_dict.get("action_tokenizer")
    if action_tokenizer_cfg is None:
        return
    if not isinstance(action_tokenizer_cfg, dict):
        raise TypeError(
            f"{policy_name}.action_tokenizer must resolve to a dict, "
            f"got {type(action_tokenizer_cfg)!r}."
        )
    if action_tokenizer_cfg.get("kind") != "tokens":
        raise ValueError(
            f"{policy_name}.action_tokenizer.kind must be 'tokens', "
            f"got {action_tokenizer_cfg.get('kind')!r}."
        )

    pretrained_path = action_tokenizer_cfg.get("pretrained_path")
    if not pretrained_path:
        raise ValueError(
            f"{policy_name} requires action_tokenizer.pretrained_path when building from scratch. "
            "Fresh/untrained tokenizers are not allowed for policy construction because tokenizer "
            "checkpoints must carry action normalization stats."
        )
    _validate_tokenizer_stats(
        policy_name=policy_name, pretrained_path=str(pretrained_path)
    )


def _validate_tokenizer_stats(*, policy_name: str, pretrained_path: str) -> None:
    """Require action normalization stats in a token-codec checkpoint sidecar."""
    norm_path = resolve_sidecar_path(pretrained_path, NORM_STATS_FILENAME)
    if norm_path is None:
        raise ValueError(
            f"{policy_name} action_tokenizer.pretrained_path={pretrained_path!r} is missing "
            f"{NORM_STATS_FILENAME}. Tokenizer checkpoints used by policies must include action "
            "normalization stats."
        )
    try:
        norm_payload = torch.load(norm_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            f"{policy_name} failed to read tokenizer norm stats from {norm_path}."
        ) from exc
    if not isinstance(norm_payload, dict):
        raise ValueError(
            f"{policy_name} tokenizer norm stats file at {norm_path} must contain a dict payload."
        )
    stats = norm_payload.get("stats", {})
    if not isinstance(stats, dict) or not stats:
        raise ValueError(
            f"{policy_name} tokenizer norm stats file at {norm_path} has no stats entries."
        )

    expected_action_keys = {CANONICAL_ACTION_KEY}
    io_path = resolve_sidecar_path(pretrained_path, POLICY_IO_FILENAME)
    if io_path is not None:
        try:
            io_payload = json.loads(io_path.read_text())
        except Exception:
            io_payload = {}
        if isinstance(io_payload, dict):
            action_feature = io_payload.get("action_feature")
            if isinstance(action_feature, dict):
                action_key = action_feature.get("key")
                if isinstance(action_key, str) and action_key:
                    expected_action_keys.add(action_key)

    if not any(key in stats for key in expected_action_keys):
        raise ValueError(
            f"{policy_name} tokenizer norm stats at {norm_path} are missing action stats. "
            f"Expected one of keys {sorted(expected_action_keys)}, got {sorted(stats)}."
        )


def _validate_pretrained_action_dim(policy: BasePolicy, io: PolicyIO | None) -> None:
    """Ensure runtime dataset action_dim matches the loaded pretrained policy."""
    if io is None or policy.io is None:
        return
    if policy.io.action_dim != io.action_dim:
        raise ValueError(
            "Pretrained policy action_dim mismatch: "
            f"checkpoint action_dim={policy.io.action_dim}, dataset action_dim={io.action_dim}."
        )


def _load_pretrained_policy(name: str, pretrained_path: str) -> BasePolicy:
    """Load a fully self-contained policy from a checkpoint directory.

    Loads model weights, PolicyIO, and normalization stats.
    """
    policy_cls = get_policy_class(name)
    with _codec_path_base_from_checkpoint(pretrained_path):
        policy = policy_cls.from_pretrained(pretrained_path)
    if policy.io is None:
        raise RuntimeError(
            f"Policy checkpoint at {pretrained_path} did not load PolicyIO. "
            "Checkpoint is invalid or missing sidecars."
        )
    return policy


@contextmanager
def _codec_path_base_from_checkpoint(pretrained_path: str):
    """Set codec path base from checkpoint repo root for relative codec paths."""
    repo_root = _find_repo_root(pretrained_path)
    if repo_root is None:
        yield
        return

    from praxis.codecs.factory import codec_path_base

    with codec_path_base(repo_root):
        yield


def _find_repo_root(path_str: str) -> Path | None:
    path = Path(path_str).resolve()
    search_from = path if path.is_dir() else path.parent
    for candidate in (search_from, *search_from.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _get_explicit_policy_override_keys() -> set[str] | None:
    """Return top-level policy keys explicitly overridden in Hydra task overrides.

    Returns:
        - set[str]: explicit top-level keys under `policy.*` when Hydra runtime is active.
        - None: Hydra runtime unavailable; callers should preserve legacy behavior.
    """
    try:
        overrides_obj = HydraConfig.get().overrides
        task_overrides = list(overrides_obj.task)
    except Exception:
        return None

    keys: set[str] = set()
    for raw in task_overrides:
        item = str(raw).lstrip("+~")
        if not item.startswith("policy."):
            continue
        lhs = item.split("=", 1)[0]
        path = lhs.removeprefix("policy.")
        if not path:
            continue
        keys.add(path.split(".", 1)[0])
    return keys
