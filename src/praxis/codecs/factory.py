# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Codec factory: instantiate action codecs from config with optional pretrained loading."""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from praxis.codecs.core import BaseCodec
from praxis.core.config import config_to_dict, config_values_equal


def _ref(module_path: str, class_name: str) -> str:
    return f"praxis.codecs.{module_path}:{class_name}"


_CODEC_REGISTRY: dict[tuple[str, str], str] = {
    ("tokens", "acodec"): _ref("tokens.acodec.tokenizer", "ActionCodecTokenizer"),
    ("tokens", "boat"): _ref("tokens.boat.tokenizer", "BOATTokenizer"),
    ("tokens", "bimanualoat"): _ref(
        "tokens.bimanualoat.tokenizer", "BimanualOATTokenizer"
    ),
    ("tokens", "bin"): _ref("tokens.bin.tokenizer", "BinTokenizer"),
    ("tokens", "fast"): _ref("tokens.fast.tokenizer", "FASTTokenizer"),
    ("tokens", "oat"): _ref("tokens.oat.tokenizer", "OATTokenizer"),
    ("tokens", "offline_targets"): _ref(
        "tokens.offline_targets.tokenizer", "OfflineTargetTokenizer"
    ),
    ("tokens", "quest"): _ref("tokens.quest.tokenizer", "QueSTTokenizer"),
    ("tokens", "bquest"): _ref("tokens.bquest.tokenizer", "BQuestTokenizer"),
    ("tokens", "btt"): _ref("tokens.btt.tokenizer", "BTTTokenizer"),
    ("tokens", "btt_oneblock"): _ref(
        "tokens.btt_oneblock.tokenizer", "BTTOneBlockTokenizer"
    ),
    ("tokens", "btt_independent"): _ref(
        "tokens.btt_independent.tokenizer", "BTTIndependentTokenizer"
    ),
    ("latents", "acodec"): _ref("latents.acodec.autoencoder", "ActionCodecAutoencoder"),
    ("latents", "oat"): _ref("latents.oat.autoencoder", "OATAutoencoder"),
    ("latents", "quest"): _ref("latents.quest.autoencoder", "QueSTAutoencoder"),
    ("latents", "xtokenizer"): _ref("latents.xtokenizer.codec", "XTokenizerCodec"),
}
_TRAINER_ONLY_CODEC_CFG_KEYS = frozenset({"max_fit_chunks"})
_CODEC_PATH_BASE: ContextVar[Path | None] = ContextVar(
    "praxis_codec_path_base", default=None
)


def register_codec(name: str, import_path: str, *, kind: str) -> None:
    """Register an action-codec class import path.

    Args:
        name: Registry key used by codec config ``name``.
        import_path: ``"module.path:ClassName"``.
        kind: Codec namespace, for example ``"tokens"`` or ``"latents"``.
    """
    registry_key = (_normalize_codec_kind(kind), _normalize_codec_name(name))
    if ":" not in import_path:
        raise ValueError(
            f"Invalid import path {import_path!r}. Expected 'module.path:ClassName'."
        )
    _CODEC_REGISTRY[registry_key] = import_path


def available_codecs(*, kind: str | None = None) -> tuple[str, ...]:
    """Return registered codec names."""
    if kind is not None:
        resolved_kind = _normalize_codec_kind(kind)
        return tuple(
            sorted(
                name
                for (codec_kind, name) in _CODEC_REGISTRY
                if codec_kind == resolved_kind
            )
        )
    return tuple(sorted(f"{codec_kind}/{name}" for codec_kind, name in _CODEC_REGISTRY))


@contextmanager
def codec_path_base(path: str | Path | None):
    """Temporarily resolve relative pretrained codec paths against ``path``."""
    resolved = Path(path).resolve() if path is not None else None
    token = _CODEC_PATH_BASE.set(resolved)
    try:
        yield
    finally:
        _CODEC_PATH_BASE.reset(token)


def current_codec_path_base() -> Path | None:
    """Return the active codec path base, when one is set."""
    return _CODEC_PATH_BASE.get()


def make_codec(codec_cfg: DictConfig | dict[str, Any]) -> BaseCodec:
    """Instantiate an action codec from config.

    Supports:
    1. from-scratch construction via config dict
    2. pretrained loading via ``pretrained_path`` using HF Hub mixin
    """
    if isinstance(codec_cfg, DictConfig):
        raw_cfg = OmegaConf.to_container(codec_cfg, resolve=True)
    else:
        raw_cfg = dict(codec_cfg)
    if not isinstance(raw_cfg, dict):
        raise TypeError(f"codec_cfg must resolve to dict, got {type(raw_cfg)!r}")
    cfg_dict = {str(key): value for key, value in cast(dict[Any, Any], raw_cfg).items()}

    if "name" not in cfg_dict:
        raise ValueError("Codec config must include a non-empty `name`.")
    if "kind" not in cfg_dict:
        raise ValueError("Codec config must include a non-empty `kind`.")
    name = _normalize_codec_name(cfg_dict["name"])
    kind = _normalize_codec_kind(cfg_dict["kind"])

    codec_cls = _get_codec_class(kind=kind, name=name)
    pretrained_path = cfg_dict.get("pretrained_path")
    if pretrained_path:
        load_id = _resolve_codec_load_id(str(pretrained_path))
        source = codec_cls.from_pretrained(load_id)

        build_cfg = _codec_build_cfg(cfg_dict)
        source_cfg = _extract_codec_cfg_dict(source)

        source_action_dim = _extract_codec_action_dim(source)
        if build_cfg.get("action_dim") is None and source_action_dim is not None:
            build_cfg["action_dim"] = source_action_dim

        _validate_pretrained_runtime_overrides(
            codec_cls=codec_cls,
            build_cfg=build_cfg,
            source_cfg=source_cfg,
            load_id=load_id,
        )

        if not _has_runtime_overrides(build_cfg, source_cfg):
            return source

        runtime_cfg = dict(cfg_dict)
        runtime_cfg.update(build_cfg)
        target = _build_codec_from_cfg(codec_cls, runtime_cfg)
        _transplant_pretrained_state(source=source, target=target, load_id=load_id)
        return target

    return _build_codec_from_cfg(codec_cls, cfg_dict)


def _normalize_codec_kind(kind: Any) -> str:
    value = str(kind).strip().lower()
    if not value:
        raise ValueError("Codec kind must be non-empty.")
    return value


def _normalize_codec_name(name: Any) -> str:
    value = str(name).strip().lower()
    if not value:
        raise ValueError("Codec config must include a non-empty `name`.")
    return value


def _resolve_codec_load_id(pretrained_path: str) -> str:
    """Resolve local codec paths while preserving Hub ids.

    Resolution order:
    1) absolute path as-is when it exists
    2) cwd-relative path when it exists
    3) active codec path base + relative path when it exists
    4) fallback to raw string (treated as Hub id by HF mixin)
    """
    path_candidate = Path(pretrained_path)
    if path_candidate.is_absolute():
        return str(path_candidate) if path_candidate.exists() else pretrained_path

    if path_candidate.exists():
        return str(path_candidate.resolve())

    base = current_codec_path_base()
    if base:
        base_candidate = (base / path_candidate).resolve()
        if base_candidate.exists():
            return str(base_candidate)

    return pretrained_path


def _codec_build_cfg(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg_dict)
    out.pop("name", None)
    out.pop("kind", None)
    out.pop("pretrained_path", None)
    for key in _TRAINER_ONLY_CODEC_CFG_KEYS:
        out.pop(key, None)
    return out


def _extract_codec_cfg_dict(codec: BaseCodec) -> dict[str, Any]:
    cfg = getattr(codec, "config", None)
    payload = config_to_dict(cfg)
    return {} if payload is None else payload


def _extract_codec_action_dim(codec: BaseCodec) -> int | None:
    action_dim = getattr(codec, "action_dim", None)
    if action_dim is None:
        cfg = getattr(codec, "config", None)
        action_dim = getattr(cfg, "action_dim", None)
    return int(action_dim) if action_dim is not None else None


def _has_runtime_overrides(
    build_cfg: dict[str, Any], source_cfg: dict[str, Any]
) -> bool:
    if not build_cfg:
        return False
    for key, value in build_cfg.items():
        if key not in source_cfg:
            return True
        if not config_values_equal(value, source_cfg[key]):
            return True
    return False


def _validate_pretrained_runtime_overrides(
    *,
    codec_cls: type[BaseCodec],
    build_cfg: dict[str, Any],
    source_cfg: dict[str, Any],
    load_id: str,
) -> None:
    validator = getattr(codec_cls, "validate_pretrained_runtime_overrides", None)
    if callable(validator):
        validator(build_cfg=build_cfg, source_cfg=source_cfg, load_id=load_id)


def _transplant_pretrained_state(
    *, source: BaseCodec, target: BaseCodec, load_id: str
) -> None:
    incompat = target.load_state_dict(source.state_dict(), strict=False)
    if incompat.missing_keys:
        raise RuntimeError(
            "Missing pretrained keys when loading into runtime codec: "
            f"{incompat.missing_keys}. Check config override compatibility for {load_id}."
        )
    if incompat.unexpected_keys:
        raise RuntimeError(
            "Unexpected pretrained keys when loading into runtime codec: "
            f"{incompat.unexpected_keys}. Check config override compatibility for {load_id}."
        )

    load_runtime_state = getattr(target, "load_pretrained_runtime_state", None)
    if callable(load_runtime_state):
        load_runtime_state(source)

    source_io = getattr(source, "io", None)
    if source_io is not None:
        target.set_io(source_io)

    normalization_payload = source._normalization_checkpoint_payload()
    if "normalization_kind" in normalization_payload or normalization_payload.get(
        "stats"
    ):
        target._load_normalization_checkpoint_payload(normalization_payload)


def _build_codec_from_cfg(
    codec_cls: type[BaseCodec], cfg_dict: dict[str, Any]
) -> BaseCodec:
    build_cfg = dict(cfg_dict)
    build_cfg.pop("name", None)
    build_cfg.pop("kind", None)
    build_cfg.pop("pretrained_path", None)
    for key in _TRAINER_ONLY_CODEC_CFG_KEYS:
        build_cfg.pop(key, None)
    return codec_cls(config=build_cfg)


def _get_codec_class(*, kind: str, name: str) -> type[BaseCodec]:
    registry_key = (_normalize_codec_kind(kind), _normalize_codec_name(name))
    if registry_key not in _CODEC_REGISTRY:
        raise ValueError(
            f"Unknown codec type: {kind}/{name}. Available: {', '.join(available_codecs())}"
        )
    module_path, class_name = _CODEC_REGISTRY[registry_key].split(":", 1)
    module = importlib.import_module(module_path)
    codec_cls = getattr(module, class_name, None)
    if codec_cls is None:
        raise ImportError(f"Could not import {class_name!r} from {module_path!r}.")
    if not isinstance(codec_cls, type) or not issubclass(codec_cls, BaseCodec):
        raise TypeError(
            f"Registered codec {kind}/{name} resolved to {codec_cls!r}, expected BaseCodec subclass."
        )
    return codec_cls
