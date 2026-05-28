"""Shared lifecycle helpers for policies with discrete action tokenizers."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import Any, ClassVar, Protocol, cast

import torch
from torch import Tensor

from praxis.codecs.tokens.core import BaseTokenizer
from praxis.core.checkpoint_artifacts import collect_normalizer_stats
from praxis.core.types import PolicyIO
from praxis.policies.core.construction import build_config_policy
from praxis.policies.core.policy import BasePolicy


def action_tokenizer_config_factory(name: str) -> Callable[[], dict[str, object]]:
    """Return a dataclass default factory for a pretrained action tokenizer."""
    tokenizer_name = str(name)

    def factory() -> dict[str, object]:
        return {"kind": "tokens", "name": tokenizer_name, "pretrained_path": None}

    return factory


class _VariableLengthTokenizer(Protocol):
    """Tokenize continuous actions into variable-length token rows."""

    def tokenize(
        self,
        actions: Tensor,
        *,
        pad: bool = False,
    ) -> Tensor | list[Tensor] | list[list[int]]: ...


class ActionTokenPolicyBase(BasePolicy):
    """Base class for policies with discrete action tokenizers."""

    CONFIG_CLASS: ClassVar[type[Any] | None] = None
    _SCHEDULE_STRIP_KEYS: ClassVar[tuple[str, ...]] = ("labels",)
    action_tokenizer: BaseTokenizer
    _eos_token_id: int | None = None

    @classmethod
    def from_io(cls, io: PolicyIO, **config_overrides):
        resolved_overrides = cls._resolve_action_token_from_io(
            io=io,
            config_overrides=config_overrides,
        )
        return build_config_policy(
            cls,
            io=io,
            config_cls=cls.CONFIG_CLASS,
            resolved_overrides=resolved_overrides,
        )

    @classmethod
    def _resolve_action_token_from_io(
        cls,
        *,
        io: PolicyIO,
        config_overrides: dict[str, Any],
    ) -> dict[str, Any]:
        overrides = dict(config_overrides)
        overrides["action_dim"] = cls._resolve_action_dim_for_from_io(
            configured_action_dim=overrides.get("action_dim"),
            io=io,
            owner=cls.__name__,
        )

        action_tokenizer_cfg = overrides.get("action_tokenizer")
        if action_tokenizer_cfg is None:
            return overrides
        if not isinstance(action_tokenizer_cfg, dict):
            raise TypeError(
                f"{cls.__name__}.action_tokenizer must be a dict, "
                f"got {type(action_tokenizer_cfg)!r}."
            )

        tokenizer_cfg = dict(action_tokenizer_cfg)
        if tokenizer_cfg.get("pretrained_path"):
            overrides["action_tokenizer"] = tokenizer_cfg
            return overrides

        tokenizer_cfg["action_dim"] = cls._resolve_action_dim_for_from_io(
            configured_action_dim=tokenizer_cfg.get("action_dim"),
            io=io,
            owner=f"{cls.__name__}.action_tokenizer",
        )
        overrides["action_tokenizer"] = tokenizer_cfg
        return overrides

    @staticmethod
    def _extract_module_action_dim(module: Any) -> int | None:
        value = getattr(module, "action_dim", None)
        if value is None:
            cfg = getattr(module, "config", None)
            value = getattr(cfg, "action_dim", None)
        return int(value) if value is not None else None

    def _validate_action_tokenizer_action_dim(self, io: PolicyIO) -> None:
        action_tokenizer = getattr(self, "action_tokenizer", None)
        if action_tokenizer is None:
            return
        tokenizer_action_dim = self._extract_module_action_dim(action_tokenizer)
        if tokenizer_action_dim is None:
            return
        io_action_dim = int(io.action_dim)
        if tokenizer_action_dim != io_action_dim:
            raise ValueError(
                "Action tokenizer action_dim mismatch: "
                f"tokenizer action_dim={tokenizer_action_dim}, dataset action_dim={io_action_dim}."
            )

    def set_io(self, io: PolicyIO) -> None:
        super().set_io(io)
        if io.action_feature is not None:
            tokenizer_io = PolicyIO(
                obs_features=[],
                action_feature=io.action_feature,
                n_obs_steps=0,
                n_action_steps=io.n_action_steps,
                fps=io.fps,
            )
            self.action_tokenizer.set_io(tokenizer_io)
        self._validate_action_tokenizer_action_dim(io)
        self._configure_action_queue(n_action_steps=self.config.n_action_steps)
        self._post_set_io()

    def _post_set_io(self) -> None:
        """Hook for family-specific setup after tokenizer IO and queue setup."""

    @property
    def _supervise_eos(self) -> bool:
        return False

    def _get_action_token_lengths(
        self,
        token_ids_batch: list[list[int]],
    ) -> list[int] | None:
        del token_ids_batch
        return None

    def _resolve_action_token_horizon(
        self,
        token_ids_batch: list[list[int]],
        *,
        token_lengths: list[int],
    ) -> int:
        return max(
            max((len(ids) for ids in token_ids_batch), default=0),
            max((int(length) for length in token_lengths), default=0),
            1,
        )

    def _resolve_action_token_block_schedule(
        self, *, token_horizon: int
    ) -> tuple[int, ...]:
        return (1,) * int(token_horizon)

    def _export_norm_stats_for_checkpoint(
        self,
    ) -> tuple[dict[str, dict[str, Tensor]], dict[str, str]]:
        stats: dict[str, dict[str, Tensor]] = {}
        modes: dict[str, str] = {}
        collect_normalizer_stats(self.normalize, stats, modes)

        if self.io is not None and self.io.action_feature is not None:
            action_key = self.io.action_feature.key
            stats.pop(action_key, None)
            modes.pop(action_key, None)

        return stats, modes

    def _validate_frozen_tokenizer_contract(self) -> None:
        if not self._exclude_frozen_tokenizer_from_checkpoint():
            return
        action_tokenizer = getattr(self, "action_tokenizer", None)
        if action_tokenizer is not None and hasattr(
            action_tokenizer, "get_optim_params"
        ):
            optim_params = action_tokenizer.get_optim_params()
        elif action_tokenizer is not None:
            params = [p for p in action_tokenizer.parameters() if p.requires_grad]
            optim_params = [{"params": params}] if params else []
        else:
            optim_params = []
        config = getattr(self, "config", None)
        tokenizer_cfg = (
            getattr(config, "action_tokenizer", None) if config is not None else None
        )
        if tokenizer_cfg is None:
            tokenizer_cfg = {}
        pretrained_path = (
            tokenizer_cfg.get("pretrained_path")
            if isinstance(tokenizer_cfg, dict)
            else None
        )
        if optim_params and not pretrained_path:
            raise ValueError(
                "freeze_action_tokenizer=true with a trainable tokenizer requires "
                "action_tokenizer.pretrained_path. Policy checkpoints intentionally "
                "exclude frozen tokenizer weights, so tokenizer must load from its own checkpoint."
            )

    def _exclude_frozen_tokenizer_from_checkpoint(self) -> bool:
        config = getattr(self, "config", None)
        return bool(getattr(config, "freeze_action_tokenizer", False))

    @staticmethod
    def _is_tokenizer_state_key(key: str) -> bool:
        return key.startswith("action_tokenizer.")

    def filter_checkpoint_export_state_dict(self, state_dict: dict) -> dict:
        out = dict(state_dict)
        if self._exclude_frozen_tokenizer_from_checkpoint():
            out = {
                key: value
                for key, value in out.items()
                if not self._is_tokenizer_state_key(key)
            }
        return out

    def filter_checkpoint_incompatible_keys(
        self,
        *,
        missing: list[str] | set[str] | tuple[str, ...],
        unexpected: list[str] | set[str] | tuple[str, ...],
    ) -> tuple[list[str], list[str]]:
        filtered_missing = list(missing)
        filtered_unexpected = list(unexpected)
        if self._exclude_frozen_tokenizer_from_checkpoint():
            filtered_missing = [
                key for key in filtered_missing if not self._is_tokenizer_state_key(key)
            ]
            filtered_unexpected = [
                key
                for key in filtered_unexpected
                if not self._is_tokenizer_state_key(key)
            ]
        return filtered_missing, filtered_unexpected

    def state_dict(self, *args, **kwargs):
        out = super().state_dict(*args, **kwargs)
        if not self._exclude_frozen_tokenizer_from_checkpoint():
            return out
        for key in list(out.keys()):
            if self._is_tokenizer_state_key(key):
                out.pop(key)
        return out

    def load_state_dict(
        self,
        state_dict,
        strict: bool = True,
        assign: bool = False,
    ):
        if not self._exclude_frozen_tokenizer_from_checkpoint():
            return super().load_state_dict(state_dict, strict=strict, assign=assign)

        filtered = OrderedDict(
            (k, v) for k, v in state_dict.items() if not self._is_tokenizer_state_key(k)
        )
        incompatible = super().load_state_dict(
            filtered,
            strict=False,
            assign=assign,
        )
        if strict:
            missing = [
                k
                for k in incompatible.missing_keys
                if not self._is_tokenizer_state_key(k)
            ]
            unexpected = [
                k
                for k in incompatible.unexpected_keys
                if not self._is_tokenizer_state_key(k)
            ]
            if missing or unexpected:
                raise RuntimeError(
                    f"Error(s) in loading state_dict for {self.__class__.__name__}: "
                    f"missing_keys={missing}, unexpected_keys={unexpected}"
                )
        return incompatible


class VariableLengthActionTokenMixin:
    """Mixin for policies whose tokenizers emit per-sample token rows."""

    action_tokenizer: object

    def tokenize_actions_to_policy_tokens(self, actions: Tensor) -> list[list[int]]:
        with torch.inference_mode():
            tokenized = self._tokenize_variable_length_actions(actions)
        return [
            ids.long().tolist() if isinstance(ids, Tensor) else [int(v) for v in ids]
            for ids in tokenized
        ]

    def _tokenize_variable_length_actions(
        self,
        actions: Tensor,
    ) -> Tensor | list[Tensor] | list[list[int]]:
        tokenizer = cast(_VariableLengthTokenizer, self.action_tokenizer)
        return tokenizer.tokenize(actions, pad=False)

    def _get_action_token_lengths(
        self,
        token_ids_batch: list[list[int]],
    ) -> list[int]:
        return [len(ids) for ids in token_ids_batch]

    def _resolve_action_token_horizon(
        self,
        token_ids_batch: list[list[int]],
        *,
        token_lengths: list[int],
    ) -> int:
        tokenizer = cast(_VariableLengthTokenizer, self.action_tokenizer)
        max_code_seq_len = getattr(tokenizer, "max_code_seq_len", None)
        if max_code_seq_len is None:
            max_code_seq_len = max(
                max((len(ids) for ids in token_ids_batch), default=0),
                max(token_lengths, default=0),
            )
        horizon = int(max_code_seq_len)
        if cast(ActionTokenPolicyBase, self)._supervise_eos:
            horizon += 1
        return max(horizon, 1)

    def _resolve_action_token_block_schedule(
        self, *, token_horizon: int
    ) -> tuple[int, ...]:
        return (1,) * int(token_horizon)
