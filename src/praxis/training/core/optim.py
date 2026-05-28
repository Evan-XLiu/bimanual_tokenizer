"""Optimizer parameter-group adapter for trainer-owned LR recipes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf


def resolve_trainer_lr_config(
    trainer_cfg: Any,
) -> tuple[float | None, dict[str, float] | None]:
    """Resolve scalar codec LR or structured policy LR set from trainer config."""
    scalar_lr = _select_optional(trainer_cfg, "lr")
    family_lrs = _select_optional(trainer_cfg, "lrs")
    if scalar_lr is not None and family_lrs is not None:
        raise ValueError("trainer.lr and trainer.lrs are mutually exclusive.")
    if scalar_lr is None and family_lrs is None:
        raise ValueError("Trainer config must define either trainer.lr or trainer.lrs.")

    resolved_scalar = (
        None if scalar_lr is None else _positive_lr(scalar_lr, "trainer.lr")
    )
    resolved_families = None
    if family_lrs is not None:
        container = (
            OmegaConf.to_container(family_lrs, resolve=True)
            if isinstance(family_lrs, DictConfig)
            else family_lrs
        )
        if not isinstance(container, Mapping):
            raise TypeError(
                "trainer.lrs must be a mapping from parameter family to LR, "
                f"got {type(container)!r}."
            )
        resolved_families = {
            str(family): _positive_lr(lr, f"trainer.lrs.{family}")
            for family, lr in cast(Mapping[Any, Any], container).items()
        }
        if not resolved_families:
            raise ValueError("trainer.lrs must define at least one parameter family.")
    return resolved_scalar, resolved_families


def adapt_optimizer_param_groups(
    optim_params: list[dict],
    *,
    scalar_lr: float | None,
    family_lrs: dict[str, float] | None,
) -> list[dict]:
    """Convert model-owned param-family specs into PyTorch AdamW param groups."""
    if scalar_lr is None and family_lrs is None:
        raise ValueError("Cannot adapt optimizer groups without trainer LR config.")

    adapted: list[dict] = []
    for index, group in enumerate(optim_params):
        if "params" not in group:
            raise ValueError(f"Optimizer group {index} is missing required 'params'.")
        family = group.get("family")
        if family_lrs is None:
            if family is not None:
                raise ValueError(
                    "Optimizer group declares parameter family "
                    f"{family!r}, but trainer.lrs is not configured."
                )
            adapted.append(
                {key: value for key, value in group.items() if key != "family"}
            )
            continue

        if "lr" in group:
            raise ValueError(
                "Model optimizer groups must not assign 'lr' when trainer.lrs is configured."
            )
        if family is None:
            raise ValueError(
                f"Optimizer group {index} has no parameter family; "
                "trainer.lrs requires every trainable parameter group to be labeled."
            )
        family_name = str(family)
        try:
            lr = family_lrs[family_name]
        except KeyError as exc:
            available = ", ".join(sorted(family_lrs))
            raise ValueError(
                f"Optimizer group {index} uses unconfigured parameter family "
                f"{family_name!r}. Configure trainer.lrs.{family_name}. "
                f"Available families: {available or 'none'}."
            ) from exc
        converted = {key: value for key, value in group.items() if key != "family"}
        converted["lr"] = lr
        adapted.append(converted)
    return adapted


def optimizer_default_lr(
    *,
    scalar_lr: float | None,
    family_lrs: dict[str, float] | None,
) -> float:
    """Return the AdamW default LR; family groups carry explicit LRs."""
    if scalar_lr is not None:
        return scalar_lr
    if family_lrs:
        first_family = sorted(family_lrs)[0]
        return family_lrs[first_family]
    raise ValueError("Cannot choose optimizer default LR without trainer LR config.")


def _positive_lr(value: Any, name: str) -> float:
    lr = float(value)
    if lr <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}.")
    return lr


def _select_optional(config: Any, key: str) -> Any:
    if OmegaConf.is_config(config):
        return OmegaConf.select(config, key, default=None)
    if isinstance(config, Mapping):
        return config.get(key)
    return getattr(config, key, None)
