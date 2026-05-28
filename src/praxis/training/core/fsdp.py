"""FSDP-specific module selection and runtime helpers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, cast

import torch
from torch import nn


def module_overrides_forward(module: nn.Module) -> bool:
    """Return whether ``module`` implements a real forward method."""
    return type(module).forward is not nn.Module.forward


def _module_param_count(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters(recurse=True))


def _module_depths(model: nn.Module) -> dict[nn.Module, int]:
    return {
        module: 0 if name == "" else name.count(".") + 1
        for name, module in model.named_modules()
    }


def _dedupe_modules(modules: Iterator[nn.Module]) -> list[nn.Module]:
    seen: set[int] = set()
    out: list[nn.Module] = []
    for module in modules:
        marker = id(module)
        if marker in seen:
            continue
        seen.add(marker)
        out.append(module)
    return out


def _normalize_fsdp_block_types(raw: Any) -> tuple[type[nn.Module], ...]:
    if raw is None:
        return ()
    if isinstance(raw, type) and issubclass(raw, nn.Module):
        return (raw,)
    if not isinstance(raw, tuple | list | set | frozenset):
        raise TypeError(
            "get_fsdp_block_types() must return an nn.Module type or an iterable "
            f"of nn.Module types, got {type(raw).__name__}."
        )
    block_types: list[type[nn.Module]] = []
    for item in raw:
        if not isinstance(item, type) or not issubclass(item, nn.Module):
            raise TypeError(
                f"get_fsdp_block_types() entries must be nn.Module types, got {item!r}."
            )
        block_types.append(cast(type[nn.Module], item))
    return tuple(block_types)


def _declared_fsdp_wrap_modules(model: nn.Module) -> list[nn.Module]:
    hook = getattr(model, "fsdp_wrap_modules", None)
    if callable(hook):
        raw_modules = hook()
        if not isinstance(raw_modules, Iterable) or isinstance(
            raw_modules, str | bytes
        ):
            raise TypeError(
                "fsdp_wrap_modules() must return an iterable of nn.Module instances, "
                f"got {type(raw_modules).__name__}."
            )
        modules = list(raw_modules)
        for module in modules:
            if not isinstance(module, nn.Module):
                raise TypeError(
                    "fsdp_wrap_modules() must return nn.Module instances, "
                    f"got {type(module).__name__}."
                )
        return _dedupe_modules(iter(cast(list[nn.Module], modules)))

    block_type_hook = getattr(model, "get_fsdp_block_types", None)
    if callable(block_type_hook):
        block_types = _normalize_fsdp_block_types(block_type_hook())
        if block_types:
            return _dedupe_modules(
                module
                for module in model.modules()
                if module is not model and isinstance(module, block_types)
            )
    return []


def _size_based_fsdp_wrap_modules(
    model: nn.Module,
    *,
    min_num_params: int,
) -> list[nn.Module]:
    return [
        module
        for module in model.modules()
        if module is not model
        and module_overrides_forward(module)
        and _module_param_count(module) >= min_num_params
    ]


def select_fsdp_wrap_modules(
    model: nn.Module,
    *,
    min_num_params: int,
) -> list[nn.Module]:
    """Select bottom-up FSDP module groups for ``fully_shard``.

    Models may explicitly declare boundaries with ``fsdp_wrap_modules()`` or
    ``get_fsdp_block_types()``. If they do not, this falls back to a conservative
    size-based policy.
    """
    has_declared_policy = callable(
        getattr(model, "fsdp_wrap_modules", None)
    ) or callable(getattr(model, "get_fsdp_block_types", None))
    declared = _declared_fsdp_wrap_modules(model)
    if has_declared_policy:
        modules = declared
    else:
        modules = _size_based_fsdp_wrap_modules(
            model,
            min_num_params=min_num_params,
        )
    depths = _module_depths(model)
    valid_modules = [
        module
        for module in modules
        if module is not model
        and module in depths
        and module_overrides_forward(module)
        and _module_param_count(module) > 0
    ]
    if has_declared_policy and not valid_modules:
        raise ValueError(
            f"{model.__class__.__name__} declares FSDP wrap policy hooks, but "
            "no valid parameterized forward modules were selected."
        )
    return sorted(
        _dedupe_modules(iter(valid_modules)),
        key=lambda module: depths[module],
        reverse=True,
    )


def scalar_parameters(module: nn.Module) -> frozenset[nn.Parameter] | None:
    params = frozenset(
        param for param in module.parameters(recurse=True) if param.ndim == 0
    )
    return params or None


def _candidate_parameters(
    module: nn.Module,
    *,
    excluded_params: set[nn.Parameter] | frozenset[nn.Parameter] | None = None,
) -> list[nn.Parameter]:
    excluded = excluded_params or frozenset()
    return [param for param in module.parameters(recurse=True) if param not in excluded]


def _dominant_parameter_dtype(module: nn.Module) -> torch.dtype | None:
    dtype_sizes: dict[torch.dtype, int] = {}
    for param in module.parameters(recurse=True):
        dtype_sizes[param.dtype] = dtype_sizes.get(param.dtype, 0) + param.numel()
    if not dtype_sizes:
        return None
    return max(dtype_sizes.items(), key=lambda item: item[1])[0]


def _dominant_dtype_for_params(params: list[nn.Parameter]) -> torch.dtype | None:
    dtype_sizes: dict[torch.dtype, int] = {}
    for param in params:
        dtype_sizes[param.dtype] = dtype_sizes.get(param.dtype, 0) + param.numel()
    if not dtype_sizes:
        return None
    return max(dtype_sizes.items(), key=lambda item: item[1])[0]


def _managed_parameters(
    module: nn.Module,
    *,
    ignored_params: frozenset[nn.Parameter] | None = None,
    excluded_params: set[nn.Parameter] | frozenset[nn.Parameter] | None = None,
) -> list[nn.Parameter]:
    ignored = ignored_params or frozenset()
    return [
        param
        for param in _candidate_parameters(module, excluded_params=excluded_params)
        if param not in ignored
    ]


def validate_fsdp_parameter_group(
    module: nn.Module,
    *,
    ignored_params: frozenset[nn.Parameter] | None = None,
    excluded_params: set[nn.Parameter] | frozenset[nn.Parameter] | None = None,
) -> None:
    """Reject FSDP groups that would silently produce invalid training behavior."""
    ignored = ignored_params or frozenset()
    trainable_scalars = [
        name
        for name, param in module.named_parameters(recurse=True)
        if param not in (excluded_params or frozenset())
        and param not in ignored
        and param.requires_grad
        and param.ndim == 0
    ]
    if trainable_scalars:
        raise ValueError(
            "FSDP cannot safely ignore or shard trainable scalar parameter(s): "
            f"{trainable_scalars}. Convert them to length-1 tensors or freeze them."
        )

    trainable_dtypes = {
        param.dtype
        for param in _candidate_parameters(module, excluded_params=excluded_params)
        if param.requires_grad and param not in ignored
    }
    if len(trainable_dtypes) > 1:
        raise ValueError(
            "FSDP parameter group contains trainable parameters with mixed dtypes "
            f"{sorted(str(dtype) for dtype in trainable_dtypes)} in "
            f"{module.__class__.__name__}. Add narrower fsdp_wrap_modules() "
            "boundaries so trainable dtype groups are sharded separately."
        )
    managed_dtypes = {
        param.dtype
        for param in _managed_parameters(
            module,
            ignored_params=ignored,
            excluded_params=excluded_params,
        )
    }
    if len(managed_dtypes) > 1:
        raise ValueError(
            "FSDP parameter group contains managed parameters with mixed dtypes "
            f"{sorted(str(dtype) for dtype in managed_dtypes)} in "
            f"{module.__class__.__name__}. Add narrower fsdp_wrap_modules() "
            "boundaries or freeze dtype outliers so each group has one dtype."
        )


def ignored_fsdp_parameters(
    module: nn.Module,
    *,
    excluded_params: set[nn.Parameter] | frozenset[nn.Parameter] | None = None,
) -> frozenset[nn.Parameter] | None:
    """Return parameters to leave out of a single FSDP group.

    FSDP requires each parameter group to have a uniform original dtype. Trainable
    mixed-dtype modules should be separated by wrap boundaries. Frozen dtype
    outliers can instead stay replicated, which is useful for small frozen
    tokenizers or frozen scalar parameters mixed into a larger BF16 model.
    """
    params = _candidate_parameters(module, excluded_params=excluded_params)
    trainable_scalars = [
        param for param in params if param.ndim == 0 and param.requires_grad
    ]
    if trainable_scalars:
        raise ValueError(
            "FSDP cannot safely handle trainable scalar parameters. Convert scalar "
            "parameters to length-1 tensors or freeze them before enabling FSDP."
        )

    ignored: set[nn.Parameter] = {param for param in params if param.ndim == 0}
    trainable_dtypes = {param.dtype for param in params if param.requires_grad}
    if len(trainable_dtypes) > 1:
        raise ValueError(
            "FSDP cannot safely group trainable parameters with mixed dtypes. "
            "Add narrower fsdp_wrap_modules() boundaries so trainable dtype "
            "groups are sharded separately."
        )
    target_dtype = (
        next(iter(trainable_dtypes))
        if len(trainable_dtypes) == 1
        else _dominant_dtype_for_params(params)
    )
    if target_dtype is not None:
        ignored.update(
            param
            for param in params
            if not param.requires_grad and param.dtype != target_dtype
        )
    return frozenset(ignored) or None


def set_fsdp_accumulation_sync(model: nn.Module, enabled: bool) -> None:
    """Toggle the FSDP sync controls used during gradient accumulation."""
    for name in (
        "set_requires_gradient_sync",
        "set_requires_all_reduce",
        "set_reshard_after_backward",
        "set_is_last_backward",
    ):
        method = getattr(model, name, None)
        if callable(method):
            method(bool(enabled))
