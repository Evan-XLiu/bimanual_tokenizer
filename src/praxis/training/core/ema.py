# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Exponential Moving Average for model parameters."""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn

from praxis.training.core.dtensor_utils import (
    dtensor_has_sharded_placement,
    dtensor_to_local,
    is_dtensor,
)


def _tensor_for_shadow(source: torch.Tensor, shadow: torch.Tensor) -> torch.Tensor:
    if is_dtensor(source):
        dtensor_source = cast(Any, source)
        if dtensor_has_sharded_placement(source):
            return cast(torch.Tensor, dtensor_source.to_local())
        if shadow.shape == source.shape:
            return cast(torch.Tensor, dtensor_source.full_tensor())
        return cast(torch.Tensor, dtensor_source.to_local())
    return source


def _to_tensor_layout(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    source_is_full_tensor: bool = False,
) -> torch.Tensor:
    """Return source with target's DTensor layout when needed."""
    out_dtype = dtype or source.dtype
    if is_dtensor(target):
        from torch.distributed.tensor import DTensor, distribute_tensor

        dtensor_target = cast(Any, target)
        local_target = cast(torch.Tensor, dtensor_target.to_local())
        if not is_dtensor(source) and source.shape == target.shape:
            if dtensor_has_sharded_placement(target) and not source_is_full_tensor:
                local_source = source.to(device=local_target.device, dtype=out_dtype)
                return DTensor.from_local(
                    local_source,
                    device_mesh=dtensor_target.device_mesh,
                    placements=dtensor_target.placements,
                    shape=target.shape,
                    stride=target.stride(),
                )
            return distribute_tensor(
                source.to(device=local_target.device, dtype=out_dtype),
                dtensor_target.device_mesh,
                dtensor_target.placements,
            )
        local_source = dtensor_to_local(source).to(
            device=local_target.device,
            dtype=out_dtype,
        )
        return DTensor.from_local(
            local_source,
            device_mesh=dtensor_target.device_mesh,
            placements=dtensor_target.placements,
            shape=target.shape,
            stride=target.stride(),
        )
    return dtensor_to_local(source).to(device=target.device, dtype=out_dtype)


def _to_param_layout(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return source with target's dtype and DTensor layout when needed."""
    return _to_tensor_layout(source, target, dtype=target.dtype)


def _reshard_fsdp_modules(model: nn.Module) -> None:
    """Return FSDP modules to their registered sharded parameter state."""
    for module in reversed(list(model.modules())):
        reshard = getattr(module, "reshard", None)
        if callable(reshard):
            reshard()


class EMAModel:
    """Maintains an exponential moving average of model parameters.

    Usage:
        ema = EMAModel(model, decay=0.995)
        # After each optimizer step:
        ema.update(model)
        # For evaluation:
        ema.apply(model)       # copy shadow → model
        # ... evaluate ...
        ema.restore(model)     # copy original → model
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.995,
        *,
        shadow_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.decay = decay
        self.shadow_dtype = shadow_dtype
        # Paired lists for foreach ops: shadow[i] corresponds to _param_refs[i].
        self._names: list[str] = []
        self._shadow_params: list[torch.Tensor] = []
        self._param_refs: list[torch.Tensor] = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._names.append(name)
                self._shadow_params.append(
                    dtensor_to_local(param.detach()).clone().to(shadow_dtype)
                )
                self._param_refs.append(param.data)
        self._tracked_names = set(self._names)
        self._shadows_are_dtensor_local = False
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow parameters with EMA.

        Keep the shadow in FP32 by default. This matters for high decay values:
        BF16 can round values such as 0.9999 to 1.0 and stop EMA updates.
        """
        # Refresh param refs in case model was re-wrapped (DDP/FSDP).
        self._refresh_param_refs(model)
        for shadow, param in zip(self._shadow_params, self._param_refs, strict=True):
            source = (
                _to_param_layout(param.detach(), shadow)
                if is_dtensor(shadow)
                else _tensor_for_shadow(param.detach(), shadow).to(
                    device=shadow.device,
                    dtype=shadow.dtype,
                )
            )
            shadow.lerp_(
                source,
                1.0 - self.decay,
            )

    def refresh(self, model: nn.Module) -> None:
        """Re-point EMA references after the model has been wrapped or sharded."""
        _reshard_fsdp_modules(model)
        self._refresh_param_refs(model)
        found_dtensor_param = False
        for i, param in enumerate(self._param_refs):
            if not is_dtensor(param):
                continue
            found_dtensor_param = True
            shadow = self._shadow_params[i]
            sharded_shadow = _to_tensor_layout(
                shadow,
                param,
                dtype=shadow.dtype,
                source_is_full_tensor=not self._shadows_are_dtensor_local,
            )
            self._shadow_params[i] = (
                cast(torch.Tensor, cast(Any, sharded_shadow).to_local())
                .detach()
                .clone()
            )
        self._shadows_are_dtensor_local = found_dtensor_param

    def _refresh_param_refs(self, model: nn.Module) -> None:
        """Re-point _param_refs to current model param data tensors."""
        idx = {name: i for i, name in enumerate(self._names)}
        for name, param in model.named_parameters():
            if name in idx:
                self._param_refs[idx[name]] = param.data

    def apply(
        self,
        model: nn.Module,
        *,
        backup_device: torch.device | str | None = None,
    ) -> None:
        """Copy shadow parameters into model (save originals for restore)."""
        _reshard_fsdp_modules(model)
        shadow_dict = self._raw_shadow_dict()
        backup_target = (
            torch.device(backup_device) if backup_device is not None else None
        )
        self.backup = {
            name: (
                param.data.clone()
                if backup_target is None
                else dtensor_to_local(param.data.detach()).to(
                    device=backup_target,
                    copy=True,
                )
            )
            for name, param in model.named_parameters()
            if name in shadow_dict
        }
        for name, param in model.named_parameters():
            if name in shadow_dict:
                param.data.copy_(_to_param_layout(shadow_dict[name], param.data))

    def restore(self, model: nn.Module) -> None:
        """Restore original parameters from backup."""
        _reshard_fsdp_modules(model)
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(_to_param_layout(self.backup[name], param.data))
        self.backup.clear()

    def _raw_shadow_dict(self) -> dict[str, torch.Tensor]:
        return dict(zip(self._names, self._shadow_params, strict=True))

    def _layout_shadow_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: _to_tensor_layout(shadow, param, dtype=shadow.dtype)
            for name, shadow, param in zip(
                self._names,
                self._shadow_params,
                self._param_refs,
                strict=True,
            )
        }

    @property
    def restore_needed(self) -> bool:
        """Return whether ``restore()`` has model parameters to restore."""
        return bool(self.backup)

    @property
    def shadow(self) -> dict[str, torch.Tensor]:
        """Named view of tracked EMA parameters."""
        return self._raw_shadow_dict()

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self._layout_shadow_dict()

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        missing = [name for name in self._names if name not in state_dict]
        if missing:
            raise RuntimeError(
                f"EMA state is missing tracked parameter(s): {', '.join(missing)}."
            )
        # Unexpected keys are ignored deliberately: EMA only tracks the current
        # trainable parameter set.

        loaded_dtensor = False
        for i, name in enumerate(self._names):
            value = state_dict[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"EMA state for {name!r} must be a tensor, "
                    f"got {type(value).__name__}."
                )
            loaded = _tensor_for_shadow(value, self._shadow_params[i])
            shadow = self._shadow_params[i]
            if loaded.shape != shadow.shape:
                raise RuntimeError(
                    f"EMA state shape mismatch for {name!r}: expected "
                    f"{tuple(shadow.shape)}, got {tuple(loaded.shape)}."
                )
            if loaded.dtype != shadow.dtype:
                raise RuntimeError(
                    f"EMA state dtype mismatch for {name!r}: expected "
                    f"{shadow.dtype}, got {loaded.dtype}."
                )
            loaded_dtensor = loaded_dtensor or is_dtensor(value)
            shadow.copy_(loaded.to(device=shadow.device))
        self._shadows_are_dtensor_local = (
            self._shadows_are_dtensor_local or loaded_dtensor
        )
