# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from praxis.training.core.fsdp import (
    ignored_fsdp_parameters,
    module_overrides_forward,
    scalar_parameters,
    select_fsdp_wrap_modules,
    validate_fsdp_parameter_group,
)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.out = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.linear(x))


class _HookedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block(), _Block()])
        self.tail = nn.Linear(2, 2)

    def fsdp_wrap_modules(self) -> tuple[nn.Module, ...]:
        return tuple(self.blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.tail(x)


class _TypedBlockModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block(), _Block()])

    def get_fsdp_block_types(self) -> tuple[type[nn.Module], ...]:
        return (_Block,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class _EmptyHookedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.large = _Block()

    def fsdp_wrap_modules(self) -> tuple[nn.Module, ...]:
        return ()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.large(x)


class _ScalarParamModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.linear = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) * self.scale


def test_fsdp_wrap_policy_skips_no_forward_containers() -> None:
    assert module_overrides_forward(nn.Linear(1, 1)) is True
    assert module_overrides_forward(nn.Sequential(nn.Linear(1, 1))) is True
    assert module_overrides_forward(nn.ModuleList([nn.Linear(1, 1)])) is False
    assert module_overrides_forward(nn.ModuleDict({"layer": nn.Linear(1, 1)})) is False


def test_fsdp_selects_explicit_model_wrap_modules() -> None:
    model = _HookedModel()

    selected = select_fsdp_wrap_modules(model, min_num_params=10**9)

    assert selected == list(model.blocks)
    assert model.tail not in selected


def test_fsdp_selects_declared_block_types() -> None:
    model = _TypedBlockModel()

    selected = select_fsdp_wrap_modules(model, min_num_params=10**9)

    assert selected == list(model.blocks)


def test_fsdp_falls_back_to_size_based_wrap_modules() -> None:
    model = nn.Sequential(_Block(), _Block())

    selected = select_fsdp_wrap_modules(model, min_num_params=7)

    assert selected == list(model)


def test_fsdp_declared_empty_wrap_policy_fails_instead_of_fallback() -> None:
    model = _EmptyHookedModel()

    with pytest.raises(ValueError, match="declares FSDP wrap policy"):
        select_fsdp_wrap_modules(model, min_num_params=1)


def test_fsdp_frozen_scalar_parameters_are_ignored() -> None:
    model = _ScalarParamModel()
    model.scale.requires_grad_(False)

    ignored = scalar_parameters(model)

    assert ignored is not None
    assert model.scale in ignored
    assert model.linear.weight not in ignored
    assert model.scale in ignored_fsdp_parameters(model)


def test_fsdp_trainable_scalar_parameters_are_rejected() -> None:
    model = _ScalarParamModel()

    with pytest.raises(ValueError, match="trainable scalar"):
        ignored_fsdp_parameters(model)


def test_fsdp_ignored_parameters_include_frozen_dtype_outliers() -> None:
    model = nn.Module()
    model.base = nn.Linear(8, 8).to(dtype=torch.bfloat16)
    model.frozen_fp32 = nn.Linear(2, 2)
    model.frozen_fp32.requires_grad_(False)

    ignored = ignored_fsdp_parameters(model)

    assert ignored is not None
    assert model.frozen_fp32.weight in ignored
    assert model.frozen_fp32.bias in ignored
    assert model.base.weight not in ignored


def test_fsdp_validate_rejects_mixed_trainable_dtypes() -> None:
    model = nn.Module()
    model.fp32 = nn.Linear(2, 2)
    model.bf16 = nn.Linear(2, 2).to(dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="mixed dtypes"):
        validate_fsdp_parameter_group(model)


def test_fsdp_ignores_frozen_dtype_outliers_relative_to_trainable_dtype() -> None:
    model = nn.Module()
    model.frozen_bf16 = nn.Linear(8, 8).to(dtype=torch.bfloat16)
    model.trainable_fp32 = nn.Linear(2, 2)
    model.frozen_bf16.requires_grad_(False)

    ignored = ignored_fsdp_parameters(model)

    assert ignored is not None
    assert model.frozen_bf16.weight in ignored
    assert model.frozen_bf16.bias in ignored
    validate_fsdp_parameter_group(model, ignored_params=ignored)
