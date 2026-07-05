# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import cast

import pytest
import torch
import torch.nn as nn

from praxis.policies.bar.core.vlm.base import VlmActionTokenPolicyBase
from praxis.policies.core.vlm.fsdp import (
    select_trainable_leaf_modules,
    select_vlm_block_modules,
)
from praxis.policies.core.vlm.lora import apply_lora_to_vlm


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _NoForwardBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))


class _VlmLike(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Block(), _Block()])
        self.no_forward = nn.ModuleList([_NoForwardBlock()])
        self.activations = nn.ModuleList([nn.ReLU()])


class _LoRABlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(2, 2).to(dtype=torch.bfloat16)
        self.adapter = nn.Linear(2, 2)
        self.base.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x.to(torch.bfloat16)).to(torch.float32) + self.adapter(x)


class _LoRAVlmLike(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_LoRABlock()])


class _PromptLeaf(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prompt = nn.Parameter(torch.ones(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.prompt


class _LoadStatePolicy(VlmActionTokenPolicyBase):
    def __init__(self, *, use_lora: bool = False) -> None:
        nn.Module.__init__(self)
        self.config = SimpleNamespace(use_lora=use_lora)
        self.weight = nn.Parameter(torch.zeros(2))
        if use_lora:
            self.vlm = nn.Module()
            self.vlm.frozen = nn.Linear(2, 2)
            self.vlm.adapter = nn.Linear(2, 2)
            self.vlm.frozen.requires_grad_(False)


def _empty_vlm_policy(*, vlm_dtype: str = "bfloat16") -> VlmActionTokenPolicyBase:
    policy = cast(
        VlmActionTokenPolicyBase,
        VlmActionTokenPolicyBase.__new__(VlmActionTokenPolicyBase),
    )
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(use_lora=False, vlm_dtype=vlm_dtype)
    policy.vlm = nn.Linear(2, 2)
    return policy


def test_select_vlm_block_modules_selects_hf_style_blocks() -> None:
    vlm = _VlmLike()

    assert select_vlm_block_modules(vlm) == tuple(vlm.layers)


def test_vlm_policy_exposes_fsdp_wrap_modules_from_vlm() -> None:
    vlm = _VlmLike()
    policy = cast(
        VlmActionTokenPolicyBase,
        VlmActionTokenPolicyBase.__new__(VlmActionTokenPolicyBase),
    )
    nn.Module.__init__(policy)
    policy.vlm = vlm

    assert policy.fsdp_wrap_modules() == tuple(vlm.layers)


def test_vlm_policy_declares_fsdp_capabilities() -> None:
    policy = cast(
        VlmActionTokenPolicyBase,
        VlmActionTokenPolicyBase.__new__(VlmActionTokenPolicyBase),
    )
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(use_lora=False, freeze_action_tokenizer=True)
    policy.vlm = _VlmLike()

    assert policy.fsdp_forward_methods() == ("predict_action", "predict_action_chunk")
    assert policy.supports_training_strategy("fsdp") is True
    policy.validate_distributed_strategy("fsdp")


def test_vlm_policy_fsdp_rejects_trainable_action_tokenizer() -> None:
    policy = cast(
        VlmActionTokenPolicyBase,
        VlmActionTokenPolicyBase.__new__(VlmActionTokenPolicyBase),
    )
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(use_lora=False, freeze_action_tokenizer=False)
    policy.vlm = _VlmLike()

    with pytest.raises(ValueError, match="freeze_action_tokenizer=true"):
        policy.validate_distributed_strategy("fsdp")


def test_lora_policy_fsdp_wraps_trainable_leaf_modules_before_blocks() -> None:
    policy = cast(
        VlmActionTokenPolicyBase,
        VlmActionTokenPolicyBase.__new__(VlmActionTokenPolicyBase),
    )
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(use_lora=True)
    policy.vlm = _LoRAVlmLike()

    wrapped = policy.fsdp_wrap_modules()

    assert wrapped == (policy.vlm.layers[0].adapter, policy.vlm.layers[0])
    assert select_trainable_leaf_modules(policy.vlm) == (policy.vlm.layers[0].adapter,)


def test_lora_policy_fsdp_wraps_trainable_non_weight_leaf_modules() -> None:
    vlm = nn.Module()
    vlm.prompt = _PromptLeaf()

    assert select_trainable_leaf_modules(vlm) == (vlm.prompt,)


def test_vlm_autocast_context_uses_configured_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, torch.dtype]] = []

    class _AutocastProbe:
        def __enter__(self):
            return None

        def __exit__(self, *args):
            return False

    def _fake_autocast(*, device_type: str, dtype: torch.dtype):
        calls.append((device_type, dtype))
        return _AutocastProbe()

    monkeypatch.setattr(torch, "autocast", _fake_autocast)
    with _empty_vlm_policy(vlm_dtype="float16")._autocast_context():
        pass

    assert calls == [("cpu", torch.float16)]


def test_vlm_autocast_context_skips_float32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _unexpected_autocast(**kwargs):
        raise AssertionError(f"unexpected autocast call: {kwargs}")

    monkeypatch.setattr(torch, "autocast", _unexpected_autocast)
    with _empty_vlm_policy(vlm_dtype="float32")._autocast_context():
        pass


def test_vlm_policy_load_state_dict_accepts_assign_kwarg() -> None:
    policy = _LoadStatePolicy()

    policy.load_state_dict({"weight": torch.ones(2)}, assign=True)

    torch.testing.assert_close(policy.weight, torch.ones(2))


def test_lora_state_dict_keeps_trainable_adapter_for_fsdp_export() -> None:
    policy = _LoadStatePolicy(use_lora=True)

    state = policy.state_dict()

    assert "vlm.frozen.weight" not in state
    assert "vlm.adapter.weight" in state
    reloaded = _LoadStatePolicy(use_lora=True)
    reloaded.load_state_dict(state, strict=True)


def test_lora_requires_vocab_modules_to_save(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(LoraConfig=object, get_peft_model=lambda model, _cfg: model),
    )

    with pytest.raises(ValueError, match="embed_tokens"):
        apply_lora_to_vlm(
            vlm=nn.Linear(2, 2),
            lora_r=8,
            lora_alpha=16,
            lora_target_modules=("q_proj",),
            lora_dropout=0.0,
            lora_modules_to_save=("lm_head",),
        )


def test_lora_tied_embeddings_save_embedding_module_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _TiedVlm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = nn.Embedding(4, 2)
            self.lm_head = nn.Linear(2, 4, bias=False)
            self.lm_head.weight = self.embed_tokens.weight

        def get_input_embeddings(self):
            return self.embed_tokens

        def get_output_embeddings(self):
            return self.lm_head

    captured: dict[str, object] = {}

    class _FakeLoraConfig:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(
            LoraConfig=_FakeLoraConfig,
            get_peft_model=lambda model, _cfg: model,
        ),
    )

    apply_lora_to_vlm(
        vlm=_TiedVlm(),
        lora_r=8,
        lora_alpha=16,
        lora_target_modules=("q_proj",),
        lora_dropout=0.0,
        lora_modules_to_save=("lm_head", "embed_tokens"),
    )

    assert captured["modules_to_save"] == ["embed_tokens"]
