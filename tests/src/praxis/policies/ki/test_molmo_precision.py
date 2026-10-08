# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Molmo precision across modules created at different lifecycle stages."""

from types import SimpleNamespace

import torch
from torch import nn

from praxis.policies.core.action_expert.flow import sample_flow_training_point
from praxis.policies.ki.backbones.molmo.molmo_base import MolmoPolicyBase
from praxis.policies.ki.backbones.molmo.molmo_config import MolmoPolicyConfig
from praxis.policies.ki.core.policy import PolicyBase
from praxis.training.core.ema import EMAModel
from praxis.training.policy_trainer import PolicyTrainer
from praxis.policies.backbones.molmo.vlm import (
    configure_molmo_vision_precision,
    configure_molmo_vision_checkpointing,
)


def test_vision_checkpointing_preserves_gradients_and_recomputes() -> None:
    transformer = nn.Module()
    transformer.resblocks = nn.ModuleList([
        nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4))
        for _ in range(2)
    ])
    model = SimpleNamespace(model=SimpleNamespace(
        vision_backbone=SimpleNamespace(image_vit=SimpleNamespace(transformer=transformer))
    ))
    x = torch.randn(2, 4)  # Images need not require gradients.
    configure_molmo_vision_checkpointing(model, False)
    expected = transformer(x)
    sum(v.sum() for v in expected).backward()
    grads = [p.grad.clone() for p in transformer.parameters()]
    transformer.zero_grad(set_to_none=True)
    configure_molmo_vision_checkpointing(model, True)
    calls = []
    handle = transformer.resblocks[0][0].register_forward_pre_hook(
        lambda *args: calls.append(1)
    )
    actual = transformer(x)
    sum(v.sum() for v in actual).backward()
    handle.remove()
    assert len(calls) == 2
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    for p, grad in zip(transformer.parameters(), grads):
        torch.testing.assert_close(p.grad, grad)


def test_all_molmo_vision_attention_uses_bf16() -> None:
    vision = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    for module in vision:
        module.float32_attention = True
    config = SimpleNamespace(
        vit_config=SimpleNamespace(float32_attention=True),
        adapter_config=SimpleNamespace(float32_attention=True),
    )
    model = SimpleNamespace(model=SimpleNamespace(vision_backbone=vision), config=config)
    configure_molmo_vision_precision(model, torch.bfloat16)
    assert all(p.dtype == torch.bfloat16 for p in vision.parameters())
    assert all(not m.float32_attention for m in vision)
    assert not config.vit_config.float32_attention
    assert not config.adapter_config.float32_attention


def test_molmo_casts_initial_and_io_created_modules(monkeypatch) -> None:
    def initialize(self, config, **kwargs):
        nn.Module.__init__(self)
        self.config = config
        self.vlm = nn.Linear(4, 4)
        self.action_tokenizer = nn.Linear(4, 4)
        self.lora_adapter = nn.Linear(4, 4)
        self.register_buffer("token_ids", torch.arange(4))

    def build_io_modules(self):
        self.action_expert = nn.Linear(4, 4)
        self.state_projector = nn.Linear(4, 4)
        self.normalizer = nn.Module()
        self.normalizer.register_buffer("mean", torch.zeros(4))

    monkeypatch.setattr(PolicyBase, "__init__", initialize)
    monkeypatch.setattr(PolicyBase, "_post_set_io", build_io_modules)
    policy = MolmoPolicyBase(MolmoPolicyConfig())
    assert all(p.dtype == torch.bfloat16 for p in policy.parameters())

    # Replacing IO modules must not reintroduce FP32 on repeated setup.
    for _ in range(2):
        policy._post_set_io()
        assert all(p.dtype == torch.bfloat16 for p in policy.parameters())
        assert policy.normalizer.mean.dtype == torch.bfloat16
        assert policy.token_ids.dtype == torch.int64


def test_molmo_action_targets_use_bf16_without_mutating_batch() -> None:
    policy = MolmoPolicyBase.__new__(MolmoPolicyBase)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(vlm_dtype="bfloat16")
    policy.io = None
    actions = torch.randn(2, 3, 4)
    batch = {"action": actions}

    converted = policy._actions_from_batch(batch)

    assert converted.dtype == torch.bfloat16
    assert batch["action"].dtype == torch.float32
    torch.testing.assert_close(converted.float(), actions, rtol=0.004, atol=0.004)


def test_default_flow_sampling_uses_bf16_without_gamma_sampler(monkeypatch) -> None:
    def unsupported_beta(*args, **kwargs):
        raise AssertionError("Default BF16 sampling must not use FP32 Beta kernels")

    monkeypatch.setattr(torch.distributions, "Beta", unsupported_beta)
    sample = sample_flow_training_point(
        torch.zeros(4096, 2, 3, dtype=torch.bfloat16),
        sampling="beta",
        beta_alpha=1.5,
        beta_beta=1.0,
        min_time=0.001,
    )
    for value in (
        sample.noise,
        sample.noisy_actions,
        sample.target_velocity,
        sample.timesteps,
    ):
        assert value.dtype == torch.bfloat16
        assert torch.isfinite(value).all()
    assert sample.timesteps.min() >= 0.001
    assert sample.timesteps.max() <= 1.0
    assert abs(sample.timesteps.float().mean().item() - 0.6004) < 0.025


def test_molmo_ema_uses_configured_dtype() -> None:
    policy = MolmoPolicyBase.__new__(MolmoPolicyBase)
    nn.Module.__init__(policy)
    policy.config = MolmoPolicyConfig()
    policy.vlm = nn.Linear(4, 4).to(dtype=torch.bfloat16)
    trainer = PolicyTrainer.__new__(PolicyTrainer)
    trainer.model = policy
    ema = EMAModel(policy, shadow_dtype=trainer._ema_shadow_dtype())
    assert all(value.dtype == torch.bfloat16 for value in ema.state_dict().values())
    policy.config.vlm_dtype = "float32"
    assert trainer._ema_shadow_dtype() == torch.float32


def test_molmo_casts_replaced_normalization_stats(monkeypatch) -> None:
    def install_stats(self, stats, modes=None):
        self.normalize = nn.Module()
        self.normalize.register_buffer("mean", stats["action"]["mean"].clone())
        self.unnormalize_action = nn.Module()
        self.unnormalize_action.register_buffer("mean", stats["action"]["mean"].clone())

    monkeypatch.setattr(PolicyBase, "set_normalization_stats", install_stats)
    policy = MolmoPolicyBase.__new__(MolmoPolicyBase)
    nn.Module.__init__(policy)
    policy.config = MolmoPolicyConfig()
    stats = {"action": {"mean": torch.zeros(4)}}
    for _ in range(2):
        policy.set_normalization_stats(stats)
        assert policy.normalize.mean.dtype == torch.bfloat16
        assert policy.unnormalize_action.mean.dtype == torch.bfloat16
    assert stats["action"]["mean"].dtype == torch.float32
