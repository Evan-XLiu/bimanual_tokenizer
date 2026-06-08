# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for end-to-end FM policy semantics."""

from __future__ import annotations

import re
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import torch
import torch.nn as nn

import praxis.policies.core.action_expert.integrated as integrated_module
import praxis.policies.core.action_expert.policy as action_expert_policy_module
from praxis.core.types import FeatureSpec, FeatureType, NormMode, PolicyIO
from praxis.policies.core.action_expert.prefix_cache import PrefixCache
from praxis.policies.core.state_prefix import (
    STATE_PREFIX_EMBEDDINGS_KEY,
    STATE_PREFIX_PLACEHOLDER_MASK_KEY,
    replace_state_prefix_placeholder_embeddings,
)
from praxis.policies.core.vlm.prefix import (
    LayerKV,
    VlmAttentionGeometry,
    VlmPrefixState,
)
from praxis.policies.fm.backbones.qwen.qwen_base import QwenPolicyBase
from praxis.policies.fm.core.config import PolicyConfig
from praxis.policies.fm.core.policy import PolicyBase
from praxis.training.core.fsdp import validate_fsdp_parameter_group


class _FakeTokenizer:
    def __init__(self, base_size: int = 8) -> None:
        self.vocab = {f"<tok_{i}>": i for i in range(base_size)}

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)

    def add_special_tokens(self, payload: dict[str, list[str]]) -> int:
        added = 0
        for token in payload["additional_special_tokens"]:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                added += 1
        return added


class _FakeVLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int = 8,
        hidden_size: int = 12,
        num_layers: int = 2,
        num_heads: int = 4,
        num_kv_heads: int = 2,
        head_dim: int = 4,
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [
                _FakeDecoderLayer(
                    hidden_size=hidden_size,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(
                hidden_size=hidden_size,
                num_hidden_layers=num_layers,
                num_attention_heads=num_heads,
                num_key_value_heads=num_kv_heads,
                head_dim=head_dim,
            )
        )
        self.forward_calls = 0
        self.last_labels: torch.Tensor | None = None
        self.last_use_cache: bool | None = None
        self.last_position_ids: torch.Tensor | None = None
        self.gradient_checkpointing_kwargs: dict[str, Any] | None = None
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    @property
    def device(self) -> torch.device:
        return self.embed_tokens.weight.device

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        self.gradient_checkpointing_kwargs = dict(kwargs)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def resize_token_embeddings(
        self, size: int, *, mean_resizing: bool = False
    ) -> nn.Embedding:
        del mean_resizing
        old = self.embed_tokens
        new = nn.Embedding(int(size), old.embedding_dim, device=old.weight.device)
        with torch.no_grad():
            keep = min(old.num_embeddings, int(size))
            new.weight[:keep].copy_(old.weight[:keep])
        self.embed_tokens = new
        return self.embed_tokens

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        use_cache: bool = False,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> SimpleNamespace:
        del return_dict
        self.last_position_ids = kwargs.get("position_ids")
        self.forward_calls += 1
        self.last_labels = labels
        self.last_use_cache = bool(use_cache)
        hidden = self.embed_tokens(input_ids)
        layer_inputs = []
        for layer in self.layers:
            layer_inputs.append(hidden)
            hidden = layer(hidden)
        past_key_values = None
        if use_cache:
            layers = []
            for layer, layer_hidden in zip(self.layers, layer_inputs, strict=True):
                decoder_layer = cast(_FakeDecoderLayer, layer)
                normed = decoder_layer.input_layernorm(layer_hidden)
                key = decoder_layer.self_attn.k_proj(normed).view(
                    hidden.shape[0],
                    hidden.shape[1],
                    self.num_kv_heads,
                    self.head_dim,
                )
                value = decoder_layer.self_attn.v_proj(normed).view(
                    hidden.shape[0],
                    hidden.shape[1],
                    self.num_kv_heads,
                    self.head_dim,
                )
                layers.append((key.transpose(1, 2), value.transpose(1, 2)))
            past_key_values = tuple(layers)
        return SimpleNamespace(past_key_values=past_key_values)


class _FakeSelfAttention(nn.Module):
    def __init__(self, *, hidden_size: int, num_kv_heads: int, head_dim: int) -> None:
        super().__init__()
        self.head_dim = int(head_dim)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)


class _FakeDecoderLayer(nn.Module):
    def __init__(self, *, hidden_size: int, num_kv_heads: int, head_dim: int) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.self_attn = _FakeSelfAttention(
            hidden_size=hidden_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        self.ff = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.ff(self.input_layernorm(hidden_states))


class _DummyPolicy(PolicyBase):
    CONFIG_CLASS = PolicyConfig

    def _load_vlm(self, model_name: str, dtype_str: str) -> None:
        del model_name, dtype_str
        self.vlm = _FakeVLM()
        self.processor = SimpleNamespace(tokenizer=_FakeTokenizer())

    def _build_vlm_inputs(
        self,
        obs: dict[str, torch.Tensor],
        *,
        instructions: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        del instructions
        first_tensor = next(
            value for value in obs.values() if isinstance(value, torch.Tensor)
        )
        batch_size = int(first_tensor.shape[0])
        rows: list[list[int]] = []
        state_prefix = self._prepare_state_prefix(obs)
        vocab = self.processor.tokenizer.get_vocab()
        for row in range(batch_size):
            ids = [1, 2, 3]
            if row < len(state_prefix.texts):
                ids.extend(
                    int(vocab[token])
                    for token in re.findall(r"<\|[^|]+\|>", state_prefix.texts[row])
                )
            rows.append(ids)
        input_ids = torch.tensor(rows, dtype=torch.long)
        out = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }
        return self._add_continuous_state_prefix_embeddings(out, state_prefix)

    def _build_vlm_prefix_adapter(self) -> _FakePrefixAdapter:
        return _FakePrefixAdapter(self.vlm)


class _FakeQwenMropeModel:
    def __init__(self) -> None:
        self.rope_deltas: torch.Tensor | None = None

    def get_rope_index(
        self,
        *,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor | None,
        video_grid_thw: torch.Tensor | None,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del input_ids, image_grid_thw, video_grid_thw
        positions = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)
        positions = positions.unsqueeze(0).expand(3, -1, -1).clone()
        positions[1] = positions[1] + 50
        return positions, torch.zeros(
            attention_mask.shape[0],
            1,
            dtype=torch.long,
            device=attention_mask.device,
        )


class _FakeQwenMropeVLM(_FakeVLM):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeQwenMropeModel()


class _QwenPrefillPolicy(QwenPolicyBase):
    def _load_vlm(self, model_name: str, dtype_str: str) -> None:
        del model_name, dtype_str
        self.vlm = _FakeQwenMropeVLM()
        self.processor = object()

    def _build_vlm_prefix_adapter(self) -> _FakePrefixAdapter:
        return _FakePrefixAdapter(self.vlm)

    def _build_vlm_inputs(
        self,
        obs: dict[str, torch.Tensor],
        *,
        instructions: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        del instructions
        first_tensor = next(
            value for value in obs.values() if isinstance(value, torch.Tensor)
        )
        batch_size = int(first_tensor.shape[0])
        input_ids = torch.tensor(
            [[0, 1, 2, 3], [4, 5, 6, 7]],
            dtype=torch.long,
        )[:batch_size]
        attention_mask = torch.tensor(
            [[0, 1, 1, 1], [1, 1, 1, 1]],
            dtype=torch.long,
        )[:batch_size]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_grid_thw": torch.ones((batch_size, 3), dtype=torch.long),
            "pixel_values": torch.zeros((batch_size, 4), dtype=torch.float32),
        }


class _FakePrefixAdapter:
    def __init__(self, vlm: _FakeVLM) -> None:
        self.vlm = vlm

    @property
    def num_layers(self) -> int:
        return len(self.vlm.layers)

    @property
    def hidden_size(self) -> int:
        return int(self.vlm.config.text_config.hidden_size)

    @property
    def attention_geometry(self) -> VlmAttentionGeometry:
        cfg = self.vlm.config.text_config
        return VlmAttentionGeometry(
            num_attention_heads=int(cfg.num_attention_heads),
            num_key_value_heads=int(cfg.num_key_value_heads),
            head_dim=int(cfg.head_dim),
        )

    def prepare_prefix(
        self,
        vlm_inputs: dict[str, torch.Tensor],
        *,
        layer_attention_mask: torch.Tensor | None = None,
    ) -> VlmPrefixState:
        del layer_attention_mask
        input_ids = vlm_inputs["input_ids"]
        mask = vlm_inputs["attention_mask"].bool()
        hidden = self.vlm.embed_tokens(input_ids)
        hidden = replace_state_prefix_placeholder_embeddings(
            input_ids=input_ids,
            inputs_embeds=hidden,
            vlm_inputs=vlm_inputs,
        )
        positions = (
            torch.arange(
                int(input_ids.shape[1]),
                device=input_ids.device,
                dtype=torch.long,
            )
            .unsqueeze(0)
            .expand(int(input_ids.shape[0]), -1)
        )
        return VlmPrefixState(
            hidden=hidden,
            mask=mask,
            attention_mask=None,
            position_ids=positions,
            cache_position=positions.squeeze(0),
            position_embeddings=(hidden, hidden),
        )

    def forward_layer(
        self, state: VlmPrefixState, *, layer_idx: int
    ) -> tuple[VlmPrefixState, LayerKV]:
        layer = cast(_FakeDecoderLayer, self.vlm.layers[layer_idx])
        normed = layer.input_layernorm(state.hidden)
        batch_size, seq_len, _ = normed.shape
        key = layer.self_attn.k_proj(normed).view(
            batch_size,
            seq_len,
            self.vlm.num_kv_heads,
            self.vlm.head_dim,
        )
        value = layer.self_attn.v_proj(normed).view(
            batch_size,
            seq_len,
            self.vlm.num_kv_heads,
            self.vlm.head_dim,
        )
        return (
            self.forward_prefix_layer(state, layer_idx=layer_idx),
            LayerKV(
                key=key.transpose(1, 2),
                value=value.transpose(1, 2),
            ),
        )

    def forward_prefix_layer(
        self,
        state: VlmPrefixState,
        *,
        layer_idx: int,
    ) -> VlmPrefixState:
        layer = cast(_FakeDecoderLayer, self.vlm.layers[layer_idx])
        hidden = layer(state.hidden)
        return VlmPrefixState(
            hidden=hidden,
            mask=state.mask,
            attention_mask=state.attention_mask,
            position_ids=state.position_ids,
            cache_position=state.cache_position,
            position_embeddings=state.position_embeddings,
        )


def _io() -> PolicyIO:
    return PolicyIO(
        obs_features=[
            FeatureSpec(
                key="observation.state",
                type=FeatureType.STATE,
                shape=(2,),
                norm_mode=NormMode.MIN_MAX,
            )
        ],
        action_feature=FeatureSpec(
            key="action",
            type=FeatureType.ACTION,
            shape=(3,),
            norm_mode=NormMode.MIN_MAX,
        ),
        n_obs_steps=1,
        n_action_steps=2,
    )


def _batch() -> dict[str, torch.Tensor]:
    return {
        "observation.state": torch.zeros((2, 2), dtype=torch.float32),
        "action": torch.randn((2, 4, 3), dtype=torch.float32),
    }


def _policy() -> _DummyPolicy:
    policy = _DummyPolicy(
        config=PolicyConfig(
            action_dim=3,
            action_horizon=4,
            n_action_steps=2,
            expert_hidden_size=16,
        ).to_dict()
    )
    policy.set_io(_io())
    return policy


def _has_nonzero_grad(module: nn.Module) -> bool:
    return any(
        param.grad is not None and bool(param.grad.detach().abs().sum().item() > 0)
        for param in module.parameters()
    )


def test_default_expert_width_uses_vlm_multiplier() -> None:
    policy = _DummyPolicy(
        config=PolicyConfig(
            action_dim=3,
            action_horizon=4,
            n_action_steps=2,
            expert_hidden_size=None,
            expert_width_multiplier=0.75,
        ).to_dict()
    )
    policy.set_io(_io())

    expert = policy._require_action_expert()
    assert expert.hidden_size == 9


def test_gradient_checkpointing_is_allowed_with_cache_free_train_forward() -> None:
    policy = _DummyPolicy(
        config=PolicyConfig(
            action_dim=3,
            action_horizon=4,
            n_action_steps=2,
            expert_hidden_size=16,
            gradient_checkpointing=True,
        ).to_dict()
    )

    assert policy.vlm.gradient_checkpointing_kwargs == {
        "gradient_checkpointing_kwargs": {"use_reentrant": False}
    }


def test_runtime_refresh_does_not_rebuild_loaded_action_expert() -> None:
    policy = _policy()
    expert = policy._require_action_expert()
    original_weight = expert.out_proj.weight.detach().clone()

    policy.config.n_action_steps = 1
    policy.refresh_after_config_update(frozenset({"n_action_steps"}))

    assert policy._require_action_expert() is expert
    assert torch.equal(expert.out_proj.weight, original_weight)
    assert policy._action_queue.maxlen == 1


def test_gradient_checkpointing_covers_integrated_fm_train_forward(monkeypatch) -> None:
    calls = 0

    def _checkpoint(
        function: Callable[..., tuple[torch.Tensor, torch.Tensor]],
        *args: torch.Tensor,
        use_reentrant: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal calls
        assert use_reentrant is False
        calls += 1
        return function(*args)

    monkeypatch.setattr(integrated_module, "checkpoint", _checkpoint)
    policy = _DummyPolicy(
        config=PolicyConfig(
            action_dim=3,
            action_horizon=4,
            n_action_steps=2,
            expert_hidden_size=16,
            gradient_checkpointing=True,
        ).to_dict()
    )
    policy.set_io(_io())

    loss, info = policy.forward(_batch())
    loss.backward()

    assert info["flow_loss"] > 0.0
    assert calls == policy._require_action_expert().vlm_num_layers
    assert _has_nonzero_grad(policy.vlm)
    assert policy.action_expert is not None
    assert _has_nonzero_grad(policy.action_expert)


def test_fm_loss_updates_vlm_and_action_expert() -> None:
    policy = _policy()

    loss, info = policy.forward(_batch())
    loss.backward()

    assert info["flow_loss"] > 0.0
    assert "ce_loss" not in info
    assert "loss" not in info
    assert _has_nonzero_grad(policy.vlm)
    assert policy.action_expert is not None
    assert _has_nonzero_grad(policy.action_expert)
    assert policy.vlm.forward_calls == 0


def test_from_io_uses_continuous_state_prefix_by_default() -> None:
    policy = _DummyPolicy.from_io(
        _io(),
        expert_hidden_size=16,
    )

    assert policy.config.state_prefix_encoding == "continuous_projection"


def test_fm_flow_loss_masks_padded_action_steps() -> None:
    policy = _policy()
    batch = _batch()
    batch["action_is_pad"] = torch.ones((2, 4), dtype=torch.bool)

    loss, info = policy.forward(batch)

    assert torch.isfinite(loss)
    assert info["flow_loss"] == 0.0
    assert loss.requires_grad


def test_fm_predict_action_chunk_uses_continuous_expert() -> None:
    policy = _policy()
    actions, info = policy.predict_action_chunk(
        {"observation.state": torch.zeros((1, 1, 2), dtype=torch.float32)},
        generator=torch.Generator().manual_seed(0),
    )

    assert actions.shape == (1, 4, 3)
    assert info["flow_sampling_steps"] == policy.config.flow_sampling_steps


def test_fm_continuous_state_prefix_updates_projector() -> None:
    policy = _DummyPolicy(
        config=PolicyConfig(
            action_dim=3,
            action_horizon=4,
            n_action_steps=2,
            expert_hidden_size=16,
            state_prefix_encoding="continuous_projection",
        ).to_dict()
    )
    policy.set_io(_io())
    batch = _batch()
    batch["observation.state"] = torch.tensor(
        [[-0.5, 0.25], [0.75, -0.25]],
        dtype=torch.float32,
    )

    vlm_inputs = policy._build_vlm_inputs_for_action_expert(
        batch,
        instructions=None,
    )
    counts = vlm_inputs["input_ids"].eq(policy.state_prefix_token_id).sum(dim=1)
    assert counts.tolist() == [1, 1]
    assert STATE_PREFIX_EMBEDDINGS_KEY in vlm_inputs
    assert torch.equal(
        vlm_inputs[STATE_PREFIX_PLACEHOLDER_MASK_KEY],
        vlm_inputs["input_ids"].eq(policy.state_prefix_token_id),
    )

    loss, info = policy.forward(batch)
    loss.backward()

    assert info["flow_loss"] > 0.0
    assert policy.state_prefix_projector is not None
    grad = policy.state_prefix_projector.weight.grad
    assert grad is not None
    assert grad.abs().sum().item() > 0.0


def test_fm_fsdp_wrap_modules_include_state_prefix_projector() -> None:
    policy = _policy()

    assert policy.state_prefix_projector is not None
    assert policy.state_prefix_projector in policy.fsdp_wrap_modules()


def test_fm_fsdp_root_group_excludes_state_prefix_projector() -> None:
    policy = _policy()
    policy.vlm.embed_tokens.to(dtype=torch.bfloat16)
    assigned_params = {
        param for module in policy.fsdp_wrap_modules() for param in module.parameters()
    }

    validate_fsdp_parameter_group(policy, excluded_params=assigned_params)


def test_qwen_predict_action_chunk_uses_native_mrope_suffix_offsets(
    monkeypatch,
) -> None:
    captured: dict[str, torch.Tensor | None] = {}

    def _sample_actions_euler(
        expert,
        *,
        cache: PrefixCache,
        action_dim: int,
        action_horizon: int,
        steps: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        del expert, steps, generator
        captured["suffix_position_offsets"] = cache.suffix_position_offsets
        return torch.zeros((2, action_horizon, action_dim), dtype=torch.float32)

    monkeypatch.setattr(
        action_expert_policy_module,
        "sample_actions_euler",
        _sample_actions_euler,
    )
    policy = _QwenPrefillPolicy(
        config=PolicyConfig(
            action_dim=3,
            action_horizon=4,
            n_action_steps=2,
            expert_hidden_size=16,
            state_prefix_encoding="none",
        ).to_dict()
    )
    policy.set_io(_io())

    actions, _info = policy.predict_action_chunk(
        {"observation.state": torch.zeros((2, 1, 2), dtype=torch.float32)}
    )

    assert actions.shape == (2, 4, 3)
    assert policy.vlm.last_position_ids is not None
    assert policy.vlm.last_position_ids.shape == (3, 2, 4)
    assert captured["suffix_position_offsets"] is not None
    assert captured["suffix_position_offsets"].tolist() == [53, 54]
