# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for backbone prefix adapters used by KI/FM action experts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from praxis.policies.backbones.pali.prefix import PaliGemmaPrefixAdapter
from praxis.policies.backbones.qwen.prefix import (
    QwenPrefixAdapter,
    with_qwen_position_ids,
)
from praxis.policies.backbones.smol.prefix import SmolPrefixAdapter
from praxis.policies.core.vlm.prefix import (
    compact_position_ids_from_mask,
    compact_suffix_position_offsets,
    suffix_position_offsets_from_vlm_inputs,
)


class _FakeSelfAttention(nn.Module):
    def __init__(self, *, hidden_size: int, num_kv_heads: int, head_dim: int) -> None:
        super().__init__()
        self.head_dim = int(head_dim)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)
        self.k_norm = nn.Identity()


class _FakeLayer(nn.Module):
    def __init__(self, *, hidden_size: int, num_kv_heads: int, head_dim: int) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.self_attn = _FakeSelfAttention(
            hidden_size=hidden_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        self.ff = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        mixed = self._attention_mix(
            hidden_states,
            attention_mask=kwargs.get("attention_mask"),
        )
        return hidden_states + self.ff(self.input_layernorm(mixed))

    @staticmethod
    def _attention_mix(
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states
        scores = hidden_states @ hidden_states.transpose(1, 2)
        scores = scores / hidden_states.shape[-1] ** 0.5
        mask = attention_mask.to(device=hidden_states.device)
        if mask.ndim == 4:
            mask = mask[:, 0]
        elif mask.ndim == 2:
            mask = mask[:, None, :].expand_as(scores)
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, -1.0e4)
        else:
            scores = scores + mask.to(dtype=scores.dtype)
        weights = torch.softmax(scores.float(), dim=-1).to(dtype=hidden_states.dtype)
        return weights @ hidden_states


class _FakeTextModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int = 16,
        hidden_size: int = 8,
        num_layers: int = 2,
        num_kv_heads: int = 2,
        head_dim: int = 4,
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [
                _FakeLayer(
                    hidden_size=hidden_size,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.config = SimpleNamespace(
            _attn_implementation="sdpa",
            hidden_size=hidden_size,
            num_attention_heads=2,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
        )
        self.head_dim = int(head_dim)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def rotary_emb(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del position_ids
        cos = torch.ones(
            hidden_states.shape[0],
            hidden_states.shape[1],
            self.head_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        sin = torch.zeros_like(cos)
        return cos, sin


class _FakeSmolInner(nn.Module):
    dtype = torch.float32

    def __init__(self, text_model: _FakeTextModel) -> None:
        super().__init__()
        self.text_model = text_model


class _FakeSmolVlm(nn.Module):
    dtype = torch.float32

    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeSmolInner(_FakeTextModel())

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


class _FakeQwenInner(nn.Module):
    def __init__(self, language_model: _FakeTextModel) -> None:
        super().__init__()
        self.language_model = language_model
        self.rope_deltas: torch.Tensor | None = None
        self.rope_position_offset = 0

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.get_input_embeddings()

    def get_rope_index(
        self,
        *,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor | None,
        video_grid_thw: torch.Tensor | None,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del image_grid_thw, video_grid_thw
        positions = (
            attention_mask.long().cumsum(-1) - 1 + int(self.rope_position_offset)
        )
        positions = positions.clamp_min(0)
        return positions.unsqueeze(0).expand(3, -1, -1), torch.zeros(
            input_ids.shape[0],
            1,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )


class _FakeQwenVlm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeQwenInner(_FakeTextModel())

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


class _FakePaliLanguageModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int = 16,
        hidden_size: int = 8,
        num_layers: int = 2,
        num_kv_heads: int = 2,
        head_dim: int = 4,
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [
                _FakeLayer(
                    hidden_size=hidden_size,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            num_attention_heads=2,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
        )

    def rotary_emb(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = torch.ones(
            position_ids.shape[0],
            position_ids.shape[1],
            self.config.head_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        sin = torch.zeros_like(cos)
        return cos, sin


class _FakePaliInner(nn.Module):
    vocab_size = 16
    last_token_type_ids: torch.Tensor | None = None
    last_is_training: bool | None = None

    def _update_causal_mask(self, *args: Any, **kwargs: Any) -> None:
        del kwargs
        self.last_token_type_ids = args[1]
        self.last_is_training = bool(args[5])

    def get_placeholder_mask(
        self,
        input_ids: torch.Tensor,
        *,
        inputs_embeds: torch.Tensor,
        image_features: torch.Tensor,
    ) -> torch.Tensor:
        del image_features
        return (input_ids == 99).unsqueeze(-1).expand_as(inputs_embeds)


class _FakePaliVlm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _FakePaliLanguageModel()
        self.model = _FakePaliInner()
        self.config = SimpleNamespace(
            image_token_id=99,
            text_config=SimpleNamespace(vocab_size=16),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.embed_tokens

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return torch.ones(
            pixel_values.shape[0],
            1,
            self.language_model.config.hidden_size,
            device=pixel_values.device,
        )


class _FakePeftPaliVlm(nn.Module):
    """Minimal PEFT shape: wrapper.model is the conditional-generation model."""

    def __init__(self, base: _FakePaliVlm) -> None:
        super().__init__()
        self.model = base

    @property
    def config(self) -> Any:
        return self.model.config

    @property
    def language_model(self) -> Any:
        return self.model.language_model

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model.get_image_features(pixel_values)


def _prefix_inputs() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[0, 3, 4], [5, 6, 7]], dtype=torch.long),
        "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]], dtype=torch.long),
    }


def test_smol_prefix_adapter_projects_masked_layer_kv(monkeypatch) -> None:
    monkeypatch.setattr(
        "praxis.policies.backbones.smol.prefix.create_causal_mask",
        lambda **kwargs: None,
    )
    adapter = SmolPrefixAdapter(_FakeSmolVlm())

    state = adapter.prepare_prefix(_prefix_inputs())
    next_state, layer_kv = adapter.forward_layer(state, layer_idx=0)

    assert next_state.hidden.shape == state.hidden.shape
    assert layer_kv.key.shape == (2, 2, 3, 4)
    assert layer_kv.value.shape == (2, 2, 3, 4)
    assert torch.count_nonzero(layer_kv.key[0, :, 0, :]) == 0
    assert torch.count_nonzero(layer_kv.value[0, :, 0, :]) == 0
    assert torch.count_nonzero(layer_kv.key[0, :, 1:, :]) > 0
    assert state.position_ids.tolist() == [[0, 0, 1], [0, 1, 2]]
    assert state.suffix_position_offsets is not None
    assert state.suffix_position_offsets.tolist() == [2, 3]
    assert adapter.num_layers == 2
    assert adapter.hidden_size == 8
    assert adapter.attention_geometry.kv_dim == 8


def test_smol_scheduled_adapter_uses_native_positions() -> None:
    adapter = SmolPrefixAdapter(_FakeSmolVlm())
    inputs = _prefix_inputs()
    layer_attention_mask = torch.zeros(
        2,
        1,
        3,
        3,
        dtype=torch.float32,
    )

    state = adapter.prepare_prefix(inputs, layer_attention_mask=layer_attention_mask)

    assert state.attention_mask is layer_attention_mask
    assert state.position_ids.tolist() == [[0, 1, 2], [0, 1, 2]]
    assert state.suffix_position_offsets is not None
    assert state.suffix_position_offsets.tolist() == [3, 3]
    assert (
        adapter.final_hidden_to_logits(state.hidden).shape[:2] == state.hidden.shape[:2]
    )


def test_pali_prefix_adapter_uses_compact_left_padding_positions() -> None:
    vlm = _FakePaliVlm()
    adapter = PaliGemmaPrefixAdapter(vlm)
    inputs = _prefix_inputs()
    inputs["token_type_ids"] = torch.tensor(
        [[0, 0, 0], [0, 0, 0]],
        dtype=torch.long,
    )

    state = adapter.prepare_prefix(inputs)

    assert state.position_ids.tolist() == [[1, 1, 2], [1, 2, 3]]
    assert state.suffix_position_offsets is not None
    assert state.suffix_position_offsets.tolist() == [3, 4]
    assert vlm.model.last_token_type_ids is not None
    assert torch.equal(vlm.model.last_token_type_ids, inputs["token_type_ids"])
    assert vlm.model.last_is_training is False
    assert adapter.num_layers == 2
    assert adapter.hidden_size == 8
    assert adapter.attention_geometry.kv_dim == 8


def test_pali_scheduled_adapter_uses_native_positions() -> None:
    vlm = _FakePaliVlm()
    adapter = PaliGemmaPrefixAdapter(vlm)
    inputs = _prefix_inputs()
    layer_attention_mask = torch.zeros(
        2,
        1,
        3,
        3,
        dtype=torch.float32,
    )

    state = adapter.prepare_prefix(inputs, layer_attention_mask=layer_attention_mask)

    assert state.position_ids.tolist() == [[1, 2, 3], [1, 2, 3]]
    assert state.suffix_position_offsets is not None
    assert state.suffix_position_offsets.tolist() == [4, 4]
    assert vlm.model.last_token_type_ids is None
    assert vlm.model.last_is_training is None


def test_pali_prefix_adapter_supports_peft_wrapped_image_inputs() -> None:
    base = _FakePaliVlm()
    adapter = PaliGemmaPrefixAdapter(_FakePeftPaliVlm(base))
    inputs = _prefix_inputs()
    inputs["input_ids"][0, 0] = 99
    inputs["pixel_values"] = torch.zeros(2, 3, 2, 2)

    state = adapter.prepare_prefix(inputs)

    assert state.hidden.shape == (2, 3, 8)
    assert base.model.last_is_training is False


def test_qwen_prefix_adapter_projects_masked_layer_kv(monkeypatch) -> None:
    monkeypatch.setattr(
        "praxis.policies.backbones.qwen.prefix.create_causal_mask",
        lambda **kwargs: None,
    )
    adapter = QwenPrefixAdapter(_FakeQwenVlm())

    state = adapter.prepare_prefix(_prefix_inputs())
    next_state, layer_kv = adapter.forward_layer(state, layer_idx=0)

    assert next_state.hidden.shape == state.hidden.shape
    assert layer_kv.key.shape == (2, 2, 3, 4)
    assert layer_kv.value.shape == (2, 2, 3, 4)
    assert torch.count_nonzero(layer_kv.key[0, :, 0, :]) == 0
    assert torch.count_nonzero(layer_kv.value[0, :, 0, :]) == 0
    assert torch.count_nonzero(layer_kv.key[0, :, 1:, :]) > 0
    assert state.position_ids.tolist() == [[0, 0, 1], [0, 1, 2]]
    assert state.suffix_position_offsets is not None
    assert state.suffix_position_offsets.tolist() == [2, 3]
    assert adapter.num_layers == 2
    assert adapter.hidden_size == 8
    assert adapter.attention_geometry.kv_dim == 8


def test_qwen_scheduled_adapter_uses_mask_and_full_expert_positions() -> None:
    vlm = _FakeQwenVlm()
    vlm.model.rope_position_offset = 50
    adapter = QwenPrefixAdapter(vlm)
    inputs = _prefix_inputs()
    layer_attention_mask = torch.zeros(
        2,
        1,
        3,
        3,
        dtype=torch.float32,
    )

    state = adapter.prepare_prefix(inputs, layer_attention_mask=layer_attention_mask)

    assert state.attention_mask is layer_attention_mask
    assert state.position_ids.tolist() == [[49, 50, 51], [50, 51, 52]]
    assert state.suffix_position_offsets is not None
    assert state.suffix_position_offsets.tolist() == [52, 53]
    assert state.extra is not None
    expert_position_ids = state.extra["expert_position_ids"]
    assert expert_position_ids.shape == (3, 2, 3)
    assert (
        adapter.final_hidden_to_logits(state.hidden).shape[:2] == state.hidden.shape[:2]
    )


@pytest.mark.parametrize("family", ["pali", "qwen", "smol"])
def test_scheduled_adapter_logits_match_native_fake_vlm_forward(family: str) -> None:
    inputs = _prefix_inputs()
    layer_attention_mask = torch.zeros(
        2,
        1,
        3,
        3,
        dtype=torch.float32,
    )
    layer_attention_mask[0, :, 2, 0] = torch.finfo(layer_attention_mask.dtype).min

    if family == "pali":
        vlm = _FakePaliVlm()
        adapter = PaliGemmaPrefixAdapter(vlm)
        native_logits = _native_pali_logits(
            vlm,
            inputs,
            layer_attention_mask=layer_attention_mask,
        )
    elif family == "qwen":
        vlm = _FakeQwenVlm()
        adapter = QwenPrefixAdapter(vlm)
        native_logits = _native_text_logits(
            vlm.model.language_model,
            inputs,
            layer_attention_mask=layer_attention_mask,
        )
    elif family == "smol":
        vlm = _FakeSmolVlm()
        adapter = SmolPrefixAdapter(vlm)
        native_logits = _native_text_logits(
            vlm.model.text_model,
            inputs,
            layer_attention_mask=layer_attention_mask,
        )
    else:
        raise AssertionError(f"Unexpected family {family!r}.")

    adapter_logits = _adapter_logits(
        adapter,
        inputs,
        layer_attention_mask=layer_attention_mask,
    )

    torch.testing.assert_close(adapter_logits, native_logits)


@pytest.mark.parametrize("family", ["pali", "qwen", "smol"])
def test_adapter_final_hidden_to_logits_accepts_sparse_rank2_hidden(
    family: str,
) -> None:
    if family == "pali":
        adapter = PaliGemmaPrefixAdapter(_FakePaliVlm())
    elif family == "qwen":
        adapter = QwenPrefixAdapter(_FakeQwenVlm())
    elif family == "smol":
        adapter = SmolPrefixAdapter(_FakeSmolVlm())
    else:
        raise AssertionError(f"Unexpected family {family!r}.")

    hidden = torch.randn(5, adapter.hidden_size)
    candidate_token_ids = torch.tensor([1, 4, 10], dtype=torch.long)

    sparse_logits = adapter.final_hidden_to_logits(hidden)
    candidate_logits = adapter.final_hidden_to_logits(
        hidden,
        candidate_token_ids=candidate_token_ids,
    )
    full_logits = adapter.final_hidden_to_logits(hidden.unsqueeze(0)).squeeze(0)

    assert sparse_logits.shape == (5, 16)
    torch.testing.assert_close(sparse_logits, full_logits)
    torch.testing.assert_close(
        candidate_logits,
        sparse_logits.index_select(-1, candidate_token_ids),
    )


def _adapter_logits(
    adapter: Any,
    inputs: dict[str, torch.Tensor],
    *,
    layer_attention_mask: torch.Tensor,
) -> torch.Tensor:
    state = adapter.prepare_prefix(
        inputs,
        layer_attention_mask=layer_attention_mask,
    )
    for layer_idx in range(adapter.num_layers):
        state = adapter.forward_prefix_layer(state, layer_idx=layer_idx)
    return adapter.final_hidden_to_logits(state.hidden)


def _native_text_logits(
    text_model: _FakeTextModel,
    inputs: dict[str, torch.Tensor],
    *,
    layer_attention_mask: torch.Tensor,
) -> torch.Tensor:
    hidden = text_model.get_input_embeddings()(inputs["input_ids"])
    for layer in text_model.layers:
        hidden = layer(hidden, attention_mask=layer_attention_mask)
    return hidden @ text_model.get_input_embeddings().weight.T


def _native_pali_logits(
    vlm: _FakePaliVlm,
    inputs: dict[str, torch.Tensor],
    *,
    layer_attention_mask: torch.Tensor,
) -> torch.Tensor:
    hidden = vlm.get_input_embeddings()(inputs["input_ids"])
    hidden = hidden * torch.tensor(
        vlm.language_model.config.hidden_size**0.5,
        dtype=hidden.dtype,
        device=hidden.device,
    )
    for layer in vlm.language_model.layers:
        hidden = layer(hidden, attention_mask=layer_attention_mask)
    return hidden @ vlm.get_input_embeddings().weight.T


def test_compact_position_ids_are_padding_invariant() -> None:
    mask = torch.tensor(
        [
            [False, False, True, True, True],
            [True, True, True, False, False],
        ],
        dtype=torch.bool,
    )

    assert compact_position_ids_from_mask(mask, first_position=0).tolist() == [
        [0, 0, 0, 1, 2],
        [0, 1, 2, 2, 2],
    ]
    assert compact_position_ids_from_mask(mask, first_position=1).tolist() == [
        [1, 1, 1, 2, 3],
        [1, 2, 3, 3, 3],
    ]
    assert compact_suffix_position_offsets(mask, first_position=0).tolist() == [3, 3]
    assert compact_suffix_position_offsets(mask, first_position=1).tolist() == [4, 4]


def test_suffix_position_offsets_follow_supplied_position_ids() -> None:
    vlm_inputs = {
        "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]], dtype=torch.long),
        "position_ids": torch.tensor(
            [
                [[1, 1, 2], [1, 2, 3]],
                [[1, 50, 51], [1, 2, 3]],
                [[1, 4, 5], [1, 2, 9]],
            ],
            dtype=torch.long,
        ),
    }

    assert suffix_position_offsets_from_vlm_inputs(vlm_inputs).tolist() == [52, 10]


def test_qwen_prefill_inputs_add_native_mrope_positions() -> None:
    vlm = _FakeQwenVlm()
    vlm.model.rope_position_offset = 50

    out = with_qwen_position_ids(vlm, _prefix_inputs())

    assert out["position_ids"].shape == (3, 2, 3)
    assert suffix_position_offsets_from_vlm_inputs(out).tolist() == [52, 53]
    assert vlm.model.rope_deltas is not None
