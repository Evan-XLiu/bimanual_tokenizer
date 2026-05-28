"""Focused tests for KI co-training semantics."""

from __future__ import annotations

import re
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import praxis.policies.core.vlm.training as vlm_training_module
import praxis.policies.ki.core.policy as policy_module
from praxis.core.types import FeatureSpec, FeatureType, NormMode, PolicyIO
from praxis.policies.bar.variants import (
    OATPaliPolicy as BarOATPaliPolicy,
    OATQwenPolicy as BarOATQwenPolicy,
    OATSmolPolicy as BarOATSmolPolicy,
)
from praxis.policies.core.action_expert import integrated as integrated_module
from praxis.policies.core.action_expert.flow import FlowSample
from praxis.policies.core.action_tokens import VariableLengthActionTokenMixin
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
from praxis.policies.core.vlm.supervision import IGNORE_INDEX
from praxis.policies.core.vlm.training import (
    build_action_token_ce_candidate_token_ids,
    build_prepared_action_token_attention_mask,
    compute_prepared_action_token_ce_loss,
    compute_prepared_action_token_ce_loss_from_hidden,
    prepare_action_token_ce,
    prepare_action_token_ce_from_token_targets,
    run_action_token_ce_prefix_forward,
)
from praxis.policies.ki.core.config import PolicyConfig
from praxis.policies.ki.core.policy import (
    PolicyBase,
    _context_mask_matching_fm_prefix,
)
from praxis.policies.ki.variants import (
    OATPaliPolicy as KIOATPaliPolicy,
    OATQwenPolicy as KIOATQwenPolicy,
    OATSmolPolicy as KIOATSmolPolicy,
)
from praxis.training.core.fsdp import validate_fsdp_parameter_group
from praxis.training.core.training_context import TrainingContext


class _FakeTokenizer:
    def __init__(self, base_size: int = 8) -> None:
        self.vocab = {f"<tok_{i}>": i for i in range(base_size)}
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.padding_side = "left"

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)

    def add_special_tokens(self, payload: dict[str, list[str]]) -> int:
        tokens = payload["additional_special_tokens"]
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
        return len(tokens)


class _FakeBatchEncoding(dict):
    def to(self, device):
        for key, value in list(self.items()):
            if isinstance(value, torch.Tensor):
                self[key] = value.to(device)
        return self


class _FakePaliProcessor:
    image_token = "<image>"

    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        *,
        text: list[str] | str | None,
        suffix: list[str] | str | None,
        images: list[list[torch.Tensor]],
        padding: bool,
        return_tensors: str,
    ) -> _FakeBatchEncoding:
        del padding, return_tensors
        bsz = len(images)
        texts = text if isinstance(text, list) else [str(text)] * bsz
        suffixes = (
            None
            if suffix is None
            else suffix
            if isinstance(suffix, list)
            else [str(suffix)] * bsz
        )
        self.calls.append({"text": list(texts), "suffix": suffixes})
        rows: list[list[int]] = []
        token_pattern = re.compile(r"<\|[^|]+\|>")
        vocab = self.tokenizer.get_vocab()
        for row in range(bsz):
            ids = [1, 2, 3]
            if suffixes is not None:
                ids.extend(
                    int(vocab[token]) for token in token_pattern.findall(suffixes[row])
                )
                ids.append(int(self.tokenizer.eos_token_id))
            rows.append(ids)
        max_len = max(len(row) for row in rows)
        input_ids = torch.full(
            (bsz, max_len), self.tokenizer.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros_like(input_ids)
        for row_idx, ids in enumerate(rows):
            start = max_len - len(ids)
            input_ids[row_idx, start:] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row_idx, start:] = 1
        out = _FakeBatchEncoding(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": torch.zeros(bsz, 3, 4, 4),
            }
        )
        if suffixes is not None:
            out["labels"] = input_ids.clone()
        return out


class _FakeChatTokenizer(_FakeTokenizer):
    def __init__(self, base_size: int = 8) -> None:
        super().__init__(base_size=base_size)
        self.last_text: list[str] | None = None

    def __call__(
        self,
        text: list[str] | str,
        padding: bool = True,
        return_tensors: str = "pt",
    ) -> _FakeBatchEncoding:
        del padding, return_tensors
        texts = text if isinstance(text, list) else [str(text)]
        self.last_text = list(texts)
        vocab = self.get_vocab()
        rows: list[list[int]] = []
        token_pattern = re.compile(r"<\|[^|]+\|>")
        for sample in texts:
            ids = [1, 2, 3]
            ids.extend(int(vocab[token]) for token in token_pattern.findall(sample))
            rows.append(ids)
        max_len = max(len(row) for row in rows)
        input_ids = torch.full(
            (len(rows), max_len), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros_like(input_ids)
        for row_idx, ids in enumerate(rows):
            start = max_len - len(ids)
            input_ids[row_idx, start:] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row_idx, start:] = 1
        return _FakeBatchEncoding(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        )


class _FakeChatProcessor:
    image_token = "<image>"
    image_seq_len = 1

    def __init__(self) -> None:
        self.tokenizer = _FakeChatTokenizer()
        self.image_processor = SimpleNamespace(
            image_mean=[0.0, 0.0, 0.0],
            image_std=[1.0, 1.0, 1.0],
        )

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        del tokenize, add_generation_prompt
        parts: list[str] = []
        for message in messages:
            content = message.get("content", [])
            if isinstance(content, list):
                for item in content:
                    item_type = item.get("type")
                    if item_type == "image":
                        parts.append(self.image_token)
                    elif item_type == "text":
                        parts.append(str(item.get("text", "")))
            elif isinstance(content, str):
                parts.append(content)
        return " ".join(parts)

    def _check_special_mm_tokens(self, text, text_inputs, modalities=None) -> None:
        del text, text_inputs, modalities

    def __call__(
        self,
        *,
        text: list[str] | str | None = None,
        images=None,
        padding: bool = True,
        return_tensors: str = "pt",
        **kwargs: Any,
    ) -> _FakeBatchEncoding:
        del kwargs
        text_inputs = self.tokenizer(
            [] if text is None else text,
            padding=padding,
            return_tensors=return_tensors,
        )
        batch_size = int(text_inputs["input_ids"].shape[0])
        first_images = images[0] if images and images[0] is not None else None
        num_images = len(first_images) if first_images is not None else 0
        device = (
            first_images[0].device
            if first_images is not None and first_images
            else torch.device("cpu")
        )
        text_inputs["pixel_values"] = torch.zeros(
            (max(1, batch_size * max(1, num_images)), 4),
            dtype=torch.float32,
            device=device,
        )
        text_inputs["image_grid_thw"] = torch.zeros(
            (batch_size * num_images, 3), dtype=torch.long, device=device
        )
        return text_inputs


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
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
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
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.last_labels: torch.Tensor | None = None
        self.last_attention_mask: torch.Tensor | None = None
        self.last_use_cache: bool | None = None
        self.last_past_key_values = None
        self.gradient_checkpointing_kwargs: dict[str, Any] | None = None

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
        old_head = self.lm_head
        new_head = nn.Linear(
            old.embedding_dim,
            int(size),
            bias=False,
            device=old_head.weight.device,
        )
        with torch.no_grad():
            keep = min(old_head.out_features, int(size))
            new_head.weight[:keep].copy_(old_head.weight[:keep])
        self.lm_head = new_head
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
        self.last_past_key_values = kwargs.get("past_key_values")
        self.last_attention_mask = kwargs.get("attention_mask")
        self.last_use_cache = bool(use_cache)
        self.forward_calls += 1
        self.last_labels = labels.detach().clone() if labels is not None else None
        hidden = self.embed_tokens(input_ids)
        layer_inputs = []
        for layer in self.layers:
            layer_inputs.append(hidden)
            hidden = layer(hidden, attention_mask=kwargs.get("attention_mask"))
        logits = hidden @ self.embed_tokens.weight.T
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
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
        return SimpleNamespace(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
        )


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

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mixed = self._attention_mix(hidden_states, attention_mask=attention_mask)
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


class _FakeTokenCodec(nn.Module):
    def __init__(self, *, codebook_size: int = 4) -> None:
        super().__init__()
        self._codebook_size = int(codebook_size)
        self.detokenize_called = False

    @property
    def codebook_size(self) -> int:
        return self._codebook_size

    @property
    def max_code_seq_len(self) -> int:
        return self._codebook_size

    @property
    def block_schedule(self) -> tuple[int, ...]:
        return (1,) * self._codebook_size

    @property
    def supports_prefix_decode(self) -> bool:
        return True

    def set_io(self, io: PolicyIO) -> None:
        self.io = io

    def tokenize(self, actions: torch.Tensor) -> torch.Tensor:
        rows = torch.arange(self._codebook_size, device=actions.device)
        return rows.unsqueeze(0).expand(int(actions.shape[0]), -1)

    def detokenize(self, tokens: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        del tokens, kwargs
        self.detokenize_called = True
        raise AssertionError("Inference must use the continuous FM expert.")


class _FakeVariableTokenCodec(_FakeTokenCodec):
    @property
    def max_code_seq_len(self) -> int:
        return 4

    @property
    def block_schedule(self) -> None:
        return None

    def tokenize(self, actions: torch.Tensor, pad: bool = True):
        self.pad_values = getattr(self, "pad_values", [])
        self.pad_values.append(bool(pad))
        return [
            torch.tensor([0, 1], device=actions.device),
            torch.tensor([2, 3, 4], device=actions.device),
        ][: int(actions.shape[0])]


class _FakeOverlongVariableTokenCodec(_FakeVariableTokenCodec):
    def tokenize(self, actions: torch.Tensor, pad: bool = True):
        self.pad_values = getattr(self, "pad_values", [])
        self.pad_values.append(bool(pad))
        return [
            torch.tensor([0, 1, 2, 3, 4, 5], device=actions.device),
            torch.tensor([5, 4, 3, 2, 1, 0], device=actions.device),
        ][: int(actions.shape[0])]


class _FakeNegativeVariableTokenCodec(_FakeVariableTokenCodec):
    def tokenize(self, actions: torch.Tensor, pad: bool = True):
        self.pad_values = getattr(self, "pad_values", [])
        self.pad_values.append(bool(pad))
        return [
            torch.tensor([0, -1], device=actions.device),
            torch.tensor([2, 3], device=actions.device),
        ][: int(actions.shape[0])]


class _DummyPolicy(PolicyBase):
    CONFIG_CLASS = PolicyConfig
    _PREFIX_ATTENTION = "bidirectional"

    def _load_vlm(self, model_name: str, dtype_str: str) -> None:
        del model_name, dtype_str
        self.vlm = _FakeVLM()
        self.processor = SimpleNamespace(tokenizer=_FakeTokenizer())

    def _build_vlm_inputs(
        self,
        obs: dict[str, torch.Tensor],
        *,
        action_token_strings: list[str] | None = None,
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
            if action_token_strings is not None:
                ids.extend(
                    int(vocab[token])
                    for token in re.findall(r"<\|[^|]+\|>", action_token_strings[row])
                )
            rows.append(ids)
        input_ids = torch.tensor(rows, dtype=torch.long)
        out = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids.clone(),
        }
        return self._add_continuous_state_prefix_embeddings(out, state_prefix)

    def _build_vlm_prefix_adapter(self) -> _FakePrefixAdapter:
        return _FakePrefixAdapter(self.vlm)


class _FakePrefixAdapter:
    def __init__(self, vlm: _FakeVLM) -> None:
        self.vlm = vlm
        self.last_layer_attention_mask: torch.Tensor | None = None

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
        input_ids = vlm_inputs["input_ids"]
        mask = vlm_inputs["attention_mask"].bool()
        if layer_attention_mask is None:
            min_val = torch.finfo(self.vlm.embed_tokens.weight.dtype).min
            valid_keys = mask[:, None, None, :]
            layer_attention_mask = torch.where(
                valid_keys,
                torch.zeros((), dtype=self.vlm.embed_tokens.weight.dtype),
                torch.full((), min_val, dtype=self.vlm.embed_tokens.weight.dtype),
            ).expand(
                int(input_ids.shape[0]),
                1,
                int(input_ids.shape[1]),
                int(input_ids.shape[1]),
            )
        self.last_layer_attention_mask = layer_attention_mask
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
            attention_mask=self.last_layer_attention_mask,
            position_ids=positions,
            cache_position=positions.squeeze(0),
            position_embeddings=(hidden, hidden),
        )

    def forward_layer(
        self, state: VlmPrefixState, *, layer_idx: int
    ) -> tuple[VlmPrefixState, LayerKV]:
        layer_kv = self.project_layer_kv(state, layer_idx=layer_idx)
        return self.forward_prefix_layer(state, layer_idx=layer_idx), layer_kv

    def forward_prefix_layer(
        self, state: VlmPrefixState, *, layer_idx: int
    ) -> VlmPrefixState:
        layer = cast(_FakeDecoderLayer, self.vlm.layers[layer_idx])
        hidden = layer(state.hidden, attention_mask=state.attention_mask)
        return VlmPrefixState(
            hidden=hidden,
            mask=state.mask,
            attention_mask=state.attention_mask,
            position_ids=state.position_ids,
            cache_position=state.cache_position,
            position_embeddings=state.position_embeddings,
            extra=state.extra,
        )

    def project_layer_kv(
        self,
        state: VlmPrefixState,
        *,
        layer_idx: int,
        position_ids: torch.Tensor | None = None,
    ) -> LayerKV:
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
        if position_ids is not None:
            key = key + position_ids[:, :, None, None].to(dtype=key.dtype)
        return LayerKV(
            key=key.transpose(1, 2),
            value=value.transpose(1, 2),
        )

    def final_hidden_to_logits(
        self,
        hidden: torch.Tensor,
        *,
        candidate_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = self.vlm.embed_tokens.weight
        if candidate_token_ids is not None:
            weight = weight.index_select(0, candidate_token_ids.to(weight.device))
        return hidden @ weight.T


class _DummyFastPolicy(VariableLengthActionTokenMixin, _DummyPolicy):
    CONFIG_CLASS = PolicyConfig


@pytest.fixture()
def fake_codec(monkeypatch) -> _FakeTokenCodec:
    codec = _FakeTokenCodec(codebook_size=4)
    monkeypatch.setattr(policy_module, "make_codec", lambda _cfg: codec)
    return codec


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


def _image_only_io() -> PolicyIO:
    return PolicyIO(
        obs_features=[
            FeatureSpec(
                key="observation.image",
                type=FeatureType.IMAGE,
                shape=(3, 4, 4),
                norm_mode=NormMode.MEAN_STD,
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


def _policy(
    *,
    state_prefix_encoding: str = "continuous_projection",
) -> _DummyPolicy:
    policy = _DummyPolicy(
        config=PolicyConfig(
            action_dim=3,
            action_horizon=4,
            n_action_steps=2,
            num_action_tokens=4,
            action_tokenizer={"name": "fake", "kind": "tokens"},
            expert_hidden_size=16,
            state_prefix_encoding=state_prefix_encoding,
        ).to_dict()
    )
    policy.set_io(_io())
    return policy


def test_config_validates_expert_time_period_range() -> None:
    with pytest.raises(ValueError, match="expert_time_min_period"):
        PolicyConfig(expert_time_min_period=4.0, expert_time_max_period=0.004)


def test_action_expert_config_defaults_to_continuous_state_prefix() -> None:
    config = PolicyConfig()
    assert config.state_prefix_encoding == "continuous_projection"
    assert (
        PolicyConfig.from_dict({"name": "ki_oat_pali"}).state_prefix_encoding
        == "continuous_projection"
    )
    assert config.ce_warmstart_steps == 0
    assert config.ce_vocab_scope == "action"


def test_from_io_uses_continuous_state_prefix_by_default(
    fake_codec: _FakeTokenCodec,
) -> None:
    del fake_codec
    policy = _DummyPolicy.from_io(
        _io(),
        num_action_tokens=4,
        action_tokenizer={"name": "fake", "kind": "tokens"},
        expert_hidden_size=16,
    )

    assert policy.config.state_prefix_encoding == "continuous_projection"


def test_optimizer_groups_label_vlm_bridge_and_expert_params(
    fake_codec: _FakeTokenCodec,
) -> None:
    del fake_codec
    policy = _policy()

    param_families: dict[int, str | None] = {}
    for group in policy.get_optim_params():
        group_family = cast(str | None, group.get("family"))
        for param in group["params"]:
            param_families[id(param)] = group_family

    vlm_param = next(policy.vlm.parameters())
    assert policy.state_prefix_projector is not None
    state_param = next(policy.state_prefix_projector.parameters())
    assert policy.action_expert is not None
    expert_param = next(policy.action_expert.parameters())

    assert param_families[id(vlm_param)] == "vlm"
    assert param_families[id(state_param)] == "bridge"
    assert param_families[id(expert_param)] == "action_expert"


def test_flow_only_ki_rejects_continuous_state_prefix(
    fake_codec: _FakeTokenCodec,
) -> None:
    del fake_codec
    policy = _DummyPolicy(
        config=PolicyConfig(
            action_dim=3,
            action_horizon=4,
            n_action_steps=2,
            num_action_tokens=4,
            action_tokenizer={"name": "fake", "kind": "tokens"},
            expert_hidden_size=16,
            ce_loss_weight=0.0,
            flow_loss_weight=1.0,
            state_prefix_encoding="continuous_projection",
        ).to_dict()
    )
    policy.set_io(_io())

    with pytest.raises(ValueError, match="flow-only"):
        policy.forward(_batch())


def test_continuous_state_prefix_ce_updates_projector(
    fake_codec: _FakeTokenCodec,
) -> None:
    del fake_codec
    policy = _DummyPolicy(
        config=PolicyConfig(
            action_dim=3,
            action_horizon=4,
            n_action_steps=2,
            num_action_tokens=4,
            action_tokenizer={"name": "fake", "kind": "tokens"},
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
    prepared = prepare_action_token_ce(
        policy,
        batch,
        actions=batch["action"],
        ignore_index=IGNORE_INDEX,
    )

    counts = (
        prepared.vlm_inputs["input_ids"].eq(policy.state_prefix_token_id).sum(dim=1)
    )
    assert counts.tolist() == [1, 1]
    assert STATE_PREFIX_EMBEDDINGS_KEY in prepared.vlm_inputs
    assert torch.equal(
        prepared.vlm_inputs[STATE_PREFIX_PLACEHOLDER_MASK_KEY],
        prepared.vlm_inputs["input_ids"].eq(policy.state_prefix_token_id),
    )

    result = run_action_token_ce_prefix_forward(
        policy,
        prepared,
        prefix_adapter=policy._require_vlm_ce_prefix_adapter(),
        ignore_index=IGNORE_INDEX,
        gradient_checkpointing=False,
    )
    result.loss.backward()

    assert policy.state_prefix_projector is not None
    grad = policy.state_prefix_projector.weight.grad
    assert grad is not None
    assert grad.abs().sum().item() > 0.0


def test_ki_fsdp_wrap_modules_include_state_prefix_projector(
    fake_codec: _FakeTokenCodec,
) -> None:
    del fake_codec
    policy = _policy()

    assert policy.state_prefix_projector is not None
    assert policy.state_prefix_projector in policy.fsdp_wrap_modules()


def test_ki_fsdp_root_group_excludes_state_prefix_projector(
    fake_codec: _FakeTokenCodec,
) -> None:
    del fake_codec
    policy = _policy()
    policy.vlm.embed_tokens.to(dtype=torch.bfloat16)
    policy.vlm.lm_head.to(dtype=torch.bfloat16)
    assigned_params = {
        param for module in policy.fsdp_wrap_modules() for param in module.parameters()
    }

    validate_fsdp_parameter_group(policy, excluded_params=assigned_params)


def test_ki_flow_context_detaches_continuous_state_prefix(
    fake_codec: _FakeTokenCodec,
) -> None:
    del fake_codec
    policy = _DummyPolicy(
        config=PolicyConfig(
            action_dim=3,
            action_horizon=4,
            n_action_steps=2,
            num_action_tokens=4,
            action_tokenizer={"name": "fake", "kind": "tokens"},
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
    prepared = prepare_action_token_ce(
        policy,
        batch,
        actions=batch["action"],
        ignore_index=IGNORE_INDEX,
    )
    sample = FlowSample(
        noisy_actions=torch.zeros_like(batch["action"]),
        target_velocity=torch.ones_like(batch["action"]),
        timesteps=torch.full((batch["action"].shape[0],), 0.5),
        noise=torch.zeros_like(batch["action"]),
    )
    model = policy._require_vlm_action_expert()
    schedule_attention_mask = build_prepared_action_token_attention_mask(
        policy,
        prepared,
    )
    expert_context_mask = policy._build_flow_expert_context_mask(batch, prepared)

    flow_loss, _output = model.loss_with_context(
        vlm_inputs=prepared.vlm_inputs,
        layer_attention_mask=schedule_attention_mask,
        expert_context_mask=expert_context_mask,
        sample=sample,
        action_is_pad=None,
        detach_expert_context=True,
        gradient_checkpointing=False,
    )
    flow_loss.backward()

    assert policy.state_prefix_projector is not None
    grad = policy.state_prefix_projector.weight.grad
    assert grad is None or grad.abs().sum().item() == 0.0


def _pali_policy_pair(monkeypatch):
    bar_processor = _FakePaliProcessor()
    ki_processor = _FakePaliProcessor()
    io = _image_only_io()

    def _bar_load_vlm(self, model_name: str, dtype_str: str) -> None:
        del model_name, dtype_str
        self.vlm = _FakeVLM()
        self.processor = bar_processor

    def _ki_load_vlm(self, model_name: str, dtype_str: str) -> None:
        del model_name, dtype_str
        self.vlm = _FakeVLM()
        self.processor = ki_processor

    monkeypatch.setattr(
        "praxis.policies.bar.backbones.pali.pali_base.PaliPolicyBase._load_vlm",
        _bar_load_vlm,
    )
    monkeypatch.setattr(
        "praxis.policies.ki.backbones.pali.pali_base.PaliPolicyBase._load_vlm",
        _ki_load_vlm,
    )
    monkeypatch.setattr(
        "praxis.policies.bar.backbones.pali.pali_base.PaliPolicyBase._build_vlm_prefix_adapter",
        lambda self: _FakePrefixAdapter(self.vlm),
    )
    monkeypatch.setattr(
        "praxis.policies.ki.backbones.pali.pali_base.PaliPolicyBase._build_vlm_prefix_adapter",
        lambda self: _FakePrefixAdapter(self.vlm),
    )
    monkeypatch.setattr(
        "praxis.policies.bar.core.vlm.base.make_codec",
        lambda _cfg: _FakeTokenCodec(codebook_size=4),
    )
    monkeypatch.setattr(
        policy_module,
        "make_codec",
        lambda _cfg: _FakeTokenCodec(codebook_size=4),
    )
    common_overrides = {
        "action_dim": 3,
        "action_horizon": 4,
        "n_action_steps": 2,
        "num_action_tokens": 4,
        "action_tokenizer": {"name": "fake", "kind": "tokens"},
        "state_prefix_encoding": "discrete_text",
    }
    bar_policy = BarOATPaliPolicy.from_io(io, **common_overrides)
    ki_policy = KIOATPaliPolicy.from_io(
        io,
        **common_overrides,
        expert_hidden_size=16,
    )
    return bar_policy, ki_policy, bar_processor, ki_processor


def _chat_policy_pair(monkeypatch, family: str):
    bar_processor = _FakeChatProcessor()
    ki_processor = _FakeChatProcessor()
    io = _image_only_io()

    if family == "qwen":
        bar_cls = BarOATQwenPolicy
        ki_cls = KIOATQwenPolicy
        bar_patch = (
            "praxis.policies.bar.backbones.qwen.qwen_base.QwenPolicyBase._load_vlm"
        )
        ki_patch = (
            "praxis.policies.ki.backbones.qwen.qwen_base.QwenPolicyBase._load_vlm"
        )
        extra_overrides: dict[str, object] = {}
    elif family == "smol":
        bar_cls = BarOATSmolPolicy
        ki_cls = KIOATSmolPolicy
        bar_patch = (
            "praxis.policies.bar.backbones.smol.smol_base.SmolPolicyBase._load_vlm"
        )
        ki_patch = (
            "praxis.policies.ki.backbones.smol.smol_base.SmolPolicyBase._load_vlm"
        )
        extra_overrides = {"resize_imgs_with_padding": (4, 4)}
    else:
        raise AssertionError(f"Unexpected family {family!r}.")

    def _bar_load_vlm(self, model_name: str, dtype_str: str) -> None:
        del model_name, dtype_str
        self.vlm = _FakeVLM()
        self.processor = bar_processor

    def _ki_load_vlm(self, model_name: str, dtype_str: str) -> None:
        del model_name, dtype_str
        self.vlm = _FakeVLM()
        self.processor = ki_processor

    monkeypatch.setattr(bar_patch, _bar_load_vlm)
    monkeypatch.setattr(ki_patch, _ki_load_vlm)
    monkeypatch.setattr(
        bar_patch.rsplit(".", 1)[0] + "._build_vlm_prefix_adapter",
        lambda self: _FakePrefixAdapter(self.vlm),
    )
    monkeypatch.setattr(
        ki_patch.rsplit(".", 1)[0] + "._build_vlm_prefix_adapter",
        lambda self: _FakePrefixAdapter(self.vlm),
    )
    monkeypatch.setattr(
        "praxis.policies.bar.core.vlm.base.make_codec",
        lambda _cfg: _FakeTokenCodec(codebook_size=4),
    )
    monkeypatch.setattr(
        policy_module,
        "make_codec",
        lambda _cfg: _FakeTokenCodec(codebook_size=4),
    )
    common_overrides = {
        "action_dim": 3,
        "action_horizon": 4,
        "n_action_steps": 2,
        "num_action_tokens": 4,
        "action_tokenizer": {"name": "fake", "kind": "tokens"},
        "state_prefix_encoding": "discrete_text",
        **extra_overrides,
    }
    bar_policy = bar_cls.from_io(io, **common_overrides)
    ki_policy = ki_cls.from_io(
        io,
        **common_overrides,
        expert_hidden_size=16,
    )
    return bar_policy, ki_policy, bar_processor, ki_processor


def _assert_matching_vlm_inputs(
    bar_inputs: dict[str, torch.Tensor],
    ki_inputs: dict[str, torch.Tensor],
) -> None:
    assert set(bar_inputs) == set(ki_inputs)
    for key in sorted(bar_inputs):
        assert torch.equal(bar_inputs[key], ki_inputs[key]), key


def _batch() -> dict[str, torch.Tensor]:
    return {
        "observation.state": torch.zeros((2, 2), dtype=torch.float32),
        "action": torch.randn((2, 4, 3), dtype=torch.float32),
    }


def _has_nonzero_grad(module: nn.Module) -> bool:
    return any(
        param.grad is not None and bool(param.grad.detach().abs().sum().item() > 0)
        for param in module.parameters()
    )


def test_ce_forward_uses_sparse_prefix_lane_without_full_vlm_forward(
    fake_codec: _FakeTokenCodec,
) -> None:
    del fake_codec
    policy = _policy()
    ce_loss = policy.compute_ce_loss(_batch())
    adapter = cast(_FakePrefixAdapter, policy._require_vlm_ce_prefix_adapter())

    assert ce_loss.requires_grad
    assert adapter.last_layer_attention_mask is not None
    assert policy.vlm.forward_calls == 0
    assert policy.vlm.last_use_cache is None
    assert policy.vlm.last_past_key_values is None
    assert policy.vlm.last_labels is None


def test_default_expert_width_uses_vlm_multiplier(fake_codec: _FakeTokenCodec) -> None:
    del fake_codec
    policy = _DummyPolicy(
        config=PolicyConfig(
            action_dim=3,
            action_horizon=4,
            n_action_steps=2,
            num_action_tokens=4,
            action_tokenizer={"name": "fake", "kind": "tokens"},
            expert_hidden_size=None,
            expert_width_multiplier=0.75,
        ).to_dict()
    )
    policy.set_io(_io())

    expert = policy._require_action_expert()
    assert expert.hidden_size == 9


def test_gradient_checkpointing_is_allowed_with_cache_free_train_forward(
    fake_codec: _FakeTokenCodec,
) -> None:
    del fake_codec
    policy = _DummyPolicy(
        config=PolicyConfig(
            action_dim=3,
            action_horizon=4,
            n_action_steps=2,
            num_action_tokens=4,
            action_tokenizer={"name": "fake", "kind": "tokens"},
            expert_hidden_size=16,
            gradient_checkpointing=True,
        ).to_dict()
    )

    assert policy.vlm.gradient_checkpointing_kwargs == {
        "gradient_checkpointing_kwargs": {"use_reentrant": False}
    }


def test_pali_vlm_prompt_and_suffix_match_bar_without_state(monkeypatch) -> None:
    # KI intentionally routes raw state only to the FM expert. BAR/KI VLM parity
    # therefore applies to the no-state prompt surface: images, instruction, and
    # action-token suffix.
    bar_policy, ki_policy, bar_processor, ki_processor = _pali_policy_pair(monkeypatch)
    batch = {
        "observation.image": torch.rand(1, 1, 3, 4, 4),
        "action": torch.randn(1, 4, 3),
    }
    action_text = ki_policy._require_action_token_vocabulary().render_raw_ids(
        [0, 1, 2, 3]
    )

    bar_inputs = bar_policy._build_vlm_inputs(
        batch,
        action_token_strings=[action_text],
        instructions=["pick up the cube"],
    )
    ki_inputs = ki_policy._build_vlm_inputs(
        batch,
        action_token_strings=[action_text],
        instructions=["pick up the cube"],
    )

    assert bar_processor.calls[-1]["text"] == ki_processor.calls[-1]["text"]
    assert bar_processor.calls[-1]["suffix"] == ki_processor.calls[-1]["suffix"]
    assert torch.equal(bar_inputs["input_ids"], ki_inputs["input_ids"])
    assert torch.equal(bar_inputs["attention_mask"], ki_inputs["attention_mask"])


@pytest.mark.parametrize("family", ["qwen", "smol"])
def test_chat_vlm_prompt_and_suffix_match_bar_without_state(
    monkeypatch,
    family: str,
) -> None:
    bar_policy, ki_policy, bar_processor, ki_processor = _chat_policy_pair(
        monkeypatch, family
    )
    batch = {
        "observation.image": torch.rand(1, 1, 3, 4, 4),
        "action": torch.randn(1, 4, 3),
    }
    action_text = ki_policy._require_action_token_vocabulary().render_raw_ids(
        [0, 1, 2, 3]
    )

    for action_token_strings in ([action_text], None):
        bar_inputs = bar_policy._build_vlm_inputs(
            batch,
            action_token_strings=action_token_strings,
            instructions=["pick up the cube"],
        )
        ki_inputs = ki_policy._build_vlm_inputs(
            batch,
            action_token_strings=action_token_strings,
            instructions=["pick up the cube"],
        )

        assert bar_processor.tokenizer.last_text == ki_processor.tokenizer.last_text
        _assert_matching_vlm_inputs(bar_inputs, ki_inputs)


@pytest.mark.parametrize("family", ["pali", "qwen", "smol"])
@pytest.mark.parametrize("ce_vocab_scope", ["full", "action"])
def test_one_forward_vlm_ce_matches_bar_without_state(
    monkeypatch,
    family: str,
    ce_vocab_scope: str,
) -> None:
    if family == "pali":
        bar_policy, ki_policy, _bar_processor, _ki_processor = _pali_policy_pair(
            monkeypatch
        )
    else:
        bar_policy, ki_policy, _bar_processor, _ki_processor = _chat_policy_pair(
            monkeypatch,
            family,
        )
    ki_policy.vlm.load_state_dict(bar_policy.vlm.state_dict())
    ki_policy.config.flow_loss_weight = 0.0
    bar_policy.config.ce_vocab_scope = ce_vocab_scope
    ki_policy.config.ce_vocab_scope = ce_vocab_scope
    batch = {
        "observation.image": torch.rand(2, 1, 3, 4, 4),
        "action": torch.randn(2, 4, 3),
    }

    bar_loss, _bar_info = bar_policy.forward(batch)
    ki_loss, ki_info = ki_policy.forward(batch)

    torch.testing.assert_close(ki_loss, bar_loss)
    assert ki_info["ce_loss"] == pytest.approx(float(bar_loss.detach().item()))
    assert ki_info["flow_loss"] == 0.0
    assert bar_policy.vlm.forward_calls == 0
    assert ki_policy.vlm.forward_calls == 0


@pytest.mark.parametrize("family", ["pali", "qwen", "smol"])
@pytest.mark.parametrize("ce_vocab_scope", ["full", "action"])
def test_bar_prefix_ce_projects_only_supervised_hidden_rows(
    monkeypatch,
    family: str,
    ce_vocab_scope: str,
) -> None:
    if family == "pali":
        bar_policy, _ki_policy, _bar_processor, _ki_processor = _pali_policy_pair(
            monkeypatch
        )
    else:
        bar_policy, _ki_policy, _bar_processor, _ki_processor = _chat_policy_pair(
            monkeypatch,
            family,
        )
    bar_policy.config.ce_vocab_scope = ce_vocab_scope
    batch = {
        "observation.image": torch.rand(2, 1, 3, 4, 4),
        "action": torch.randn(2, 4, 3),
    }
    prepared = prepare_action_token_ce(
        bar_policy,
        batch,
        actions=batch["action"],
        ignore_index=IGNORE_INDEX,
    )
    adapter = bar_policy._require_vlm_ce_prefix_adapter()
    original_project = adapter.final_hidden_to_logits
    projected_shapes: list[tuple[int, ...]] = []
    projected_candidate_widths: list[int | None] = []

    def project_hidden_to_logits(
        hidden: torch.Tensor,
        *,
        candidate_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        projected_shapes.append(tuple(hidden.shape))
        projected_candidate_widths.append(
            None if candidate_token_ids is None else int(candidate_token_ids.numel())
        )
        return original_project(hidden, candidate_token_ids=candidate_token_ids)

    monkeypatch.setattr(adapter, "final_hidden_to_logits", project_hidden_to_logits)
    prefix_result = run_action_token_ce_prefix_forward(
        bar_policy,
        prepared,
        prefix_adapter=adapter,
        ignore_index=IGNORE_INDEX,
        gradient_checkpointing=False,
        projection_chunk_size=2,
    )

    supervised_count = int(prepared.labels.ne(IGNORE_INDEX).sum().item())
    assert prefix_result.loss.requires_grad
    assert sum(shape[0] for shape in projected_shapes) == supervised_count
    assert all(len(shape) == 2 for shape in projected_shapes)
    assert all(shape[0] <= 2 for shape in projected_shapes)
    if ce_vocab_scope == "action":
        candidate_ids = build_action_token_ce_candidate_token_ids(
            bar_policy,
            device=prepared.labels.device,
        )
        assert candidate_ids is not None
        assert set(projected_candidate_widths) == {int(candidate_ids.numel())}
    else:
        assert set(projected_candidate_widths) == {None}
    assert bar_policy.vlm.forward_calls == 0


def test_pali_vlm_action_labels_and_prefix_mask_match_bar(monkeypatch) -> None:
    # Keep this parity fixture state-free. KI VLM conditioning excludes state by
    # design, while its FM expert receives continuous state separately.
    bar_policy, ki_policy, _bar_processor, _ki_processor = _pali_policy_pair(
        monkeypatch
    )
    batch = {
        "observation.image": torch.rand(1, 1, 3, 4, 4),
        "action": torch.randn(1, 4, 3),
    }
    prepared = prepare_action_token_ce(
        ki_policy,
        batch,
        actions=batch["action"],
        ignore_index=IGNORE_INDEX,
    )
    bar_prepared = prepare_action_token_ce_from_token_targets(
        bar_policy,
        batch,
        token_ids_batch=[[0, 1, 2, 3]],
        token_lengths=None,
        ignore_index=IGNORE_INDEX,
    )

    assert bar_prepared.block_schedule == prepared.block_schedule
    assert torch.equal(
        bar_prepared.vlm_inputs["input_ids"], prepared.vlm_inputs["input_ids"]
    )
    assert torch.equal(
        bar_prepared.vlm_inputs["attention_mask"], prepared.vlm_inputs["attention_mask"]
    )
    assert torch.equal(bar_prepared.labels, prepared.labels)
    assert _ki_processor.calls[-1]["suffix"] == [
        "<|action_mask|><|action_0|><|action_1|><|action_2|>"
    ]

    ki_prefix_inputs = ki_policy._build_vlm_inputs(
        batch,
        action_token_strings=None,
        instructions=["pick up the cube"],
    )
    bar_prefix_inputs = bar_policy._build_vlm_inputs(
        batch,
        action_token_strings=None,
        instructions=["pick up the cube"],
    )
    ce_prefix = prepared.vlm_inputs["input_ids"][0][prepared.supervision.prefix_mask[0]]
    inference_prefix = ki_prefix_inputs["input_ids"][0][
        ki_prefix_inputs["attention_mask"][0].bool()
    ]

    assert torch.equal(ce_prefix, inference_prefix)
    assert torch.equal(bar_prefix_inputs["input_ids"], ki_prefix_inputs["input_ids"])


def test_pali_ce_attention_mask_uses_action_positions_before_processor_eos(
    monkeypatch,
) -> None:
    _bar_policy, ki_policy, _bar_processor, _ki_processor = _pali_policy_pair(
        monkeypatch
    )
    batch = {
        "observation.image": torch.rand(1, 1, 3, 4, 4),
        "action": torch.randn(1, 4, 3),
    }

    _loss = ki_policy.compute_ce_loss(batch)

    adapter = cast(_FakePrefixAdapter, ki_policy._require_vlm_ce_prefix_adapter())
    mask = adapter.last_layer_attention_mask
    assert mask is not None
    mask_2d = mask[0, 0]
    min_val = torch.finfo(mask_2d.dtype).min
    # Fake Pali input is [prefix x3, action x4, EOS]. Prefix queries must not
    # see action or EOS.
    assert mask_2d[0, 2] == 0.0
    assert mask_2d[0, 3] == min_val
    assert mask_2d[0, 7] == min_val
    # The first action query sees the prefix and itself, not future action tokens
    # or the processor-appended EOS tail.
    assert mask_2d[3, 0] == 0.0
    assert mask_2d[3, 3] == 0.0
    assert mask_2d[3, 4] == min_val
    assert mask_2d[3, 7] == min_val


def test_ce_loss_ignores_samples_with_padded_actions(
    fake_codec: _FakeTokenCodec,
) -> None:
    del fake_codec
    policy = _policy()
    batch = _batch()
    batch["action_is_pad"] = torch.tensor(
        [[False, False, False, False], [False, True, True, True]],
        dtype=torch.bool,
    )

    prepared = prepare_action_token_ce(
        policy,
        batch,
        actions=batch["action"],
        ignore_index=IGNORE_INDEX,
    )

    assert prepared.labels[0].ne(IGNORE_INDEX).any()
    assert not prepared.labels[1].ne(IGNORE_INDEX).any()


def test_fast_ce_uses_variable_lengths_and_eos(monkeypatch) -> None:
    codec = _FakeVariableTokenCodec(codebook_size=6)
    monkeypatch.setattr(policy_module, "make_codec", lambda _cfg: codec)
    policy = _DummyFastPolicy(
        config=PolicyConfig(
            action_dim=3,
            action_horizon=4,
            n_action_steps=2,
            num_action_tokens=6,
            action_tokenizer={"name": "fast", "kind": "tokens"},
            expert_hidden_size=16,
            ce_supervise_eos=True,
        ).to_dict()
    )
    policy.set_io(_io())

    batch = _batch()
    prepared = prepare_action_token_ce(
        policy,
        batch,
        actions=batch["action"],
        ignore_index=IGNORE_INDEX,
    )
    vocab = policy._require_action_token_vocabulary()
    prediction_labels = prepared.labels[
        prepared.supervision.prediction_position_mask
    ].view(2, -1)

    assert prepared.block_schedule == (1, 1, 1, 1, 1)
    assert prepared.token_lengths == [2, 3]
    assert codec.pad_values == [False]
    assert prediction_labels[0].tolist() == [
        vocab.token_min,
        vocab.token_min + 1,
        policy._eos_token_id,
        IGNORE_INDEX,
        IGNORE_INDEX,
    ]
    assert prediction_labels[1].tolist() == [
        vocab.token_min + 2,
        vocab.token_min + 3,
        vocab.token_min + 4,
        policy._eos_token_id,
        IGNORE_INDEX,
    ]


def test_fast_one_forward_ce_matches_standalone_variable_length_ce(monkeypatch) -> None:
    monkeypatch.setattr(
        policy_module,
        "make_codec",
        lambda _cfg: _FakeVariableTokenCodec(codebook_size=6),
    )
    policy = _DummyFastPolicy(
        config=PolicyConfig(
            action_dim=3,
            action_horizon=4,
            n_action_steps=2,
            num_action_tokens=6,
            action_tokenizer={"name": "fast", "kind": "tokens"},
            expert_hidden_size=16,
            ce_supervise_eos=True,
            flow_loss_weight=0.0,
        ).to_dict()
    )
    policy.set_io(_io())
    batch = _batch()

    standalone_ce = policy.compute_ce_loss(batch, actions=batch["action"])
    one_forward_loss, info = policy.forward(batch)

    torch.testing.assert_close(one_forward_loss, standalone_ce)
    assert info["ce_loss"] == pytest.approx(float(standalone_ce.detach().item()))
    assert info["flow_loss"] == 0.0


def test_fast_ce_overlong_rows_keep_final_eos_slot(monkeypatch) -> None:
    monkeypatch.setattr(
        policy_module,
        "make_codec",
        lambda _cfg: _FakeOverlongVariableTokenCodec(codebook_size=6),
    )
    policy = _DummyFastPolicy(
        config=PolicyConfig(
            action_dim=3,
            action_horizon=4,
            n_action_steps=2,
            num_action_tokens=6,
            action_tokenizer={"name": "fast", "kind": "tokens"},
            expert_hidden_size=16,
            ce_supervise_eos=True,
        ).to_dict()
    )
    policy.set_io(_io())

    batch = _batch()
    prepared = prepare_action_token_ce(
        policy,
        batch,
        actions=batch["action"],
        ignore_index=IGNORE_INDEX,
    )
    vocab = policy._require_action_token_vocabulary()
    prediction_labels = prepared.labels[
        prepared.supervision.prediction_position_mask
    ].view(2, -1)

    assert prepared.block_schedule == (1, 1, 1, 1, 1)
    assert prepared.token_lengths == [4, 4]
    assert prediction_labels[0].tolist() == [
        vocab.token_min,
        vocab.token_min + 1,
        vocab.token_min + 2,
        vocab.token_min + 3,
        policy._eos_token_id,
    ]
    assert prediction_labels[1].tolist() == [
        vocab.token_min + 5,
        vocab.token_min + 4,
        vocab.token_min + 3,
        vocab.token_min + 2,
        policy._eos_token_id,
    ]


def test_fast_ce_rejects_negative_raw_token_ids(monkeypatch) -> None:
    monkeypatch.setattr(
        policy_module,
        "make_codec",
        lambda _cfg: _FakeNegativeVariableTokenCodec(codebook_size=6),
    )
    policy = _DummyFastPolicy(
        config=PolicyConfig(
            action_dim=3,
            action_horizon=4,
            n_action_steps=2,
            num_action_tokens=6,
            action_tokenizer={"name": "fast", "kind": "tokens"},
            expert_hidden_size=16,
            ce_supervise_eos=True,
        ).to_dict()
    )
    policy.set_io(_io())
    batch = _batch()

    with pytest.raises(ValueError) as exc_info:
        prepare_action_token_ce(
            policy,
            batch,
            actions=batch["action"],
            ignore_index=IGNORE_INDEX,
        )
    assert "outside [0, 6)" in str(exc_info.value)
    assert "-1" in str(exc_info.value)


def test_flow_loss_masks_padded_action_steps(
    fake_codec: _FakeTokenCodec,
) -> None:
    del fake_codec
    policy = _policy()
    batch = _batch()
    batch["action_is_pad"] = torch.ones((2, 4), dtype=torch.bool)

    assert policy.vlm.last_labels is None

    flow_loss = policy.compute_integrated_flow_loss(
        batch,
        actions=batch["action"],
        vlm_inputs=policy._build_vlm_inputs(batch, action_token_strings=None),
        detach_prefix=True,
    )

    assert torch.isfinite(flow_loss)
    assert flow_loss.item() == 0.0
    assert flow_loss.requires_grad


def test_action_expert_uses_smolvla_style_action_time_fusion(
    fake_codec: _FakeTokenCodec,
) -> None:
    del fake_codec
    policy = _policy()
    expert = policy._require_action_expert()

    assert expert.action_time_mlp_in.in_features == 2 * expert.hidden_size
    assert expert.action_time_mlp_in.out_features == expert.hidden_size
    assert expert.action_time_mlp_out.in_features == expert.hidden_size
    assert expert.action_time_mlp_out.out_features == expert.hidden_size
    assert expert.time_min_period == policy.config.expert_time_min_period
    assert expert.time_max_period == policy.config.expert_time_max_period


def test_lora_vocab_gradient_mask_keeps_only_action_owned_rows(
    fake_codec: _FakeTokenCodec,
) -> None:
    del fake_codec
    policy = _policy()
    policy.config.use_lora = True
    policy._install_vocab_row_gradient_masks()
    vocab = policy._require_action_token_vocabulary()

    loss = policy.vlm.embed_tokens.weight.sum() + policy.vlm.lm_head.weight.sum()
    loss.backward()

    assert policy.vlm.embed_tokens.weight.grad is not None
    assert policy.vlm.lm_head.weight.grad is not None
    assert policy.vlm.embed_tokens.weight.grad[1].abs().sum().item() == 0.0
    assert policy.vlm.lm_head.weight.grad[1].abs().sum().item() == 0.0
    assert policy.vlm.embed_tokens.weight.grad[vocab.token_min].abs().sum().item() > 0.0
    assert (
        policy.vlm.embed_tokens.weight.grad[vocab.mask_token_id].abs().sum().item()
        > 0.0
    )


def test_lora_optim_groups_put_vocab_weights_on_zero_decay(
    fake_codec: _FakeTokenCodec,
) -> None:
    del fake_codec
    policy = _policy()
    policy.config.use_lora = True

    groups = policy.get_optim_params()

    zero_decay_groups = [
        group for group in groups if float(group.get("weight_decay", 1.0)) == 0.0
    ]
    assert zero_decay_groups
    zero_decay_param_ids = {
        id(param) for group in zero_decay_groups for param in group["params"]
    }
    assert id(policy.vlm.embed_tokens.weight) in zero_decay_param_ids
    assert id(policy.vlm.lm_head.weight) in zero_decay_param_ids


def test_flow_loss_updates_expert_not_vlm(fake_codec: _FakeTokenCodec) -> None:
    del fake_codec
    policy = _policy()
    policy.zero_grad(set_to_none=True)

    batch = _batch()
    flow_loss = policy.compute_integrated_flow_loss(
        batch,
        actions=batch["action"],
        vlm_inputs=policy._build_vlm_inputs(batch, action_token_strings=None),
        detach_prefix=True,
    )
    flow_loss.backward()

    assert not _has_nonzero_grad(policy.vlm)
    assert policy.action_expert is not None
    assert _has_nonzero_grad(policy.action_expert)


def test_combined_loss_updates_vlm_and_expert(fake_codec: _FakeTokenCodec) -> None:
    del fake_codec
    policy = _policy()
    loss, info = policy.forward(_batch())
    loss.backward()

    assert info["ce_loss"] > 0.0
    assert info["flow_loss"] > 0.0
    assert "loss" not in info
    assert _has_nonzero_grad(policy.vlm)
    assert policy.action_expert is not None
    assert _has_nonzero_grad(policy.action_expert)
    assert policy.vlm.forward_calls == 0
    assert policy.vlm_action_expert is not None
    adapter = cast(_FakePrefixAdapter, policy.vlm_action_expert.prefix_adapter)
    assert adapter.last_layer_attention_mask is not None


def test_ce_warmstart_steps_disable_flow_until_boundary(
    fake_codec: _FakeTokenCodec,
) -> None:
    del fake_codec
    policy = _policy()
    policy.config.ce_warmstart_steps = 2

    policy.set_training_context(TrainingContext(global_step=1))
    policy.zero_grad(set_to_none=True)
    loss, info = policy.forward(_batch())
    loss.backward()

    assert info["ce_loss"] > 0.0
    assert info["flow_loss"] == 0.0
    assert _has_nonzero_grad(policy.vlm)
    assert policy.action_expert is not None
    assert not _has_nonzero_grad(policy.action_expert)

    policy.set_training_context(TrainingContext(global_step=2))
    policy.zero_grad(set_to_none=True)
    loss, info = policy.forward(_batch())
    loss.backward()

    assert info["ce_loss"] > 0.0
    assert info["flow_loss"] > 0.0
    assert _has_nonzero_grad(policy.vlm)
    assert _has_nonzero_grad(policy.action_expert)


@pytest.mark.parametrize(
    ("ce_weight", "flow_weight", "expect_vlm_grad", "expect_expert_grad"),
    [
        (1.0, 1.0, True, True),
        (0.0, 1.0, False, True),
        (1.0, 0.0, True, False),
    ],
)
def test_gradient_checkpointing_covers_integrated_ki_one_forward_train_forward(
    monkeypatch,
    fake_codec: _FakeTokenCodec,
    ce_weight: float,
    flow_weight: float,
    expect_vlm_grad: bool,
    expect_expert_grad: bool,
) -> None:
    del fake_codec
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
            num_action_tokens=4,
            action_tokenizer={"name": "fake", "kind": "tokens"},
            expert_hidden_size=16,
            gradient_checkpointing=True,
            ce_loss_weight=ce_weight,
            flow_loss_weight=flow_weight,
            state_prefix_encoding=(
                "discrete_text" if ce_weight == 0.0 else "continuous_projection"
            ),
        ).to_dict()
    )
    policy.set_io(_io())
    policy.zero_grad(set_to_none=True)

    loss, info = policy.forward(_batch())
    loss.backward()

    assert (info["ce_loss"] > 0.0) is (ce_weight > 0.0)
    assert (info["flow_loss"] > 0.0) is (flow_weight > 0.0)
    expected_calls = (
        policy._require_action_expert().vlm_num_layers
        if ce_weight > 0.0 and flow_weight > 0.0
        else 0
    )
    assert calls == expected_calls
    assert _has_nonzero_grad(policy.vlm) is expect_vlm_grad
    assert policy.action_expert is not None
    assert _has_nonzero_grad(policy.action_expert) is expect_expert_grad


def test_gradient_checkpointing_covers_ce_only_sparse_prefix_forward(
    monkeypatch: pytest.MonkeyPatch,
    fake_codec: _FakeTokenCodec,
) -> None:
    del fake_codec
    calls = 0

    def _checkpoint(
        function: Callable[..., torch.Tensor],
        *args: torch.Tensor,
        use_reentrant: bool,
    ) -> torch.Tensor:
        nonlocal calls
        assert use_reentrant is False
        calls += 1
        return function(*args)

    monkeypatch.setattr(vlm_training_module, "checkpoint", _checkpoint)
    policy = _DummyPolicy(
        config=PolicyConfig(
            action_dim=3,
            action_horizon=4,
            n_action_steps=2,
            num_action_tokens=4,
            action_tokenizer={"name": "fake", "kind": "tokens"},
            expert_hidden_size=16,
            gradient_checkpointing=True,
            flow_loss_weight=0.0,
        ).to_dict()
    )
    policy.set_io(_io())
    policy.zero_grad(set_to_none=True)

    loss, info = policy.forward(_batch())
    loss.backward()

    assert info["ce_loss"] > 0.0
    assert info["flow_loss"] == 0.0
    assert calls == policy._require_vlm_ce_prefix_adapter().num_layers
    assert _has_nonzero_grad(policy.vlm)
    assert policy.action_expert is not None
    assert not _has_nonzero_grad(policy.action_expert)


def test_one_forward_ce_matches_standalone_scheduled_ce(
    fake_codec: _FakeTokenCodec,
) -> None:
    del fake_codec
    policy = _policy()
    batch = _batch()

    standalone_ce = policy.compute_ce_loss(batch, actions=batch["action"])
    prepared = prepare_action_token_ce(
        policy,
        batch,
        actions=batch["action"],
        ignore_index=IGNORE_INDEX,
    )
    schedule_attention_mask = build_prepared_action_token_attention_mask(
        policy,
        prepared,
    )
    sample = FlowSample(
        noisy_actions=torch.zeros_like(batch["action"]),
        target_velocity=torch.zeros_like(batch["action"]),
        timesteps=torch.full((batch["action"].shape[0],), 0.5),
        noise=torch.zeros_like(batch["action"]),
    )
    model = policy._require_vlm_action_expert()
    adapter = cast(_FakePrefixAdapter, model.prefix_adapter)
    with policy._autocast_context():
        _flow_loss, integrated_output = model.loss_with_context(
            vlm_inputs=prepared.vlm_inputs,
            layer_attention_mask=schedule_attention_mask,
            expert_context_mask=prepared.supervision.prefix_mask,
            sample=sample,
            action_is_pad=None,
            detach_expert_context=True,
            gradient_checkpointing=False,
        )
        one_forward_ce = compute_prepared_action_token_ce_loss_from_hidden(
            policy,
            prepared,
            hidden=integrated_output.prefix_state.hidden,
            project_hidden_to_logits=adapter.final_hidden_to_logits,
            ignore_index=IGNORE_INDEX,
        )

    torch.testing.assert_close(one_forward_ce, standalone_ce)


def test_ki_flow_lane_matches_fm_prompt_flow_lane(
    fake_codec: _FakeTokenCodec,
) -> None:
    del fake_codec
    policy = _policy()
    batch = _batch()
    norm_batch = policy._prepare_normalized_batch(batch)
    norm_actions = policy._actions_from_batch(norm_batch).to(
        device=batch["action"].device,
        dtype=batch["action"].dtype,
    )
    sample = FlowSample(
        noisy_actions=torch.full_like(norm_actions, 0.25),
        target_velocity=torch.zeros_like(norm_actions),
        timesteps=torch.full((norm_actions.shape[0],), 0.5),
        noise=torch.zeros_like(norm_actions),
    )
    action_is_pad = policy._action_pad_mask_from_batch(
        norm_batch,
        actions=norm_actions,
    )
    model = policy._require_vlm_action_expert()
    fm_vlm_inputs = policy._build_vlm_inputs_for_action_expert(
        batch,
        instructions=None,
    )
    fm_loss, fm_pred = model.loss(
        vlm_inputs=fm_vlm_inputs,
        sample=sample,
        action_is_pad=action_is_pad,
        detach_prefix=False,
        gradient_checkpointing=False,
    )

    prepared = prepare_action_token_ce(
        policy,
        batch,
        actions=batch["action"],
        ignore_index=IGNORE_INDEX,
    )
    schedule_attention_mask = build_prepared_action_token_attention_mask(
        policy,
        prepared,
    )
    expert_context_mask = policy._build_flow_expert_context_mask(batch, prepared)
    ki_flow_loss, ki_output = model.loss_with_context(
        vlm_inputs=prepared.vlm_inputs,
        layer_attention_mask=schedule_attention_mask,
        expert_context_mask=expert_context_mask,
        sample=sample,
        action_is_pad=action_is_pad,
        detach_expert_context=True,
        gradient_checkpointing=False,
    )

    torch.testing.assert_close(ki_flow_loss, fm_loss)
    torch.testing.assert_close(ki_output.pred_velocity, fm_pred)


def test_sparse_prepared_ce_matches_full_logits_and_grads(
    fake_codec: _FakeTokenCodec,
) -> None:
    del fake_codec
    policy = _policy()
    batch = _batch()
    prepared = prepare_action_token_ce(
        policy,
        batch,
        actions=batch["action"],
        ignore_index=IGNORE_INDEX,
    )
    adapter = cast(_FakePrefixAdapter, policy._require_vlm_ce_prefix_adapter())
    base_hidden = torch.randn(
        (*prepared.labels.shape, adapter.hidden_size),
        dtype=torch.float32,
    )

    full_hidden = base_hidden.detach().clone().requires_grad_(True)
    full_loss = compute_prepared_action_token_ce_loss(
        policy,
        prepared,
        logits=adapter.final_hidden_to_logits(full_hidden),
        ignore_index=IGNORE_INDEX,
    )
    full_hidden_grad, full_weight_grad = torch.autograd.grad(
        full_loss,
        (full_hidden, policy.vlm.embed_tokens.weight),
    )

    sparse_hidden = base_hidden.detach().clone().requires_grad_(True)
    projected_rows: list[int] = []

    def project_hidden_to_logits(
        hidden: torch.Tensor,
        *,
        candidate_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        projected_rows.append(int(hidden.shape[0]))
        return adapter.final_hidden_to_logits(
            hidden,
            candidate_token_ids=candidate_token_ids,
        )

    sparse_loss = compute_prepared_action_token_ce_loss_from_hidden(
        policy,
        prepared,
        hidden=sparse_hidden,
        project_hidden_to_logits=project_hidden_to_logits,
        ignore_index=IGNORE_INDEX,
        projection_chunk_size=2,
    )
    sparse_hidden_grad, sparse_weight_grad = torch.autograd.grad(
        sparse_loss,
        (sparse_hidden, policy.vlm.embed_tokens.weight),
    )

    supervised_count = int(prepared.labels.ne(IGNORE_INDEX).sum().item())
    assert sum(projected_rows) == supervised_count
    assert all(count <= 2 for count in projected_rows)
    torch.testing.assert_close(sparse_loss, full_loss)
    torch.testing.assert_close(sparse_hidden_grad, full_hidden_grad)
    torch.testing.assert_close(sparse_weight_grad, full_weight_grad)


def test_sparse_prepared_ce_matches_full_logits_for_partial_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        policy_module,
        "make_codec",
        lambda _cfg: _FakeVariableTokenCodec(codebook_size=6),
    )
    policy = _DummyFastPolicy(
        config=PolicyConfig(
            action_dim=3,
            action_horizon=4,
            n_action_steps=2,
            num_action_tokens=6,
            action_tokenizer={"name": "fast", "kind": "tokens"},
            expert_hidden_size=16,
            ce_supervise_eos=True,
        ).to_dict()
    )
    policy.set_io(_io())
    batch = _batch()
    prepared = prepare_action_token_ce(
        policy,
        batch,
        actions=batch["action"],
        ignore_index=IGNORE_INDEX,
    )
    assert prepared.token_lengths is not None
    adapter = cast(_FakePrefixAdapter, policy._require_vlm_ce_prefix_adapter())
    base_hidden = torch.randn(
        (*prepared.labels.shape, adapter.hidden_size),
        dtype=torch.float32,
    )

    full_hidden = base_hidden.detach().clone().requires_grad_(True)
    full_loss = compute_prepared_action_token_ce_loss(
        policy,
        prepared,
        logits=adapter.final_hidden_to_logits(full_hidden),
        ignore_index=IGNORE_INDEX,
    )
    full_hidden_grad, full_weight_grad = torch.autograd.grad(
        full_loss,
        (full_hidden, policy.vlm.embed_tokens.weight),
    )

    sparse_hidden = base_hidden.detach().clone().requires_grad_(True)
    sparse_loss = compute_prepared_action_token_ce_loss_from_hidden(
        policy,
        prepared,
        hidden=sparse_hidden,
        project_hidden_to_logits=adapter.final_hidden_to_logits,
        ignore_index=IGNORE_INDEX,
        projection_chunk_size=2,
    )
    sparse_hidden_grad, sparse_weight_grad = torch.autograd.grad(
        sparse_loss,
        (sparse_hidden, policy.vlm.embed_tokens.weight),
    )

    torch.testing.assert_close(sparse_loss, full_loss)
    torch.testing.assert_close(sparse_hidden_grad, full_hidden_grad)
    torch.testing.assert_close(sparse_weight_grad, full_weight_grad)


def test_action_vocab_scope_sparse_ce_matches_candidate_sliced_full_ce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        policy_module,
        "make_codec",
        lambda _cfg: _FakeVariableTokenCodec(codebook_size=6),
    )
    policy = _DummyFastPolicy(
        config=PolicyConfig(
            action_dim=3,
            action_horizon=4,
            n_action_steps=2,
            num_action_tokens=6,
            action_tokenizer={"name": "fast", "kind": "tokens"},
            expert_hidden_size=16,
            ce_supervise_eos=True,
            ce_vocab_scope="action",
        ).to_dict()
    )
    policy.set_io(_io())
    prepared = prepare_action_token_ce(
        policy,
        _batch(),
        ignore_index=IGNORE_INDEX,
    )
    adapter = cast(_FakePrefixAdapter, policy._require_vlm_ce_prefix_adapter())
    candidate_ids = build_action_token_ce_candidate_token_ids(
        policy,
        device=prepared.labels.device,
    )
    assert candidate_ids is not None
    vocab = policy._require_action_token_vocabulary()
    assert candidate_ids.tolist() == [
        *list(vocab.token_ids),
        policy._eos_token_id,
    ]
    assert vocab.mask_token_id not in candidate_ids.tolist()

    base_hidden = torch.randn(
        (*prepared.labels.shape, adapter.hidden_size),
        dtype=torch.float32,
    )
    full_hidden = base_hidden.detach().clone().requires_grad_(True)
    full_loss = compute_prepared_action_token_ce_loss(
        policy,
        prepared,
        logits=adapter.final_hidden_to_logits(full_hidden),
        ignore_index=IGNORE_INDEX,
    )
    full_hidden_grad, full_weight_grad = torch.autograd.grad(
        full_loss,
        (full_hidden, policy.vlm.embed_tokens.weight),
    )

    sparse_hidden = base_hidden.detach().clone().requires_grad_(True)
    projected_widths: list[int] = []

    def project_hidden_to_logits(
        hidden: torch.Tensor,
        *,
        candidate_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert candidate_token_ids is not None
        logits = adapter.final_hidden_to_logits(
            hidden,
            candidate_token_ids=candidate_token_ids,
        )
        projected_widths.append(int(logits.shape[-1]))
        return logits

    sparse_loss = compute_prepared_action_token_ce_loss_from_hidden(
        policy,
        prepared,
        hidden=sparse_hidden,
        project_hidden_to_logits=project_hidden_to_logits,
        ignore_index=IGNORE_INDEX,
        projection_chunk_size=2,
    )
    sparse_hidden_grad, sparse_weight_grad = torch.autograd.grad(
        sparse_loss,
        (sparse_hidden, policy.vlm.embed_tokens.weight),
    )

    assert projected_widths
    assert set(projected_widths) == {int(candidate_ids.numel())}
    torch.testing.assert_close(sparse_loss, full_loss)
    torch.testing.assert_close(sparse_hidden_grad, full_hidden_grad)
    torch.testing.assert_close(sparse_weight_grad, full_weight_grad)


def test_forward_projects_only_supervised_ce_hidden_rows(
    fake_codec: _FakeTokenCodec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_codec
    policy = _policy()
    batch = _batch()
    prepared = prepare_action_token_ce(
        policy,
        batch,
        actions=batch["action"],
        ignore_index=IGNORE_INDEX,
    )
    supervised_count = int(prepared.labels.ne(IGNORE_INDEX).sum().item())
    adapter = cast(
        _FakePrefixAdapter, policy._require_vlm_action_expert().prefix_adapter
    )
    original_project = adapter.final_hidden_to_logits
    projected_shapes: list[tuple[int, ...]] = []

    def project_hidden_to_logits(
        hidden: torch.Tensor,
        *,
        candidate_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        projected_shapes.append(tuple(hidden.shape))
        return original_project(hidden, candidate_token_ids=candidate_token_ids)

    monkeypatch.setattr(adapter, "final_hidden_to_logits", project_hidden_to_logits)

    loss, info = policy.forward(batch)

    assert loss.requires_grad
    assert info["ce_loss"] > 0.0
    assert sum(shape[0] for shape in projected_shapes) == supervised_count
    assert all(len(shape) == 2 for shape in projected_shapes)


def test_combined_flow_only_weight_keeps_vlm_gradient_insulated(
    fake_codec: _FakeTokenCodec,
) -> None:
    del fake_codec
    policy = _policy(state_prefix_encoding="discrete_text")
    policy.config.ce_loss_weight = 0.0
    policy.zero_grad(set_to_none=True)

    loss, info = policy.forward(_batch())
    loss.backward()

    assert info["ce_loss"] == 0.0
    assert info["flow_loss"] > 0.0
    assert not _has_nonzero_grad(policy.vlm)
    assert policy.action_expert is not None
    assert _has_nonzero_grad(policy.action_expert)


def test_combined_ce_only_weight_does_not_update_expert(
    fake_codec: _FakeTokenCodec,
) -> None:
    del fake_codec
    policy = _policy()
    policy.config.flow_loss_weight = 0.0
    policy.zero_grad(set_to_none=True)

    loss, info = policy.forward(_batch())
    loss.backward()

    assert info["ce_loss"] > 0.0
    assert info["flow_loss"] == 0.0
    assert _has_nonzero_grad(policy.vlm)
    assert policy.action_expert is not None
    assert not _has_nonzero_grad(policy.action_expert)


def test_combined_flow_only_weight_skips_ce_projection(
    fake_codec: _FakeTokenCodec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_codec
    policy = _policy(state_prefix_encoding="discrete_text")
    policy.config.ce_loss_weight = 0.0
    adapter = policy._require_vlm_action_expert().prefix_adapter

    def _fail_final_hidden_to_logits(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("flow-only KI forward should not project CE logits")

    monkeypatch.setattr(
        adapter,
        "final_hidden_to_logits",
        _fail_final_hidden_to_logits,
    )

    loss, info = policy.forward(_batch())

    assert loss.requires_grad
    assert info["ce_loss"] == 0.0
    assert info["flow_loss"] > 0.0


def test_combined_ce_only_weight_skips_expert_suffix(
    fake_codec: _FakeTokenCodec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_codec
    policy = _policy()
    policy.config.flow_loss_weight = 0.0
    model = policy._require_vlm_action_expert()

    def _fail_loss_with_context(*args: object, **kwargs: object) -> object:
        raise AssertionError("CE-only KI forward should not run the expert suffix")

    monkeypatch.setattr(model, "loss_with_context", _fail_loss_with_context)

    loss, info = policy.forward(_batch())

    assert loss.requires_grad
    assert info["ce_loss"] > 0.0
    assert info["flow_loss"] == 0.0


def test_one_forward_flow_masks_teacher_forced_action_tokens_from_expert_context(
    fake_codec: _FakeTokenCodec,
) -> None:
    del fake_codec
    policy = _policy()
    batch = _batch()
    prepared = prepare_action_token_ce(
        policy,
        batch,
        actions=batch["action"],
        ignore_index=IGNORE_INDEX,
    )
    vocab = policy._require_action_token_vocabulary()
    mutated_inputs = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in prepared.vlm_inputs.items()
    }
    prediction_mask = prepared.supervision.prediction_position_mask
    mutated_inputs["input_ids"][prediction_mask] = int(vocab.token_max)

    schedule_attention_mask = build_prepared_action_token_attention_mask(
        policy,
        prepared,
    )

    sample = FlowSample(
        noisy_actions=torch.full_like(batch["action"], 0.25),
        target_velocity=torch.zeros_like(batch["action"]),
        timesteps=torch.full((batch["action"].shape[0],), 0.5),
        noise=torch.zeros_like(batch["action"]),
    )
    model = policy._require_vlm_action_expert()
    prefix_mask = prepared.supervision.prefix_mask
    full_valid_mask = prepared.vlm_inputs["attention_mask"].bool()

    original = model.forward_with_context(
        vlm_inputs=prepared.vlm_inputs,
        layer_attention_mask=schedule_attention_mask,
        expert_context_mask=prefix_mask,
        noisy_actions=sample.noisy_actions,
        timesteps=sample.timesteps,
        action_is_pad=None,
        detach_expert_context=True,
        gradient_checkpointing=False,
    )
    mutated = model.forward_with_context(
        vlm_inputs=mutated_inputs,
        layer_attention_mask=schedule_attention_mask,
        expert_context_mask=prefix_mask,
        noisy_actions=sample.noisy_actions,
        timesteps=sample.timesteps,
        action_is_pad=None,
        detach_expert_context=True,
        gradient_checkpointing=False,
    )
    leaky = model.forward_with_context(
        vlm_inputs=mutated_inputs,
        layer_attention_mask=schedule_attention_mask,
        expert_context_mask=full_valid_mask,
        noisy_actions=sample.noisy_actions,
        timesteps=sample.timesteps,
        action_is_pad=None,
        detach_expert_context=True,
        gradient_checkpointing=False,
    )

    assert torch.allclose(original.pred_velocity, mutated.pred_velocity)
    assert not torch.allclose(original.pred_velocity, leaky.pred_velocity)


def test_flow_context_mask_keeps_only_fm_prompt_prefix() -> None:
    prepared_vlm_inputs = {
        "input_ids": torch.tensor(
            [
                [0, 10, 11, 12, 90, 91],
                [20, 21, 22, 23, 90, 91],
            ]
        )
    }
    prepared_prefix_mask = torch.tensor(
        [
            [False, True, True, True, False, False],
            [True, True, True, True, False, False],
        ]
    )
    fm_vlm_inputs = {
        "input_ids": torch.tensor(
            [
                [10, 11],
                [0, 20],
            ]
        ),
        "attention_mask": torch.tensor(
            [
                [True, True],
                [False, True],
            ]
        ),
    }

    context_mask = _context_mask_matching_fm_prefix(
        prepared_vlm_inputs=prepared_vlm_inputs,
        prepared_prefix_mask=prepared_prefix_mask,
        fm_vlm_inputs=fm_vlm_inputs,
    )

    assert context_mask.tolist() == [
        [False, True, True, False, False, False],
        [True, False, False, False, False, False],
    ]


def test_predict_action_chunk_uses_continuous_expert_only(
    fake_codec: _FakeTokenCodec,
) -> None:
    policy = _policy()
    actions, info = policy.predict_action_chunk(
        {"observation.state": torch.zeros((1, 1, 2), dtype=torch.float32)},
        generator=torch.Generator().manual_seed(0),
    )

    assert actions.shape == (1, 4, 3)
    assert info["flow_sampling_steps"] == policy.config.flow_sampling_steps
    assert not fake_codec.detokenize_called


def test_save_load_preserves_config_io_and_norm_stats(
    fake_codec: _FakeTokenCodec, tmp_path
) -> None:
    del fake_codec
    policy = _policy()
    stats = {
        "observation.state": {
            "min": torch.full((2,), -1.0),
            "max": torch.full((2,), 1.0),
        },
        "action": {
            "min": torch.full((3,), -0.5),
            "max": torch.full((3,), 0.5),
        },
    }
    modes = {"observation.state": NormMode.MIN_MAX, "action": NormMode.MIN_MAX}
    policy.set_normalization_stats(stats, modes=modes)

    save_dir = tmp_path / "policy"
    policy.save_pretrained(save_dir)
    loaded = _DummyPolicy.from_pretrained(save_dir)

    assert loaded.config.name == policy.config.name
    assert loaded.io is not None
    assert loaded.io.action_dim == 3
    assert loaded.normalize is not None
    assert loaded.unnormalize_action is not None
