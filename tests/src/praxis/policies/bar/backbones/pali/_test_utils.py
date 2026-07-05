# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Shared Pali policy test utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
import torch.nn as nn

from ...core._test_utils import (
    BasicPrefixAdapter,
    basic_stats_for_io as _stats_for_io_impl,
    make_basic_policy_io as _make_io_impl,
    mock_build_vlm_inputs_all_ignore_basic as _mock_build_vlm_inputs_all_ignore_impl,
    mock_build_vlm_inputs_basic as _mock_build_vlm_inputs_impl,
)


def _make_io(*, with_image: bool = False):
    return _make_io_impl(with_image=with_image)


def _stats_for_io(io):
    return _stats_for_io_impl(io)


def _mock_build_vlm_inputs(self, obs, *, action_token_strings=None, instructions=None):
    return _mock_build_vlm_inputs_impl(
        self,
        obs,
        action_token_strings=action_token_strings,
        instructions=instructions,
    )


def _mock_build_vlm_inputs_all_ignore(
    self, obs, *, action_token_strings=None, instructions=None
):
    return _mock_build_vlm_inputs_all_ignore_impl(
        self,
        obs,
        action_token_strings=action_token_strings,
        instructions=instructions,
    )


class _DummyFASTTokenizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._codebook_size = 8
        self._pad_token_id = 7
        self.max_token_length = 6
        self.stats_set_calls = 0
        self.normalize_action = None
        self.unnormalize_action = None
        self.io = None

    @property
    def codebook_size(self) -> int:
        return self._codebook_size

    @property
    def variable_length_codes(self) -> bool:
        return True

    @property
    def max_code_seq_len(self) -> int | None:
        return getattr(self, "max_token_length", None)

    def tokenize(self, actions, *, pad=False, **kwargs):
        del pad, kwargs
        bsz = int(actions.shape[0])
        out: list[torch.Tensor] = []
        for i in range(bsz):
            n = 1 + (i % 3)
            out.append(torch.arange(1, n + 1, dtype=torch.long))
        return out

    def detokenize(self, tokens, **kwargs):
        del kwargs
        if not isinstance(tokens, torch.Tensor):
            max_len = max((len(row) for row in tokens), default=0)
            tensor = torch.zeros((len(tokens), max_len), dtype=torch.long)
            for row_idx, row in enumerate(tokens):
                if row:
                    tensor[row_idx, : len(row)] = torch.tensor(row, dtype=torch.long)
            tokens = tensor
        bsz = int(tokens.shape[0])
        return torch.zeros((bsz, 2, 2), dtype=torch.float32, device=tokens.device)

    def set_io(self, io):
        self.io = io

    def set_normalization_stats(self, stats, modes=None):
        del stats, modes
        self.stats_set_calls += 1
        self.normalize_action = object()
        self.unnormalize_action = object()


class _DummyFixedTokenizer(nn.Module):
    def __init__(self, fixed_len: int = 4, codebook: int = 16) -> None:
        super().__init__()
        self._fixed_code_seq_len = fixed_len
        self._codebook_size = codebook
        self.stats_set_calls = 0
        self.normalize_action = None
        self.unnormalize_action = None
        self.io = None

    @property
    def codebook_size(self) -> int:
        return self._codebook_size

    @property
    def max_code_seq_len(self) -> int | None:
        return self._fixed_code_seq_len

    @property
    def supports_prefix_decode(self) -> bool:
        return True

    def tokenize(self, actions, **kwargs):
        del kwargs
        bsz = int(actions.shape[0])
        return torch.randint(0, self._codebook_size, (bsz, self._fixed_code_seq_len))

    def detokenize(self, tokens, **kwargs):
        del kwargs
        if not isinstance(tokens, torch.Tensor):
            max_len = max((len(row) for row in tokens), default=0)
            tensor = torch.zeros((len(tokens), max_len), dtype=torch.long)
            for row_idx, row in enumerate(tokens):
                if row:
                    tensor[row_idx, : len(row)] = torch.tensor(row, dtype=torch.long)
            tokens = tensor
        bsz = int(tokens.shape[0])
        return torch.zeros((bsz, 2, 2), dtype=torch.float32, device=tokens.device)

    def set_io(self, io):
        self.io = io

    def set_normalization_stats(self, stats, modes=None):
        del stats, modes
        self.stats_set_calls += 1
        self.normalize_action = object()
        self.unnormalize_action = object()


@dataclass
class _VLMOutputs:
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    past_key_values: object | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None


class _DummyVLM(nn.Module):
    def __init__(self, vocab_size: int = 256, hidden_dim: int = 16) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.vision_tower = nn.Linear(1, 1, bias=False)
        self._generate_action_ids: list[int] | None = None
        self.last_generate_kwargs: dict[str, object] | None = None
        self.last_forward_kwargs: dict[str, object] | None = None
        self.last_prepare_kwargs: dict[str, object] | None = None
        self.prepare_calls: list[dict[str, object]] = []
        self.forward_calls: list[dict[str, object]] = []

    def get_input_embeddings(self):
        return self.embed_tokens

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def resize_token_embeddings(self, new_size: int, **kwargs):
        del kwargs
        old_size = self.embed_tokens.weight.shape[0]
        hidden = self.embed_tokens.weight.shape[1]
        if new_size == old_size:
            return

        new_embed = nn.Embedding(new_size, hidden)
        copy_size = min(old_size, new_size)
        new_embed.weight.data[:copy_size] = self.embed_tokens.weight.data[:copy_size]
        self.embed_tokens = new_embed

        new_lm_head = nn.Linear(hidden, new_size, bias=False)
        old_out = self.lm_head.weight.shape[0]
        copy_out = min(old_out, new_size)
        new_lm_head.weight.data[:copy_out] = self.lm_head.weight.data[:copy_out]
        self.lm_head = new_lm_head

    def forward(self, input_ids=None, labels=None, **kwargs):
        self.last_forward_kwargs = dict(kwargs)
        self.forward_calls.append(
            {
                "input_ids_shape": tuple(input_ids.shape)
                if input_ids is not None
                else None,
                "labels_present": labels is not None,
                **dict(kwargs),
            }
        )
        if labels is not None:
            return _VLMOutputs(
                loss=torch.tensor(1.0, dtype=torch.float32, requires_grad=True)
            )

        if input_ids is None:
            raise ValueError("Dummy VLM forward expects input_ids during inference.")
        bsz = int(input_ids.shape[0])
        seq_len = int(input_ids.shape[1])
        logits = torch.full(
            (bsz, seq_len, self.lm_head.out_features),
            fill_value=-1e9,
            dtype=torch.float32,
            device=input_ids.device,
        )
        token_id = 0
        past_key_values = kwargs.get("past_key_values")
        step_idx = 0
        if isinstance(past_key_values, dict):
            step_idx = int(past_key_values.get("step", 0))
        if self._generate_action_ids:
            idx = min(step_idx, len(self._generate_action_ids) - 1)
            token_id = int(self._generate_action_ids[idx])
        logits[:, -1, token_id] = 0.0
        hidden_states = None
        if bool(kwargs.get("output_hidden_states", False)):
            hidden = self.embed_tokens(input_ids)
            hidden_states = (hidden, hidden + 0.5)
        return _VLMOutputs(
            logits=logits,
            past_key_values={"step": step_idx + 1},
            hidden_states=hidden_states,
        )

    def prepare_inputs_for_generation(self, **kwargs):
        self.last_prepare_kwargs = dict(kwargs)
        self.prepare_calls.append(dict(kwargs))
        return dict(kwargs)

    def generate(self, input_ids=None, max_new_tokens: int = 1, **kwargs):
        self.last_generate_kwargs = dict(kwargs)
        if input_ids is None:
            raise ValueError("Dummy VLM generate expects input_ids.")
        bsz = int(input_ids.shape[0])
        max_new_tokens = int(max_new_tokens)

        if self._generate_action_ids:
            ids = []
            for i in range(max_new_tokens):
                idx = min(i, len(self._generate_action_ids) - 1)
                ids.append(int(self._generate_action_ids[idx]))
            append = torch.tensor(
                ids, dtype=torch.long, device=input_ids.device
            ).unsqueeze(0)
            append = append.expand(bsz, -1)
        else:
            append = torch.zeros(
                (bsz, max_new_tokens), dtype=torch.long, device=input_ids.device
            )
        return torch.cat([input_ids, append], dim=1)


class _DummyTokenizerWrapper:
    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._next_id = 100
        self.eos_token_id = 99
        self.pad_token_id = 0
        self.padding_side = "right"

    def get_vocab(self):
        return dict(self._vocab)

    def add_special_tokens(self, special_tokens_dict):
        tokens = special_tokens_dict.get("additional_special_tokens", [])
        added = 0
        for t in tokens:
            if t not in self._vocab:
                self._vocab[t] = self._next_id
                self._next_id += 1
                added += 1
        return added

    def convert_tokens_to_ids(self, token: str) -> int:
        return int(self._vocab.get(token, 0))


class _DummyBatchEncoding(dict):
    def to(self, device):
        for key, value in list(self.items()):
            if isinstance(value, torch.Tensor):
                self[key] = value.to(device)
        return self


class _DummyProcessor:
    def __init__(self) -> None:
        self.tokenizer = _DummyTokenizerWrapper()

    def __call__(
        self, text=None, images=None, padding=True, return_tensors="pt", **kwargs
    ):
        del images, padding, return_tensors
        bsz = len(text) if isinstance(text, list) else 1
        suffix = kwargs.get("suffix")
        seq_len = 10
        input_ids = torch.zeros(bsz, seq_len, dtype=torch.long)

        if suffix is not None:
            suffix_list = suffix if isinstance(suffix, list) else [suffix]
            token_pat = re.compile(r"<\|[^|]+\|>")
            eos_id = int(getattr(self.tokenizer, "eos_token_id", 1))
            for i, sfx in enumerate(suffix_list):
                token_strs = token_pat.findall(str(sfx))
                token_ids = [
                    self.tokenizer.convert_tokens_to_ids(tok) for tok in token_strs
                ]
                suffix_ids = token_ids + [eos_id]
                n = min(len(suffix_ids), seq_len)
                input_ids[i, seq_len - n :] = torch.tensor(
                    suffix_ids[-n:], dtype=torch.long
                )

        labels = input_ids.clone()
        return _DummyBatchEncoding(
            {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": torch.ones(bsz, seq_len, dtype=torch.long),
                "pixel_values": torch.randn(bsz, 3, 4, 4, dtype=torch.float32),
            }
        )


def _patch_pali(
    monkeypatch, *, tokenizer: nn.Module, processor: _DummyProcessor, vlm: _DummyVLM
) -> None:
    def _load_vlm(self, model_name, dtype_str):
        del model_name, dtype_str
        self.vlm = vlm
        self.processor = processor

    monkeypatch.setattr(
        "praxis.policies.bar.backbones.pali.pali_base.PaliPolicyBase._load_vlm",
        _load_vlm,
    )
    monkeypatch.setattr(
        "praxis.policies.bar.core.vlm.base.make_codec",
        lambda _cfg: tokenizer,
    )
    monkeypatch.setattr(
        "praxis.policies.bar.backbones.pali.pali_base.PaliPolicyBase._build_vlm_prefix_adapter",
        lambda self: BasicPrefixAdapter(self.vlm),
    )
