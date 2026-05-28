"""Shared Smol policy test utilities."""

from __future__ import annotations

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
    inputs = _mock_build_vlm_inputs_impl(
        self,
        obs,
        action_token_strings=action_token_strings,
        instructions=instructions,
    )
    if action_token_strings is None:
        return inputs

    input_ids = inputs["input_ids"]
    seq_len = int(input_ids.shape[1])
    for row, action_str in enumerate(action_token_strings):
        n_action_tokens = int(action_str.count("<|action_"))
        if n_action_tokens <= 0:
            continue
        start = max(0, seq_len - n_action_tokens)
        token_ids = torch.arange(
            int(self._require_action_token_vocabulary().token_min),
            int(self._require_action_token_vocabulary().token_min) + (seq_len - start),
            dtype=torch.long,
            device=input_ids.device,
        )
        input_ids[row, start:] = token_ids
    return inputs


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
        del actions, kwargs
        return [
            torch.tensor([1, 2, 3], dtype=torch.long),
            torch.tensor([4], dtype=torch.long),
        ]

    def detokenize(self, tokens, **kwargs):
        del kwargs
        bsz = tokens.shape[0]
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
        bsz = tokens.shape[0]
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


class _DummyVLM(nn.Module):
    def __init__(self, vocab_size: int = 256, hidden_dim: int = 16) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self._generate_action_ids: list[int] | None = None
        self._decode_step = 0
        self._last_prefill_attention_rank: int | None = None
        self._last_decode_attention_rank: int | None = None

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

    def _token_for_step(self, step: int) -> int:
        if not self._generate_action_ids:
            return 0
        return int(
            self._generate_action_ids[min(step, len(self._generate_action_ids) - 1)]
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        **kwargs,
    ):
        del kwargs
        if labels is not None:
            return _VLMOutputs(
                loss=torch.tensor(1.0, dtype=torch.float32, requires_grad=True)
            )

        if past_key_values is None:
            self._decode_step = 0
            if attention_mask is not None:
                self._last_prefill_attention_rank = int(attention_mask.dim())
        else:
            self._decode_step += 1
            if attention_mask is not None:
                self._last_decode_attention_rank = int(attention_mask.dim())

        bsz, seq_len = int(input_ids.shape[0]), int(input_ids.shape[1])
        token_id = self._token_for_step(self._decode_step)
        logits = torch.full(
            (bsz, seq_len, self.lm_head.out_features),
            fill_value=-1e9,
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits[:, -1, token_id] = 0.0
        return _VLMOutputs(logits=logits, past_key_values={"step": self._decode_step})

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        if past_key_values is not None and kwargs.get("cache_position") is None:
            raise TypeError("cache_position is required for cached decode")
        if past_key_values is not None and int(kwargs.get("logits_to_keep", 0)) < 1:
            raise TypeError(
                "positive logits_to_keep is required for cached Smol decode"
            )
        out = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
        }
        out.update(kwargs)
        return out


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

    def __call__(self, text=None, padding=True, return_tensors="pt", **kwargs):
        del padding, return_tensors, kwargs
        bsz = len(text) if isinstance(text, list) else 1
        seq_len = 10
        return _DummyBatchEncoding(
            {
                "input_ids": torch.zeros(bsz, seq_len, dtype=torch.long),
                "attention_mask": torch.ones(bsz, seq_len, dtype=torch.long),
            }
        )


class _DummyBatchEncoding(dict):
    def to(self, device):
        for key, value in list(self.items()):
            if isinstance(value, torch.Tensor):
                self[key] = value.to(device)
        return self


class _DummyProcessor:
    def __init__(self) -> None:
        class _DummyImageProcessor:
            max_image_size = {"longest_edge": 8}
            image_mean = [0.5, 0.5, 0.5]
            image_std = [0.5, 0.5, 0.5]
            do_resize = False
            do_image_splitting = False

        self.tokenizer = _DummyTokenizerWrapper()
        self.image_token = "<image>"
        self.image_seq_len = 4
        self.image_processor = _DummyImageProcessor()
        self.last_kwargs: dict[str, object] | None = None
        self.expand_text_calls = 0

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False
    ):
        del tokenize, add_generation_prompt
        # Keep one <image> token per image entry so native Smol path image-count check passes.
        image_count = 0
        if messages and isinstance(messages, list):
            for msg in messages:
                for part in msg.get("content", []):
                    if part.get("type") == "image":
                        image_count += 1
        return "<image>" * image_count

    def expand_text_with_image_tokens(self, text, image_rows=None, image_cols=None):
        del image_rows, image_cols
        self.expand_text_calls += 1
        return text

    def _check_special_mm_tokens(self, text, text_inputs, modalities=None):
        del text, text_inputs, modalities

    def __call__(
        self, text=None, images=None, padding=True, return_tensors="pt", **kwargs
    ):
        del padding, return_tensors
        self.last_kwargs = dict(kwargs)
        bsz = len(text) if text else 1
        seq_len = 10
        out = {
            "input_ids": torch.zeros(bsz, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(bsz, seq_len, dtype=torch.long),
        }
        if images is not None and len(images) > 0 and len(images[0]) > 0:
            first = images[0][0]
            out["pixel_values"] = torch.randn(
                bsz, 1, 3, 8, 8, dtype=torch.float32, device=first.device
            )
        return _DummyBatchEncoding(out)


def _patch_smol(
    monkeypatch, *, tokenizer: nn.Module, processor: _DummyProcessor, vlm: _DummyVLM
) -> None:
    def _load_vlm(self, model_name, dtype_str):
        del model_name, dtype_str
        self.vlm = vlm
        self.processor = processor

    monkeypatch.setattr(
        "praxis.policies.bar.backbones.smol.smol_base.SmolPolicyBase._load_vlm",
        _load_vlm,
    )
    monkeypatch.setattr(
        "praxis.policies.bar.core.vlm.base.make_codec",
        lambda _cfg: tokenizer,
    )
    monkeypatch.setattr(
        "praxis.policies.bar.backbones.smol.smol_base.SmolPolicyBase._build_vlm_prefix_adapter",
        lambda self: BasicPrefixAdapter(self.vlm),
    )
