from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers.feature_extraction_utils import BatchFeature

from praxis.core.types import PolicyIO
from praxis.policies.bar.variants import FASTQwenPolicy

from ...core._test_utils import (
    BasicPrefixAdapter,
    bounded_minmax_stats_for_io as _bounded_minmax_stats_for_io,
    make_basic_policy_io as _make_basic_policy_io,
    mock_build_vlm_inputs_all_ignore_basic as _mock_build_vlm_inputs_all_ignore_basic,
)


class DummyFASTTokenizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._codebook_size = 8
        self._pad_token_id = 7
        self.max_token_length = 6
        self.last_detok_tokens: torch.Tensor | None = None
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

    def tokenize(self, actions, *, pad=True, pad_value=None, return_lengths=False):
        batch_size = int(actions.shape[0])
        rows = [
            torch.tensor([1, 2, 3], dtype=torch.long),
            torch.tensor([4], dtype=torch.long),
        ]
        rows = rows[:batch_size]
        lengths = torch.tensor([row.shape[0] for row in rows], dtype=torch.long)
        if not pad:
            return (rows, lengths) if return_lengths else rows
        pad_value = self._pad_token_id if pad_value is None else int(pad_value)
        out = torch.full(
            (len(rows), self.max_token_length), pad_value, dtype=torch.long
        )
        for index, row in enumerate(rows):
            out[index, : row.shape[0]] = row
        return (out, lengths) if return_lengths else out

    def detokenize(self, tokens, **kwargs):
        del kwargs
        self.last_detok_tokens = tokens.detach().clone()
        batch_size = tokens.shape[0]
        return torch.zeros(
            (batch_size, 2, 2), dtype=torch.float32, device=tokens.device
        )

    def set_io(self, io):
        self.io = io

    def set_normalization_stats(self, stats, modes=None):
        del stats, modes
        self.stats_set_calls += 1
        self.normalize_action = object()
        self.unnormalize_action = object()


@dataclass
class VLMOutputs:
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    past_key_values: object | None = None


class DummyVLM(nn.Module):
    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 16) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self._vocab_size = vocab_size
        self._generate_action_ids: list[int] | None = None
        self._decode_step = 0
        self._last_prefill_attention_rank: int | None = None
        self._last_decode_attention_rank: int | None = None
        self._last_prefill_logits_to_keep: int | None = None
        self._last_decode_logits_to_keep: int | None = None

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_rope_index(
        self,
        input_ids,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
    ):
        del image_grid_thw, video_grid_thw, attention_mask
        batch_size, seq_len = int(input_ids.shape[0]), int(input_ids.shape[1])
        position_ids = (
            torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        return position_ids, None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

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
        self._vocab_size = new_size

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
        logits_to_keep = kwargs.get("logits_to_keep")
        if labels is not None:
            loss = torch.tensor(1.5, dtype=torch.float32, requires_grad=True)
            return VLMOutputs(loss=loss)

        if input_ids is None:
            raise ValueError("DummyVLM.forward requires input_ids for generation.")

        if past_key_values is None:
            self._decode_step = 0
            if attention_mask is not None:
                self._last_prefill_attention_rank = int(attention_mask.dim())
            if logits_to_keep is not None:
                self._last_prefill_logits_to_keep = int(logits_to_keep)
        else:
            self._decode_step += 1
            if attention_mask is not None:
                self._last_decode_attention_rank = int(attention_mask.dim())
            if logits_to_keep is not None:
                self._last_decode_logits_to_keep = int(logits_to_keep)

        batch_size, seq_len = int(input_ids.shape[0]), int(input_ids.shape[1])
        token_id = self._token_for_step(self._decode_step)
        logits = torch.full(
            (batch_size, seq_len, self._vocab_size),
            fill_value=-1e9,
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits[:, -1, token_id] = 0.0
        return VLMOutputs(logits=logits, past_key_values={"step": self._decode_step})

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        if past_key_values is not None and kwargs.get("cache_position") is None:
            raise TypeError("cache_position is required for cached decode")
        if past_key_values is not None and kwargs.get("logits_to_keep") != 1:
            raise TypeError("logits_to_keep=1 is required for cached Qwen decode")
        out = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
        }
        out.update(kwargs)
        return out


IM_END_TOKEN_ID = 99


class DummyTokenizerWrapper:
    def __init__(self) -> None:
        self._vocab: dict[str, int] = {"<|im_end|>": IM_END_TOKEN_ID}
        self._next_id = 100
        self.eos_token_id = IM_END_TOKEN_ID
        self.padding_side = "right"
        self.pad_token_id = 0
        self.last_text: list[str] | None = None

    def get_vocab(self):
        return dict(self._vocab)

    def add_special_tokens(self, special_tokens_dict):
        tokens = special_tokens_dict.get("additional_special_tokens", [])
        added = 0
        for token in tokens:
            if token not in self._vocab:
                self._vocab[token] = self._next_id
                self._next_id += 1
                added += 1
        return added

    def convert_tokens_to_ids(self, token):
        return self._vocab.get(token, -1)

    def __call__(self, text, padding=True, return_tensors="pt"):
        del padding
        if not isinstance(text, list):
            text = [text]
        self.last_text = list(text)
        batch_size = len(text)
        seq_len = 10
        return {
            "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        }


class DummyProcessor:
    def __init__(self) -> None:
        self.tokenizer = DummyTokenizerWrapper()
        self.image_token = "<|image_pad|>"

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False
    ):
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
                        parts.append(item.get("text", ""))
            elif isinstance(content, str):
                parts.append(content)
        return " ".join(parts)

    def _check_special_mm_tokens(self, text, text_inputs, modalities=None):
        del text, text_inputs, modalities

    def __call__(
        self,
        text=None,
        images=None,
        videos=None,
        padding=True,
        return_tensors="pt",
        **kwargs,
    ):
        del videos, padding, kwargs
        if isinstance(text, str):
            text = [text]
        self.tokenizer.last_text = list(text) if text is not None else None
        batch_size = len(text) if text else 1
        seq_len = 10
        first_images = images[0] if images and images[0] is not None else None
        num_images = len(first_images) if first_images is not None else 0
        device = first_images[0].device if first_images else torch.device("cpu")
        return BatchFeature(
            {
                "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
                "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
                "pixel_values": torch.zeros(
                    (max(1, batch_size * max(1, num_images)), 4),
                    dtype=torch.float32,
                    device=device,
                ),
                "image_grid_thw": torch.zeros(
                    (batch_size * num_images, 3), dtype=torch.long, device=device
                ),
            },
            tensor_type=return_tensors,
        )


def make_io() -> PolicyIO:
    return _make_basic_policy_io(with_image=False)


def stats_for_io(io: PolicyIO) -> dict[str, dict[str, torch.Tensor]]:
    return _bounded_minmax_stats_for_io(io)


def mock_build_vlm_inputs(self, obs, *, action_token_strings=None, instructions=None):
    del instructions
    batch_size = 1
    for value in obs.values():
        if isinstance(value, torch.Tensor):
            batch_size = value.shape[0]
            break

    seq_len = 10
    inputs = {
        "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
    }

    if action_token_strings is not None:
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
        for index, action_str in enumerate(action_token_strings):
            n_action_tokens = action_str.count("<|action_")
            start = max(0, seq_len - n_action_tokens)
            for j in range(start, seq_len):
                token_id = self._require_action_token_vocabulary().token_min + (
                    j - start
                )
                inputs["input_ids"][index, j] = token_id
                labels[index, j] = token_id
        inputs["labels"] = labels

    return inputs


def mock_build_vlm_inputs_all_ignore(
    self, obs, *, action_token_strings=None, instructions=None
):
    return _mock_build_vlm_inputs_all_ignore_basic(
        self,
        obs,
        action_token_strings=action_token_strings,
        instructions=instructions,
    )


def patch_fast_qwen_dependencies(
    monkeypatch,
    *,
    tokenizer: DummyFASTTokenizer | None = None,
    vlm: DummyVLM | None = None,
    processor: DummyProcessor | None = None,
    use_mock_inputs: bool = True,
) -> tuple[DummyFASTTokenizer, DummyVLM, DummyProcessor]:
    dummy_tokenizer = DummyFASTTokenizer() if tokenizer is None else tokenizer
    dummy_vlm = DummyVLM(vocab_size=200, hidden_dim=16) if vlm is None else vlm
    dummy_processor = DummyProcessor() if processor is None else processor

    def mock_load_vlm(self, model_name, dtype_str):
        del model_name, dtype_str
        self.vlm = dummy_vlm
        self.processor = dummy_processor

    monkeypatch.setattr(
        "praxis.policies.bar.backbones.qwen.qwen_base.QwenPolicyBase._load_vlm",
        mock_load_vlm,
    )
    monkeypatch.setattr(
        "praxis.policies.bar.core.vlm.base.make_codec",
        lambda _cfg: dummy_tokenizer,
    )
    monkeypatch.setattr(
        "praxis.policies.bar.backbones.qwen.qwen_base.QwenPolicyBase._build_vlm_prefix_adapter",
        lambda self: BasicPrefixAdapter(self.vlm),
    )
    if use_mock_inputs:
        monkeypatch.setattr(
            "praxis.policies.bar.backbones.qwen.qwen_base.QwenPolicyBase._build_vlm_inputs",
            mock_build_vlm_inputs,
        )
    return dummy_tokenizer, dummy_vlm, dummy_processor


def make_policy(monkeypatch) -> FASTQwenPolicy:
    io = make_io()
    patch_fast_qwen_dependencies(monkeypatch)
    return FASTQwenPolicy.from_io(
        io,
        n_obs_steps=1,
        action_horizon=2,
        n_action_steps=1,
        action_tokenizer={"kind": "tokens", "name": "fast", "pretrained_path": None},
        freeze_action_tokenizer=True,
    )


def make_policy_with_real_vlm_inputs(
    monkeypatch,
    *,
    device: torch.device,
    processor: DummyProcessor,
) -> FASTQwenPolicy:
    io = make_io()
    dummy_tokenizer = DummyFASTTokenizer()
    dummy_vlm = DummyVLM(vocab_size=200, hidden_dim=16).to(device)
    patch_fast_qwen_dependencies(
        monkeypatch,
        tokenizer=dummy_tokenizer,
        vlm=dummy_vlm,
        processor=processor,
        use_mock_inputs=False,
    )
    policy = FASTQwenPolicy.from_io(
        io,
        n_obs_steps=1,
        action_horizon=2,
        n_action_steps=1,
        action_tokenizer={"kind": "tokens", "name": "fast", "pretrained_path": None},
        freeze_action_tokenizer=True,
    )
    policy.vlm = policy.vlm.to(device)
    return policy
