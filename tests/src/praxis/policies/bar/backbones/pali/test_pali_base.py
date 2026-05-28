"""Tests for shared Pali action-token policy base behavior."""

from __future__ import annotations

import pytest
import torch

from praxis.policies.bar.core.vlm.decode_tokens import (
    extract_contiguous_action_token_ids,
)
from praxis.policies.bar.variants import FASTPaliPolicy

from ._test_utils import (
    _DummyBatchEncoding,
    _DummyFASTTokenizer,
    _DummyProcessor,
    _DummyVLM,
    _make_io,
    _patch_pali,
    _stats_for_io,
)


def _extract_action_token_ids(policy, generated_ids, *, prompt_length: int = 0):
    vocab = policy._require_action_token_vocabulary()
    return extract_contiguous_action_token_ids(
        generated_ids,
        vocab=vocab,
        prompt_length=prompt_length,
    )


def test_fast_pali_action_token_range_uses_tokenizer_ids(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM(vocab_size=256)
    _patch_pali(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)

    policy = FASTPaliPolicy.from_io(io)
    vocab = processor.tokenizer.get_vocab()
    last_token = (
        f"<|action_{policy._require_action_token_vocabulary().num_tokens - 1}|>"
    )
    assert policy._require_action_token_vocabulary().token_min == vocab["<|action_0|>"]
    assert policy._require_action_token_vocabulary().token_max == vocab[last_token]
    assert policy.vlm.get_input_embeddings().weight.shape[0] >= (
        policy._require_action_token_vocabulary().token_max + 1
    )


def test_fast_pali_load_vlm_uses_low_cpu_memory_loading(monkeypatch):
    captured: dict[str, object] = {}
    processor = type(
        "_Processor",
        (),
        {
            "tokenizer": type("_Tokenizer", (), {"padding_side": "right"})(),
            "image_token": "<image>",
        },
    )()
    vlm = object()

    def _from_pretrained(*args, **kwargs):
        del args
        captured.update(kwargs)
        return vlm

    monkeypatch.setattr(
        "praxis.policies.backbones.pali.vlm.PaliGemmaForConditionalGeneration.from_pretrained",
        staticmethod(_from_pretrained),
    )
    monkeypatch.setattr(
        "praxis.policies.backbones.pali.vlm.AutoProcessor.from_pretrained",
        staticmethod(lambda *args, **kwargs: processor),
    )

    policy = object.__new__(FASTPaliPolicy)
    policy._load_vlm("dummy/model", "bfloat16")

    assert policy.vlm is vlm
    assert policy.processor is processor
    assert captured["low_cpu_mem_usage"] is True
    assert processor.tokenizer.padding_side == "left"


def test_fast_pali_shared_vlm_base_adds_action_mask_trainable_row(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM(vocab_size=256)
    _patch_pali(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)

    policy = FASTPaliPolicy.from_io(io)
    vocab = processor.tokenizer.get_vocab()
    assert policy._action_mask_token_id == vocab["<|action_mask|>"]

    rows = policy._build_trainable_vocab_rows(
        vocab_size=policy.vlm.get_input_embeddings().weight.shape[0]
    )
    assert policy._action_mask_token_id in rows.tolist()


def test_fast_pali_forward_all_ignore_labels_skips_batch(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_pali(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)

    policy = FASTPaliPolicy.from_io(io)
    batch = {
        "observation.state": torch.randn(2, 1, 4),
        "action": torch.randn(2, 2, 2),
        "action_is_pad": torch.ones(2, 2, dtype=torch.bool),
    }
    loss, info = policy(batch)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    assert info.get("skipped_all_ignore_batch") == 1.0


def test_fast_pali_predict_action_chunk_shape(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_pali(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)

    policy = FASTPaliPolicy.from_io(io)
    policy.set_normalization_stats(_stats_for_io(io))
    vlm._generate_action_ids = [
        policy._require_action_token_vocabulary().token_min + i for i in [1, 2, 3]
    ]

    actions, _ = policy.predict_action_chunk(
        {"observation.state": torch.randn(1, 1, 4)}
    )
    assert actions.shape == (1, 2, 2)


def test_fast_pali_decode_rejects_unexpected_kwargs(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_pali(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)

    policy = FASTPaliPolicy.from_io(io)
    with pytest.raises(TypeError, match="Unsupported VLM decode kwargs: unexpected"):
        policy.predict_action_chunk(
            {"observation.state": torch.zeros(1, 1, 4)},
            unexpected=True,
        )


def test_fast_pali_skips_tokenizer_stats_init_without_action_stats(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_pali(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)

    policy = FASTPaliPolicy.from_io(io)
    obs_only_stats = {
        "observation.state": {
            "min": torch.full((4,), -1.0),
            "max": torch.full((4,), 1.0),
        }
    }
    policy.set_normalization_stats(obs_only_stats)
    assert tok.stats_set_calls == 0


def test_fast_pali_extraction_ignores_prompt_prefix_with_prompt_length(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_pali(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)

    policy = FASTPaliPolicy.from_io(io)
    low = policy._require_action_token_vocabulary().token_min
    gen_ids = torch.tensor([[low, low + 1, 0, low + 2, low + 3]], dtype=torch.long)
    decoded = _extract_action_token_ids(policy, gen_ids, prompt_length=3)
    assert decoded[0] == [2, 3]


def test_fast_pali_build_vlm_inputs_prefixes_image_tokens(monkeypatch):
    io = _make_io(with_image=True)
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM(vocab_size=400)
    _patch_pali(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)

    captured: dict[str, list[str]] = {}

    def spy_run_processor(
        *, processor, text, suffix=None, image_batch, padding=True, return_tensors="pt"
    ):
        del processor, suffix, image_batch, padding, return_tensors
        captured["text"] = list(text)
        return _DummyBatchEncoding(
            {
                "input_ids": torch.zeros(1, 5, dtype=torch.long),
                "attention_mask": torch.ones(1, 5, dtype=torch.long),
                "pixel_values": torch.randn(1, 3, 4, 4),
            }
        )

    monkeypatch.setattr(
        "praxis.policies.bar.backbones.pali.pali_base.run_paligemma_processor",
        spy_run_processor,
    )

    policy = FASTPaliPolicy.from_io(io)
    _ = policy._build_vlm_inputs(
        {
            "observation.state": torch.randn(1, 1, 4),
            "observation.image": torch.rand(1, 1, 3, 4, 4),
        },
        instructions=["do task"],
    )
    assert captured["text"][0].startswith("<image>\nTask:")
    assert captured["text"][0].endswith("Action: ")


def test_fast_pali_training_path_passes_action_tokens_via_suffix(monkeypatch):
    io = _make_io(with_image=True)
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM(vocab_size=400)
    _patch_pali(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    policy = FASTPaliPolicy.from_io(io)
    action_text = "<|action_1|><|action_2|>"
    captured: dict[str, list[str]] = {}

    def spy_run_processor(
        *, processor, text, suffix=None, image_batch, padding=True, return_tensors="pt"
    ):
        del processor, image_batch, padding, return_tensors
        captured["text"] = list(text)
        captured["suffix"] = [] if suffix is None else list(suffix)
        return _DummyBatchEncoding(
            {
                "input_ids": torch.tensor(
                    [
                        [
                            10,
                            20,
                            policy._require_action_token_vocabulary().token_min,
                            policy._require_action_token_vocabulary().token_min + 1,
                            99,
                        ]
                    ],
                    dtype=torch.long,
                ),
                "attention_mask": torch.ones(1, 5, dtype=torch.long),
                "pixel_values": torch.randn(1, 3, 4, 4),
            }
        )

    monkeypatch.setattr(
        "praxis.policies.bar.backbones.pali.pali_base.run_paligemma_processor",
        spy_run_processor,
    )

    _ = policy._build_vlm_inputs(
        {
            "observation.state": torch.randn(1, 1, 4),
            "observation.image": torch.rand(1, 1, 3, 4, 4),
        },
        action_token_strings=[action_text],
        instructions=["do task"],
    )
    assert captured["suffix"] == [action_text]
    assert captured["text"][0].endswith("Action: ")


def test_fast_pali_lora_state_dict_excludes_frozen_vlm_and_tokenizer(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_pali(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)

    policy = FASTPaliPolicy.from_io(io)
    policy.config.use_lora = True
    for param in policy.vlm.parameters():
        param.requires_grad_(False)
    policy.vlm.lm_head.weight.requires_grad_(True)

    state_dict = policy.state_dict()
    assert "vlm.embed_tokens.weight" not in state_dict
    assert "vlm.lm_head.weight" in state_dict
    assert not any(key.startswith("action_tokenizer.") for key in state_dict)


def test_fast_pali_lora_optim_groups_put_vocab_weights_on_zero_decay(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_pali(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)

    policy = FASTPaliPolicy.from_io(io)
    policy.config.use_lora = True
    groups = policy.get_optim_params()

    zero_decay_groups = [g for g in groups if float(g.get("weight_decay", 1.0)) == 0.0]
    assert zero_decay_groups
    zero_decay_param_ids = {id(p) for g in zero_decay_groups for p in g["params"]}
    assert id(policy.vlm.embed_tokens.weight) in zero_decay_param_ids
    assert id(policy.vlm.lm_head.weight) in zero_decay_param_ids
