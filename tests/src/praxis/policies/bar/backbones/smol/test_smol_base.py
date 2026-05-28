"""Tests for shared Smol action-token policy base behavior."""

from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from praxis.policies.bar.core.vlm.decode_tokens import (
    extract_contiguous_action_token_ids,
)
from praxis.policies.bar.variants import FASTSmolPolicy
from praxis.policies.core.observations import extract_instructions
from praxis.policies.core.state_prefix import (
    DEFAULT_STATE_PREFIX_TOKEN,
    prepare_state_prefix,
)

from ._test_utils import (
    _DummyFASTTokenizer,
    _DummyProcessor,
    _DummyVLM,
    _make_io,
    _mock_build_vlm_inputs,
    _patch_smol,
    _stats_for_io,
)


def _extract_action_token_ids(policy, generated_ids, *, prompt_length: int = 0):
    vocab = policy._require_action_token_vocabulary()
    return extract_contiguous_action_token_ids(
        generated_ids,
        vocab=vocab,
        prompt_length=prompt_length,
    )


def test_fast_smol_action_token_setup(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    policy = FASTSmolPolicy.from_io(io)
    assert (
        policy._require_action_token_vocabulary().token_max
        == policy._require_action_token_vocabulary().token_min
        + policy.action_tokenizer.codebook_size
        - 1
    )


def test_fast_smol_action_token_range_uses_tokenizer_ids(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM(vocab_size=256)
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    policy = FASTSmolPolicy.from_io(io)

    vocab = processor.tokenizer.get_vocab()
    last_token = (
        f"<|action_{policy._require_action_token_vocabulary().num_tokens - 1}|>"
    )
    assert policy._require_action_token_vocabulary().token_min == vocab["<|action_0|>"]
    assert policy._require_action_token_vocabulary().token_max == vocab[last_token]
    assert policy.vlm.get_input_embeddings().weight.shape[0] >= (
        policy._require_action_token_vocabulary().token_max + 1
    )


def test_fast_smol_shared_vlm_base_adds_action_mask_trainable_row(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM(vocab_size=256)
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    policy = FASTSmolPolicy.from_io(io)

    vocab = processor.tokenizer.get_vocab()
    assert policy._action_mask_token_id == vocab["<|action_mask|>"]

    rows = policy._build_trainable_vocab_rows(
        vocab_size=policy.vlm.get_input_embeddings().weight.shape[0]
    )
    assert policy._action_mask_token_id in rows.tolist()


def test_fast_smol_action_tokenizer_receives_action_only_io(monkeypatch):
    io = _make_io(with_image=True)
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    policy = FASTSmolPolicy.from_io(io)
    assert policy.action_tokenizer.io is not None
    assert policy.action_tokenizer.io.action_feature.key == "action"
    assert policy.action_tokenizer.io.obs_features == []


def test_fast_smol_token_id_to_vlm_string_roundtrip(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    policy = FASTSmolPolicy.from_io(io)
    fast_ids = [0, 5, 7]
    vlm_string = policy._require_action_token_vocabulary().render_raw_ids(fast_ids)
    assert "<|action_0|>" in vlm_string
    assert "<|action_5|>" in vlm_string
    assert "<|action_7|>" in vlm_string
    vlm_ids = torch.tensor(
        [[policy._require_action_token_vocabulary().token_min + i for i in fast_ids]],
        dtype=torch.long,
    )
    decoded = _extract_action_token_ids(policy, vlm_ids)
    assert decoded[0] == fast_ids


def test_fast_smol_num_action_tokens_mismatch_raises(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    with pytest.raises(ValueError, match="num_action_tokens"):
        _ = FASTSmolPolicy.from_io(io, num_action_tokens=50)


def test_fast_smol_build_inputs_uses_custom_fast_image_preprocess(monkeypatch):
    io = _make_io(with_image=True)
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)

    policy = FASTSmolPolicy.from_io(io)
    vlm_inputs = policy._build_vlm_inputs(
        {
            "observation.state": torch.randn(1, 1, 4),
            "observation.image": torch.rand(1, 1, 3, 4, 4),
        },
        instructions=["do task"],
    )
    assert "pixel_values" in vlm_inputs
    assert vlm_inputs["pixel_values"].dtype == torch.float32


def test_fast_smol_build_inputs_uses_smolvla_style_processor_by_default(monkeypatch):
    io = _make_io(with_image=True)
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)

    policy = FASTSmolPolicy.from_io(io)
    vlm_inputs = policy._build_vlm_inputs(
        {
            "observation.state": torch.randn(1, 1, 4),
            "observation.image": torch.rand(1, 1, 3, 4, 4),
        },
        instructions=["do task"],
    )
    assert processor.last_kwargs is None
    assert processor.expand_text_calls == 0
    assert "pixel_values" in vlm_inputs
    assert vlm_inputs["pixel_values"].dtype == torch.float32


def test_fast_smol_load_vlm_preserves_hf_image_processor_defaults(monkeypatch):
    class _ImageProcessor:
        def __init__(self) -> None:
            self.do_resize = True
            self.do_image_splitting = True

    class _Processor:
        def __init__(self) -> None:
            self.tokenizer = type("_Tokenizer", (), {"padding_side": "right"})()
            self.image_processor = _ImageProcessor()

    processor = _Processor()
    vlm = object()
    captured: dict[str, object] = {}

    def _from_pretrained(*args, **kwargs):
        del args
        captured.update(kwargs)
        return vlm

    monkeypatch.setattr(
        AutoModelForImageTextToText,
        "from_pretrained",
        staticmethod(_from_pretrained),
    )
    monkeypatch.setattr(
        AutoProcessor,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: processor),
    )

    policy = object.__new__(FASTSmolPolicy)
    policy._load_vlm("dummy/model", "bfloat16")

    assert policy.vlm is vlm
    assert policy.processor is processor
    assert captured["low_cpu_mem_usage"] is True
    assert processor.tokenizer.padding_side == "left"
    assert processor.image_processor.do_resize is True
    assert processor.image_processor.do_image_splitting is True


def test_fast_smol_lora_state_dict_excludes_frozen_vlm_and_tokenizer(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)

    policy = FASTSmolPolicy.from_io(io)
    policy.config.use_lora = True
    for param in policy.vlm.parameters():
        param.requires_grad_(False)
    policy.vlm.lm_head.weight.requires_grad_(True)

    state_dict = policy.state_dict()
    assert "vlm.embed_tokens.weight" not in state_dict
    assert "vlm.lm_head.weight" in state_dict
    assert not any(key.startswith("action_tokenizer.") for key in state_dict)


def test_fast_smol_lora_strict_load_allows_missing_frozen_vlm_params(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor_source = _DummyProcessor()
    vlm_source = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor_source, vlm=vlm_source)
    source = FASTSmolPolicy.from_io(io)
    source.config.use_lora = True
    for param in source.vlm.parameters():
        param.requires_grad_(False)
    source.vlm.lm_head.weight.requires_grad_(True)
    filtered_state_dict = source.state_dict()

    tok_target = _DummyFASTTokenizer()
    processor_target = _DummyProcessor()
    vlm_target = _DummyVLM()
    _patch_smol(
        monkeypatch,
        tokenizer=tok_target,
        processor=processor_target,
        vlm=vlm_target,
    )
    target = FASTSmolPolicy.from_io(io)
    target.config.use_lora = True
    for param in target.vlm.parameters():
        param.requires_grad_(False)
    target.vlm.lm_head.weight.requires_grad_(True)
    _ = target.load_state_dict(filtered_state_dict, strict=True)


def test_fast_smol_lora_optim_groups_put_vocab_weights_on_zero_decay(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    policy = FASTSmolPolicy.from_io(io)
    policy.config.use_lora = True

    groups = policy.get_optim_params()
    zero_decay_groups = [g for g in groups if float(g.get("weight_decay", 1.0)) == 0.0]
    assert zero_decay_groups
    zero_decay_param_ids = {id(p) for g in zero_decay_groups for p in g["params"]}
    assert id(policy.vlm.embed_tokens.weight) in zero_decay_param_ids
    assert id(policy.vlm.lm_head.weight) in zero_decay_param_ids


def test_fast_smol_forward_all_ignore_labels_skips_batch(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    policy = FASTSmolPolicy.from_io(io)
    batch = {
        "observation.state": torch.randn(2, 1, 4),
        "action": torch.randn(2, 2, 2),
        "action_is_pad": torch.ones(2, 2, dtype=torch.bool),
    }
    loss, info = policy(batch)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    assert info.get("skipped_all_ignore_batch") == 1.0


def test_fast_smol_state_discretization(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    policy = FASTSmolPolicy.from_io(io)
    state_texts = prepare_state_prefix(
        {"observation.state": torch.tensor([[-1.0, 0.0, 1.0, 0.5]])},
        state_features=policy.io.state_features,
        encoding="discrete_text",
        num_bins=policy.config.state_prefix_num_bins,
        placeholder_token=DEFAULT_STATE_PREFIX_TOKEN,
    ).texts
    assert len(state_texts) == 1
    bins = [int(x) for x in state_texts[0].replace("State: ", "").split()]
    assert bins[0] == 0
    assert bins[1] == 128
    assert bins[2] == 255


def test_fast_smol_forward_returns_scalar_loss(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    monkeypatch.setattr(
        "praxis.policies.bar.backbones.smol.smol_base.SmolPolicyBase._build_vlm_inputs",
        _mock_build_vlm_inputs,
    )
    policy = FASTSmolPolicy.from_io(io)
    policy.set_normalization_stats(_stats_for_io(io))
    batch = {
        "observation.state": torch.randn(2, 1, 4),
        "action": torch.randn(2, 2, 2),
    }
    loss, info = policy(batch)
    assert loss.shape == ()
    assert isinstance(info, dict)


def test_fast_smol_extract_instructions_uses_language_key(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    policy = FASTSmolPolicy.from_io(io)

    assert extract_instructions({"task": ["do something"]}, language_key=None) is None
    policy.io.language_key = "task"
    assert extract_instructions(
        {"task": ["pick up the cup"]},
        language_key=policy.io.language_key,
    ) == ["pick up the cup"]
    assert (
        extract_instructions(
            {"other": ["x"]},
            language_key=policy.io.language_key,
        )
        is None
    )
    assert (
        extract_instructions(
            {"task": torch.zeros(2)},
            language_key=policy.io.language_key,
        )
        is None
    )


def test_fast_smol_predict_action_chunk_uses_language_instruction(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    policy = FASTSmolPolicy.from_io(io)
    policy.io.language_key = "task"
    captured: dict[str, object] = {}
    vlm._generate_action_ids = [
        policy._require_action_token_vocabulary().token_min + i for i in [1, 2, 3]
    ]

    def _spy_build_inputs(obs, *, action_token_strings=None, instructions=None):
        del action_token_strings
        captured["instructions"] = instructions
        bsz = obs["observation.state"].shape[0]
        return {
            "input_ids": torch.zeros(bsz, 10, dtype=torch.long),
            "attention_mask": torch.ones(bsz, 10, dtype=torch.long),
        }

    monkeypatch.setattr(policy, "_build_vlm_inputs", _spy_build_inputs)
    _ = policy.predict_action_chunk(
        {"observation.state": torch.randn(1, 1, 4), "task": ["open drawer"]}
    )
    assert captured["instructions"] == ["open drawer"]


def test_fast_smol_predict_action_queue_path_preserves_language_instruction(
    monkeypatch,
):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    policy = FASTSmolPolicy.from_io(io)
    policy.io.language_key = "task"
    captured: dict[str, object] = {}
    vlm._generate_action_ids = [
        policy._require_action_token_vocabulary().token_min + i for i in [1, 2, 3]
    ]

    def _spy_build_inputs(obs, *, action_token_strings=None, instructions=None):
        del action_token_strings
        captured["instructions"] = instructions
        bsz = obs["observation.state"].shape[0]
        return {
            "input_ids": torch.zeros(bsz, 10, dtype=torch.long),
            "attention_mask": torch.ones(bsz, 10, dtype=torch.long),
        }

    monkeypatch.setattr(policy, "_build_vlm_inputs", _spy_build_inputs)
    _ = policy.predict_action(
        {"observation.state": torch.randn(4), "task": "open drawer"}
    )
    assert captured["instructions"] == ["open drawer"]


def test_fast_smol_extract_empty_when_no_action_tokens(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    policy = FASTSmolPolicy.from_io(io)
    gen_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    decoded = _extract_action_token_ids(policy, gen_ids)
    assert decoded[0] == []


def test_fast_smol_contiguous_extraction_stops_at_gap(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    policy = FASTSmolPolicy.from_io(io)
    lo = policy._require_action_token_vocabulary().token_min
    gen_ids = torch.tensor([[0, lo, lo + 1, 0, lo + 2]], dtype=torch.long)
    decoded = _extract_action_token_ids(policy, gen_ids)
    assert decoded[0] == [0, 1]


def test_fast_smol_contiguous_extraction_full_span(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    policy = FASTSmolPolicy.from_io(io)
    lo = policy._require_action_token_vocabulary().token_min
    gen_ids = torch.tensor([[0, lo, lo + 1, lo + 2]], dtype=torch.long)
    decoded = _extract_action_token_ids(policy, gen_ids)
    assert decoded[0] == [0, 1, 2]


def test_fast_smol_extraction_ignores_prompt_prefix_with_prompt_length(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    policy = FASTSmolPolicy.from_io(io)
    lo = policy._require_action_token_vocabulary().token_min
    gen_ids = torch.tensor([[lo, lo + 1, 0, lo + 2, lo + 3]], dtype=torch.long)

    decoded = _extract_action_token_ids(policy, gen_ids, prompt_length=3)
    assert decoded[0] == [2, 3]


def test_fast_smol_instruction_length_mismatch_raises(monkeypatch):
    io = _make_io(with_image=True)
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    policy = FASTSmolPolicy.from_io(io)

    with pytest.raises(ValueError, match="Instructions batch length mismatch"):
        _ = policy._build_vlm_inputs(
            {
                "observation.state": torch.randn(2, 1, 4),
                "observation.image": torch.rand(2, 1, 3, 4, 4),
            },
            instructions=["one only"],
        )


def test_fast_smol_state_only_normalization_keeps_images_raw(monkeypatch):
    io = _make_io(with_image=True)
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    policy = FASTSmolPolicy.from_io(io)
    policy.set_normalization_stats(_stats_for_io(io))

    state = torch.full((1, 4), 5.0)
    image = torch.rand(1, 3, 2, 2)
    prepared = policy._prepare_chunk_observations(
        {"observation.state": state, "observation.image": image},
        normalize=True,
    )
    assert torch.allclose(prepared["observation.image"][:, 0], image, atol=1e-6)
    assert torch.allclose(
        prepared["observation.state"][:, 0], torch.zeros_like(state), atol=1e-6
    )


def test_fast_smol_initializes_action_tokenizer_stats_once(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    policy = FASTSmolPolicy.from_io(io)

    policy.set_normalization_stats(_stats_for_io(io))
    assert policy.action_tokenizer.stats_set_calls == 0
    policy.set_normalization_stats(_stats_for_io(io))
    assert policy.action_tokenizer.stats_set_calls == 0


def test_fast_smol_does_not_overwrite_existing_action_tokenizer_stats(monkeypatch):
    io = _make_io()
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    policy = FASTSmolPolicy.from_io(io)
    policy.action_tokenizer.normalize_action = object()
    policy.action_tokenizer.unnormalize_action = object()

    policy.set_normalization_stats(_stats_for_io(io))
    assert policy.action_tokenizer.stats_set_calls == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fast_smol_build_vlm_inputs_gpu_images_stay_on_device(monkeypatch):
    io = _make_io(with_image=True)
    tok = _DummyFASTTokenizer()
    processor = _DummyProcessor()
    vlm = _DummyVLM().to(torch.device("cuda"))
    _patch_smol(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)

    policy = FASTSmolPolicy.from_io(io)
    policy.vlm = policy.vlm.to(torch.device("cuda"))
    obs = {
        "observation.state": torch.randn(2, 1, 4, device=torch.device("cuda")),
        "observation.image": torch.rand(2, 1, 3, 16, 16, device=torch.device("cuda")),
    }
    out = policy._build_vlm_inputs(obs)
    assert out["pixel_values"].device.type == "cuda"
