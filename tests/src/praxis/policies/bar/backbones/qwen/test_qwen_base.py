# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared Qwen action-token policy base behavior."""

from __future__ import annotations

import pytest
import torch

from praxis.core.types import FeatureSpec, FeatureType, NormMode, PolicyIO
from praxis.policies.bar.backbones.qwen.qwen_config import QwenPolicyConfig
from praxis.policies.bar.core.vlm.decode_tokens import (
    extract_contiguous_action_token_ids,
)
from praxis.policies.bar.core.vlm.decoding import build_vlm_prefill_inputs
from praxis.policies.bar.variants import FASTQwenPolicy
from praxis.policies.core.observations import extract_instructions
from praxis.policies.core.state_prefix import (
    DEFAULT_STATE_PREFIX_TOKEN,
    prepare_state_prefix,
)

from ._test_utils import (
    DummyFASTTokenizer,
    DummyProcessor,
    DummyVLM,
    make_io,
    make_policy,
    make_policy_with_real_vlm_inputs,
    patch_fast_qwen_dependencies,
    stats_for_io,
)


def _extract_action_token_ids(policy, generated_ids, *, prompt_length: int = 0):
    vocab = policy._require_action_token_vocabulary()
    return extract_contiguous_action_token_ids(
        generated_ids,
        vocab=vocab,
        prompt_length=prompt_length,
    )


class TestQwenPolicyBase:
    def test_qwen_default_model_name_targets_qwen3_vl_2b(self):
        assert QwenPolicyConfig().vlm_model_name == "Qwen/Qwen3-VL-2B-Instruct"

    def test_bar_rejects_continuous_state_prefix(self):
        with pytest.raises(ValueError, match="BAR does not support"):
            QwenPolicyConfig(state_prefix_encoding="continuous_projection")

    def test_load_vlm_uses_low_cpu_memory_loading(self, monkeypatch):
        captured: dict[str, object] = {}
        processor = type(
            "_Processor",
            (),
            {"tokenizer": type("_Tokenizer", (), {"padding_side": "right"})()},
        )()
        vlm = object()

        def _from_pretrained(*args, **kwargs):
            del args
            captured.update(kwargs)
            return vlm

        monkeypatch.setattr(
            "praxis.policies.backbones.qwen.vlm.Qwen3VLForConditionalGeneration.from_pretrained",
            staticmethod(_from_pretrained),
        )
        monkeypatch.setattr(
            "praxis.policies.backbones.qwen.vlm.AutoProcessor.from_pretrained",
            staticmethod(lambda *args, **kwargs: processor),
        )

        policy = object.__new__(FASTQwenPolicy)
        policy._load_vlm("dummy/model", "bfloat16")

        assert policy.vlm is vlm
        assert policy.processor is processor
        assert captured["low_cpu_mem_usage"] is True
        assert processor.tokenizer.padding_side == "left"

    def test_action_token_setup(self, monkeypatch):
        policy = make_policy(monkeypatch)
        action_vocab = policy._require_action_token_vocabulary()
        assert (
            action_vocab.token_max
            == action_vocab.token_min + policy.action_tokenizer.codebook_size - 1
        )

    def test_action_token_range_uses_tokenizer_ids(self, monkeypatch):
        policy = make_policy(monkeypatch)
        vocab = policy.processor.tokenizer.get_vocab()
        last_token = (
            f"<|action_{policy._require_action_token_vocabulary().num_tokens - 1}|>"
        )
        assert (
            policy._require_action_token_vocabulary().token_min == vocab["<|action_0|>"]
        )
        assert policy._require_action_token_vocabulary().token_max == vocab[last_token]
        assert policy.vlm.get_input_embeddings().weight.shape[0] >= (
            policy._require_action_token_vocabulary().token_max + 1
        )

    def test_shared_vlm_base_adds_action_mask_trainable_row(self, monkeypatch):
        policy = make_policy(monkeypatch)
        vocab = policy.processor.tokenizer.get_vocab()
        assert policy._action_mask_token_id == vocab["<|action_mask|>"]

        rows = policy._build_trainable_vocab_rows(
            vocab_size=policy.vlm.get_input_embeddings().weight.shape[0]
        )
        assert policy._action_mask_token_id in rows.tolist()

    def test_action_tokenizer_receives_action_only_io(self, monkeypatch):
        policy = make_policy(monkeypatch)
        assert policy.action_tokenizer.io is not None
        assert policy.action_tokenizer.io.action_feature.key == "action"
        assert policy.action_tokenizer.io.obs_features == []

    def test_token_id_to_vlm_string_roundtrip(self, monkeypatch):
        policy = make_policy(monkeypatch)
        fast_ids = [0, 5, 7]
        vlm_string = policy._require_action_token_vocabulary().render_raw_ids(fast_ids)
        assert "<|action_0|>" in vlm_string
        assert "<|action_5|>" in vlm_string
        assert "<|action_7|>" in vlm_string

        vlm_ids = torch.tensor(
            [
                [
                    policy._require_action_token_vocabulary().token_min + token
                    for token in fast_ids
                ]
            ],
            dtype=torch.long,
        )
        decoded = _extract_action_token_ids(policy, vlm_ids)
        assert decoded[0] == fast_ids

    def test_num_action_tokens_mismatch_raises(self, monkeypatch):
        io = make_io()
        patch_fast_qwen_dependencies(
            monkeypatch,
            tokenizer=DummyFASTTokenizer(),
            vlm=DummyVLM(vocab_size=200, hidden_dim=16),
            processor=DummyProcessor(),
        )

        with pytest.raises(ValueError, match="num_action_tokens"):
            FASTQwenPolicy.from_io(
                io,
                action_tokenizer={
                    "kind": "tokens",
                    "name": "fast",
                    "pretrained_path": None,
                },
                freeze_action_tokenizer=True,
                num_action_tokens=50,
            )

    def test_state_discretization(self, monkeypatch):
        policy = make_policy(monkeypatch)
        state_texts = prepare_state_prefix(
            {"observation.state": torch.tensor([[-1.0, 0.0, 1.0, 0.5]])},
            state_features=policy.io.state_features,
            encoding="discrete_text",
            num_bins=policy.config.state_prefix_num_bins,
            placeholder_token=DEFAULT_STATE_PREFIX_TOKEN,
        ).texts
        assert len(state_texts) == 1
        text = state_texts[0]
        assert text.startswith("State: ")
        bins = [int(x) for x in text.replace("State: ", "").split()]
        assert bins[0] == 0
        assert bins[1] == 128
        assert bins[2] == 255

    def test_forward_returns_scalar_loss(self, monkeypatch):
        policy = make_policy(monkeypatch)
        io = make_io()
        policy.set_normalization_stats(stats_for_io(io))

        batch = {
            "observation.state": torch.randn(2, 1, 4),
            "action": torch.randn(2, 2, 2),
        }
        loss, info = policy(batch)
        assert loss.shape == ()
        assert isinstance(info, dict)

    def test_forward_all_ignore_labels_skips_batch(self, monkeypatch):
        policy = make_policy(monkeypatch)
        io = make_io()
        policy.set_normalization_stats(stats_for_io(io))

        batch = {
            "observation.state": torch.randn(2, 1, 4),
            "action": torch.randn(2, 2, 2),
            "action_is_pad": torch.ones(2, 2, dtype=torch.bool),
        }
        loss, info = policy(batch)
        assert torch.isfinite(loss)
        assert loss.item() == 0.0
        assert info.get("skipped_all_ignore_batch") == 1.0

    def test_predict_action_chunk_uses_language_instruction(self, monkeypatch):
        policy = make_policy(monkeypatch)
        policy.io.language_key = "task"
        policy.vlm._generate_action_ids = [
            policy._require_action_token_vocabulary().token_min + i for i in [1, 2, 3]
        ]
        captured: dict[str, object] = {}

        def spy_build_inputs(obs, *, action_token_strings=None, instructions=None):
            del action_token_strings
            captured["instructions"] = instructions
            batch_size = obs["observation.state"].shape[0]
            return {
                "input_ids": torch.zeros(batch_size, 10, dtype=torch.long),
                "attention_mask": torch.ones(batch_size, 10, dtype=torch.long),
            }

        monkeypatch.setattr(policy, "_build_vlm_inputs", spy_build_inputs)
        _ = policy.predict_action_chunk(
            {"observation.state": torch.randn(1, 1, 4), "task": ["open the drawer"]}
        )
        assert captured["instructions"] == ["open the drawer"]

    def test_predict_action_queue_path_preserves_language_instruction(
        self, monkeypatch
    ):
        policy = make_policy(monkeypatch)
        policy.io.language_key = "task"
        policy.vlm._generate_action_ids = [
            policy._require_action_token_vocabulary().token_min + i for i in [1, 2, 3]
        ]
        captured: dict[str, object] = {}

        def spy_build_inputs(obs, *, action_token_strings=None, instructions=None):
            del action_token_strings
            captured["instructions"] = instructions
            batch_size = obs["observation.state"].shape[0]
            return {
                "input_ids": torch.zeros(batch_size, 10, dtype=torch.long),
                "attention_mask": torch.ones(batch_size, 10, dtype=torch.long),
            }

        monkeypatch.setattr(policy, "_build_vlm_inputs", spy_build_inputs)
        _ = policy.predict_action(
            {"observation.state": torch.randn(4), "task": "open the drawer"}
        )
        assert captured["instructions"] == ["open the drawer"]

    def test_frozen_tokenizer_excluded_from_state_dict(self, monkeypatch):
        policy = make_policy(monkeypatch)
        state_dict = policy.state_dict()
        tokenizer_keys = [
            key for key in state_dict if key.startswith("action_tokenizer.")
        ]
        assert len(tokenizer_keys) == 0

    def test_lora_state_dict_excludes_frozen_vlm_params(self, monkeypatch):
        policy = make_policy(monkeypatch)
        policy.config.use_lora = True
        for param in policy.vlm.parameters():
            param.requires_grad_(False)
        policy.vlm.lm_head.weight.requires_grad_(True)
        policy.register_buffer(
            "norm_stat_probe", torch.tensor([1.0], dtype=torch.float32)
        )

        state_dict = policy.state_dict()
        assert "vlm.embed_tokens.weight" not in state_dict
        assert "vlm.lm_head.weight" in state_dict
        assert "norm_stat_probe" in state_dict

    def test_lora_state_dict_also_excludes_frozen_action_tokenizer(self, monkeypatch):
        policy = make_policy(monkeypatch)
        policy.config.use_lora = True
        for param in policy.vlm.parameters():
            param.requires_grad_(False)
        policy.vlm.lm_head.weight.requires_grad_(True)

        state_dict = policy.state_dict()
        assert "vlm.embed_tokens.weight" not in state_dict
        assert "vlm.lm_head.weight" in state_dict
        assert not any(key.startswith("action_tokenizer.") for key in state_dict)

    def test_lora_strict_load_allows_missing_frozen_vlm_params(self, monkeypatch):
        source = make_policy(monkeypatch)
        source.config.use_lora = True
        for param in source.vlm.parameters():
            param.requires_grad_(False)
        source.vlm.lm_head.weight.requires_grad_(True)
        filtered_state_dict = source.state_dict()

        target = make_policy(monkeypatch)
        target.config.use_lora = True
        for param in target.vlm.parameters():
            param.requires_grad_(False)
        target.vlm.lm_head.weight.requires_grad_(True)

        _ = target.load_state_dict(filtered_state_dict, strict=True)

    def test_lora_vocab_grad_mask_keeps_only_action_rows(self, monkeypatch):
        policy = make_policy(monkeypatch)
        policy.config.use_lora = True
        policy._install_vocab_row_gradient_masks()

        weight = policy.vlm.embed_tokens.weight
        weight.grad = None
        weight.sum().backward()
        grad = weight.grad
        assert grad is not None

        action_min = int(policy._require_action_token_vocabulary().token_min)
        action_max = int(policy._require_action_token_vocabulary().token_max)
        allowed = torch.zeros(grad.shape[0], dtype=torch.bool)
        allowed[action_min : action_max + 1] = True
        action_mask_token_id = getattr(policy, "_action_mask_token_id", None)
        if action_mask_token_id is not None:
            allowed[int(action_mask_token_id)] = True
        if policy._supervise_eos and policy._eos_token_id is not None:
            allowed[int(policy._eos_token_id)] = True

        assert grad[allowed].abs().sum().item() > 0.0
        assert grad[~allowed].abs().sum().item() == 0.0

    def test_lora_optim_groups_put_vocab_weights_on_zero_decay(self, monkeypatch):
        policy = make_policy(monkeypatch)
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

    def test_extract_empty_when_no_action_tokens(self, monkeypatch):
        policy = make_policy(monkeypatch)
        generated_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        decoded = _extract_action_token_ids(policy, generated_ids)
        assert decoded[0] == []

    def test_extract_instructions_uses_language_key(self, monkeypatch):
        policy = make_policy(monkeypatch)
        assert (
            extract_instructions({"task": ["do something"]}, language_key=None) is None
        )

        policy.io.language_key = "task"
        result = extract_instructions(
            {"task": ["pick up the cup"]},
            language_key=policy.io.language_key,
        )
        assert result == ["pick up the cup"]
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

    def test_state_only_normalization_keeps_images_raw(self, monkeypatch):
        policy = make_policy(monkeypatch)
        io = PolicyIO(
            obs_features=[
                FeatureSpec(
                    key="observation.state",
                    type=FeatureType.STATE,
                    shape=(4,),
                    norm_mode=NormMode.MIN_MAX,
                ),
                FeatureSpec(
                    key="observation.image",
                    type=FeatureType.IMAGE,
                    shape=(3, 2, 2),
                    norm_mode=NormMode.MEAN_STD,
                ),
            ],
            action_feature=FeatureSpec(
                key="action",
                type=FeatureType.ACTION,
                shape=(2,),
                norm_mode=NormMode.MIN_MAX,
            ),
            n_obs_steps=1,
            n_action_steps=1,
            fps=10,
        )
        policy.set_io(io)
        policy.set_normalization_stats(
            {
                "observation.state": {
                    "min": torch.zeros(4),
                    "max": torch.full((4,), 10.0),
                    "mean": torch.full((4,), 5.0),
                    "std": torch.ones(4),
                },
                "observation.image": {
                    "mean": torch.full((3, 2, 2), 0.5),
                    "std": torch.full((3, 2, 2), 0.25),
                },
            }
        )

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

    def test_policy_initializes_action_tokenizer_stats_once(self, monkeypatch):
        policy = make_policy(monkeypatch)
        io = make_io()
        assert policy.action_tokenizer.stats_set_calls == 0

        policy.set_normalization_stats(stats_for_io(io))
        assert policy.action_tokenizer.stats_set_calls == 0

        policy.set_normalization_stats(stats_for_io(io))
        assert policy.action_tokenizer.stats_set_calls == 0

    def test_policy_does_not_overwrite_existing_action_tokenizer_stats(
        self, monkeypatch
    ):
        policy = make_policy(monkeypatch)
        io = make_io()
        policy.action_tokenizer.normalize_action = object()
        policy.action_tokenizer.unnormalize_action = object()

        policy.set_normalization_stats(stats_for_io(io))
        assert policy.action_tokenizer.stats_set_calls == 0

    def test_contiguous_extraction_stops_at_gap(self, monkeypatch):
        policy = make_policy(monkeypatch)
        low = policy._require_action_token_vocabulary().token_min
        generated_ids = torch.tensor([[0, low, low + 1, 0, low + 2]], dtype=torch.long)
        decoded = _extract_action_token_ids(policy, generated_ids)
        assert decoded[0] == [0, 1]

    def test_contiguous_extraction_full_span(self, monkeypatch):
        policy = make_policy(monkeypatch)
        low = policy._require_action_token_vocabulary().token_min
        generated_ids = torch.tensor([[0, low, low + 1, low + 2]], dtype=torch.long)
        decoded = _extract_action_token_ids(policy, generated_ids)
        assert decoded[0] == [0, 1, 2]

    def test_extraction_ignores_prompt_prefix_with_prompt_length(self, monkeypatch):
        policy = make_policy(monkeypatch)
        low = policy._require_action_token_vocabulary().token_min
        generated_ids = torch.tensor(
            [[low, low + 1, 0, low + 2, low + 3]], dtype=torch.long
        )
        decoded = _extract_action_token_ids(policy, generated_ids, prompt_length=3)
        assert decoded[0] == [2, 3]

    def test_prefill_inputs_forward_image_grid_thw(self, monkeypatch):
        policy = make_policy(monkeypatch)
        captured: dict[str, object] = {}

        def spy_prepare_inputs_for_generation(**kwargs):
            captured.update(kwargs)
            return {
                "input_ids": kwargs["input_ids"],
                "attention_mask": kwargs["attention_mask"],
                "cache_position": kwargs["cache_position"],
                "pixel_values": kwargs["pixel_values"],
                "image_grid_thw": kwargs["image_grid_thw"],
            }

        monkeypatch.setattr(
            policy.vlm,
            "prepare_inputs_for_generation",
            spy_prepare_inputs_for_generation,
        )
        prompt_input_ids = torch.zeros((1, 3), dtype=torch.long)
        pred_block_vlm_ids = torch.full(
            (1, 2),
            fill_value=int(policy._require_action_token_vocabulary().token_min),
            dtype=torch.long,
        )
        image_grid_thw = torch.zeros((1, 3), dtype=torch.long)
        pixel_values = torch.zeros((1, 4), dtype=torch.float32)

        _ = build_vlm_prefill_inputs(
            vlm=policy.vlm,
            base_vlm_inputs={
                "input_ids": prompt_input_ids,
                "attention_mask": torch.ones((1, 5), dtype=torch.long),
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            },
            prompt_input_ids=prompt_input_ids,
            running_attention_mask_2d=torch.ones((1, 5), dtype=torch.long),
            pred_block_vlm_ids=pred_block_vlm_ids,
            block_schedule=(2,),
            mask_dtype=torch.float32,
            prefix_attention=policy._require_prefix_attention(),
            extra_prepare_keys=policy._PREFILL_EXTRA_KEYS,
        )

        assert captured["pixel_values"] is pixel_values
        assert captured["image_grid_thw"] is image_grid_thw

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_build_vlm_inputs_gpu_images_stay_on_device(self, monkeypatch):
        device = torch.device("cuda")
        processor = DummyProcessor()
        policy = make_policy_with_real_vlm_inputs(
            monkeypatch,
            device=device,
            processor=processor,
        )
        io = PolicyIO(
            obs_features=[
                FeatureSpec(
                    key="observation.state",
                    type=FeatureType.STATE,
                    shape=(4,),
                    norm_mode=NormMode.MIN_MAX,
                ),
                FeatureSpec(
                    key="observation.image",
                    type=FeatureType.IMAGE,
                    shape=(3, 16, 16),
                    norm_mode=NormMode.MEAN_STD,
                ),
            ],
            action_feature=FeatureSpec(
                key="action",
                type=FeatureType.ACTION,
                shape=(2,),
                norm_mode=NormMode.MIN_MAX,
            ),
            n_obs_steps=1,
            n_action_steps=1,
            fps=10,
        )
        policy.set_io(io)

        batch_size = 2
        obs = {
            "observation.state": torch.randn(batch_size, 1, 4, device=device),
            "observation.image": torch.rand(batch_size, 1, 3, 16, 16, device=device),
        }
        out = policy._build_vlm_inputs(obs)

        assert processor.tokenizer.last_text is not None
        assert out["pixel_values"].device.type == "cuda"
        assert out["image_grid_thw"].device.type == "cuda"
