"""Tests for ACodec Qwen policy specifics."""

from __future__ import annotations

import torch

from praxis.policies.bar.variants import ACodecQwenPolicy

from ._bounded_test_utils import (
    DummyFixedTokenizer,
    apply_monkeypatches,
    make_bounded_policy,
    make_io,
    stats_for_io,
)
from ._test_utils import DummyProcessor, DummyVLM


class TestACodecQwenPolicy:
    def test_acodec_qwen_init_and_max_decode_tokens(self, monkeypatch):
        policy = make_bounded_policy(monkeypatch, ACodecQwenPolicy)
        assert policy.config.name == "bar_acodec_qwen"
        assert policy.max_code_seq_len == 4
        assert policy.max_decode_tokens == 4

    def test_acodec_qwen_supervise_eos_is_false(self, monkeypatch):
        policy = make_bounded_policy(monkeypatch, ACodecQwenPolicy)
        assert policy._supervise_eos is False

    def test_acodec_qwen_decodes_single_parallel_block(self, monkeypatch):
        class SingleBlockTokenizer(DummyFixedTokenizer):
            @property
            def block_schedule(self) -> tuple[int, ...]:
                return (4,)

        io = make_io()
        tokenizer = SingleBlockTokenizer(fixed_len=4)
        vlm = DummyVLM(vocab_size=200, hidden_dim=16)
        apply_monkeypatches(
            monkeypatch,
            tokenizer=tokenizer,
            processor=DummyProcessor(),
            vlm=vlm,
        )
        policy = ACodecQwenPolicy.from_io(
            io,
            n_obs_steps=1,
            action_horizon=2,
            n_action_steps=1,
            freeze_action_tokenizer=True,
        )
        policy.set_normalization_stats(stats_for_io(io))
        policy.vlm._generate_action_ids = [
            policy._require_action_token_vocabulary().token_min + i for i in range(4)
        ]

        _actions, info = policy.predict_action_chunk(
            {"observation.state": torch.randn(1, 1, 4)}
        )

        assert info["block_schedule"] == [4]
        assert info["decode_steps"] == 1
        assert policy.vlm._last_prefill_attention_rank == 4
        assert policy.vlm._last_decode_attention_rank is None
        assert policy.vlm._last_prefill_logits_to_keep == 4
