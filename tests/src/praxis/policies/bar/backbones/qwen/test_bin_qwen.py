"""Tests for Bin Qwen policy specifics."""

from __future__ import annotations

import torch

from praxis.policies.bar.variants import BinQwenPolicy

from ._bounded_test_utils import (
    apply_monkeypatches,
    make_bounded_policy,
    make_io,
    stats_for_io,
)


class TestBinQwenPolicy:
    def test_bin_qwen_init_and_max_decode_tokens(self, monkeypatch):
        policy = make_bounded_policy(monkeypatch, BinQwenPolicy)
        assert policy.config.name == "bar_bin_qwen"
        assert policy.max_code_seq_len == 4
        assert policy.max_decode_tokens == 4

    def test_bin_qwen_supervise_eos_is_false(self, monkeypatch):
        policy = make_bounded_policy(monkeypatch, BinQwenPolicy)
        assert policy._supervise_eos is False

    def test_bin_qwen_decode_uses_action_only_sampling(self, monkeypatch):
        policy = make_bounded_policy(monkeypatch, BinQwenPolicy)
        io = make_io()
        policy.set_normalization_stats(stats_for_io(io))
        captured: dict[str, object] = {}

        def spy_sample(logits, *, temperature, top_k, allowed_token_ids=None):
            captured["temperature"] = temperature
            captured["top_k"] = top_k
            captured["allowed"] = (
                allowed_token_ids.detach().cpu().tolist()
                if allowed_token_ids is not None
                else None
            )
            token = int(allowed_token_ids[0])
            return torch.full(
                (logits.shape[0], 1), token, dtype=torch.long, device=logits.device
            )

        monkeypatch.setattr(
            "praxis.policies.bar.core.vlm.decoding.sample_next_token_ids",
            spy_sample,
        )
        _ = policy.predict_action_chunk({"observation.state": torch.zeros(1, 4)})
        assert captured["temperature"] is None
        assert captured["top_k"] is None
        allowed = set(captured["allowed"])
        assert 99 not in allowed
        assert policy._require_action_token_vocabulary().token_min in allowed
        assert policy._require_action_token_vocabulary().token_max in allowed

    def test_bin_qwen_does_not_require_eos_token_id(self, monkeypatch):
        io = make_io()
        _, dummy_processor, _ = apply_monkeypatches(monkeypatch)
        dummy_processor.tokenizer.eos_token_id = None

        policy = BinQwenPolicy.from_io(
            io,
            n_obs_steps=1,
            action_horizon=2,
            n_action_steps=1,
            freeze_action_tokenizer=True,
        )
        policy.set_normalization_stats(stats_for_io(io))
        _ = policy.predict_action_chunk({"observation.state": torch.zeros(1, 4)})
