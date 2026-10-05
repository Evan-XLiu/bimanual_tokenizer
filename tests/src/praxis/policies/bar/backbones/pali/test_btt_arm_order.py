"""Check supervision, conditional generation and native codec arm layout."""

import pytest
import torch

from praxis.policies.bar.variants import BTTPaliPolicy, BTTPaliPolicyConfig
from praxis.policies.core.vlm.training import prepare_action_token_ce
from ._test_utils import _DummyFixedTokenizer, _DummyVLM, _make_io, _patch_pali, _stats_for_io
from .test_btt_oneblock_pali import FullBlockProcessor


class PairedTokenizer(_DummyFixedTokenizer):
    block_schedule = (8, 8)

    def tokenize(self, actions, **kwargs):
        return torch.arange(16).expand(actions.shape[0], -1)

    def detokenize(self, tokens, **kwargs):
        self.decoded_tokens = tokens.clone()
        return super().detokenize(tokens, **kwargs)


class BlockVLM(_DummyVLM):
    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)
        if result.logits is not None:
            selected = result.logits[:, -1].argmax(-1)
            result.logits.fill_(-1e9)
            result.logits.scatter_(2, selected[:, None, None].expand(-1, result.logits.shape[1], 1), 0)
        return result


def build_policy(monkeypatch, order):
    tokenizer = PairedTokenizer(fixed_len=16)
    vlm = BlockVLM()
    _patch_pali(monkeypatch, tokenizer=tokenizer, processor=FullBlockProcessor(), vlm=vlm)
    policy = BTTPaliPolicy.from_io(_make_io(), btt_arm_order=order)
    policy.set_normalization_stats(_stats_for_io(_make_io()))
    return policy, tokenizer, vlm


@pytest.mark.parametrize("order", ["left_then_right", "right_then_left"])
def test_training_targets_and_previous_block_inputs(monkeypatch, order):
    policy, _, _ = build_policy(monkeypatch, order)
    batch = {"observation.state": torch.randn(1, 1, 4), "action": torch.zeros(1, 2, 2)}
    prepared = prepare_action_token_ce(policy, batch)
    vocab = policy._require_action_token_vocabulary()
    expected = list(range(16)) if order == "left_then_right" else list(range(8, 16)) + list(range(8))
    positions = prepared.supervision.prediction_position_mask
    assert prepared.block_schedule == (8, 8)
    assert prepared.labels[positions].tolist() == [(vocab.token_min + i) for i in expected]
    inputs = prepared.vlm_inputs["input_ids"][positions].tolist()
    assert inputs[:8] == [vocab.mask_token_id] * 8
    assert inputs[8:] == [(vocab.token_min + i) for i in expected[:8]]


@pytest.mark.parametrize("order", ["left_then_right", "right_then_left"])
def test_two_generation_steps_restore_native_codec_layout(monkeypatch, order):
    policy, tokenizer, vlm = build_policy(monkeypatch, order)
    vocab = policy._require_action_token_vocabulary()
    vlm._generate_action_ids = [(vocab.token_min + 3), (vocab.token_min + 4)]
    policy.predict_action_chunk({"observation.state": torch.zeros(1, 1, 4)})
    assert len(vlm.forward_calls) == 2
    assert vlm.prepare_calls[0]["input_ids"][:, -8:].eq(vocab.mask_token_id).all()
    assert vlm.prepare_calls[1]["input_ids"][:, -8:].eq((vocab.token_min + 3)).all()
    expected = [3] * 8 + [4] * 8 if order == "left_then_right" else [4] * 8 + [3] * 8
    assert tokenizer.decoded_tokens.tolist() == [expected]


def test_config_roundtrip_and_validation():
    config = BTTPaliPolicyConfig(btt_arm_order="right_then_left")
    assert BTTPaliPolicyConfig.from_dict(config.to_dict()).btt_arm_order == "right_then_left"
    assert BTTPaliPolicyConfig.from_dict({}).btt_arm_order == "left_then_right"
    with pytest.raises(ValueError, match="btt_arm_order"):
        BTTPaliPolicyConfig(btt_arm_order="bad")
    with pytest.raises(ValueError, match="standard btt"):
        BTTPaliPolicyConfig(btt_arm_order="right_then_left", action_tokenizer={"name": "btt_oneblock"})


def test_incomplete_decode_rejected(monkeypatch):
    policy, _, _ = build_policy(monkeypatch, "right_then_left")
    with pytest.raises(ValueError, match="full 16"):
        policy._decode_generated_action_tokens([[1] * 8], device=torch.device("cpu"), horizon=8)
