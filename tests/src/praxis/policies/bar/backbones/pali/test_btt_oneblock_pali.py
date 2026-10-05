"""Exercise the real BAR train-input and generation loop with a dummy VLM."""

import re

import torch

from praxis.policies.bar.variants import BTTOneBlockPaliPolicy
from praxis.policies.core.vlm.training import (
    prepare_action_token_ce,
    prepare_action_token_ce_from_token_targets,
)

from ._bounded_test_utils import _patch_bounded_vlm_inputs
from ._test_utils import (
    _DummyFixedTokenizer,
    _DummyProcessor,
    _DummyVLM,
    _make_io,
    _patch_pali,
    _stats_for_io,
)


class SingleBlockTokenizer(_DummyFixedTokenizer):
    @property
    def block_schedule(self):
        return (16,)


class FullBlockProcessor(_DummyProcessor):
    """The shared dummy truncates to ten positions; retain the full block."""

    def __call__(self, *args, **kwargs):
        result = super().__call__(*args, **kwargs)
        suffix = kwargs.get("suffix")
        if suffix is not None:
            suffixes = suffix if isinstance(suffix, list) else [suffix]
            rows = [
                [1, 2, 3]
                + [
                    self.tokenizer.convert_tokens_to_ids(t)
                    for t in re.findall(r"<\|[^|]+\|>", s)
                ]
                + [self.tokenizer.eos_token_id]
                for s in suffixes
            ]
            result["input_ids"] = torch.tensor(rows, dtype=torch.long)
            result["labels"] = result["input_ids"].clone()
            result["attention_mask"] = torch.ones_like(result["input_ids"])
        return result


def build_policy(monkeypatch, *, patch_decode_inputs=True):
    io = _make_io()
    tokenizer = SingleBlockTokenizer(fixed_len=16)
    vlm = _DummyVLM()
    _patch_pali(
        monkeypatch, tokenizer=tokenizer, processor=FullBlockProcessor(), vlm=vlm
    )
    if patch_decode_inputs:
        _patch_bounded_vlm_inputs(monkeypatch)
    policy = BTTOneBlockPaliPolicy.from_io(io)
    policy.set_normalization_stats(_stats_for_io(io))
    return policy, vlm


def test_one_forward_produces_all_16_tokens(monkeypatch):
    policy, vlm = build_policy(monkeypatch)
    vlm._generate_action_ids = [policy._require_action_token_vocabulary().token_min]
    _actions, info = policy.predict_action_chunk(
        {"observation.state": torch.randn(1, 1, 4)}
    )
    assert info["block_schedule"] == [16]
    assert info["decode_steps"] == 1
    assert len(vlm.forward_calls) == 1
    assert vlm.forward_calls[0]["logits_to_keep"] == 16
    mask = vlm.forward_calls[0]["attention_mask"]
    assert mask[0, 0, -16:, -16:].eq(0).all()
    assert (
        vlm.prepare_calls[0]["input_ids"][:, -16:]
        .eq(policy._require_action_token_vocabulary().mask_token_id)
        .all()
    )


def test_training_uses_masks_and_same_position_labels(monkeypatch):
    policy, _vlm = build_policy(monkeypatch, patch_decode_inputs=False)
    batch = {"observation.state": torch.randn(1, 1, 4), "action": torch.randn(1, 2, 2)}
    prepared = prepare_action_token_ce(policy, batch)
    vocab = policy._require_action_token_vocabulary()
    positions = prepared.supervision.prediction_position_mask
    assert prepared.block_schedule == (16,)
    assert prepared.vlm_inputs["input_ids"][positions].eq(vocab.mask_token_id).all()
    assert prepared.labels[positions].numel() == 16
    assert prepared.labels[positions].ge(vocab.token_min).all()
    assert prepared.labels[positions].le(vocab.token_max).all()


def test_changing_targets_does_not_change_single_block_inputs(monkeypatch):
    policy, _vlm = build_policy(monkeypatch, patch_decode_inputs=False)
    batch = {"observation.state": torch.randn(1, 1, 4)}
    first = prepare_action_token_ce_from_token_targets(
        policy, batch, token_ids_batch=[list(range(16))], token_lengths=[16]
    )
    second = prepare_action_token_ce_from_token_targets(
        policy, batch, token_ids_batch=[list(reversed(range(16)))], token_lengths=[16]
    )
    assert torch.equal(first.vlm_inputs["input_ids"], second.vlm_inputs["input_ids"])
    assert torch.equal(
        first.vlm_inputs["attention_mask"], second.vlm_inputs["attention_mask"]
    )
    positions = first.supervision.prediction_position_mask
    assert not torch.equal(first.labels[positions], second.labels[positions])
