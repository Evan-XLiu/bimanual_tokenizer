"""Tests for BOAT Pali fixed-budget policy behavior."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from praxis.policies.bar.variants import BOATPaliPolicy, BOATPaliPolicyConfig

from ._bounded_test_utils import _patch_bounded_vlm_inputs
from ._test_utils import (
    _DummyFixedTokenizer,
    _DummyProcessor,
    _DummyVLM,
    _make_io,
    _patch_pali,
    _stats_for_io,
)


class _DummyBOATTokenizer(_DummyFixedTokenizer):
    @property
    def variable_length_codes(self) -> bool:
        return False

    @property
    def block_schedule(self) -> tuple[int, ...]:
        return (1, 1, 2)


def _make_policy(
    monkeypatch: pytest.MonkeyPatch,
    *,
    patch_decode_inputs: bool = True,
    **config_kwargs,
) -> tuple[BOATPaliPolicy, _DummyVLM]:
    io = _make_io()
    tok = _DummyBOATTokenizer(fixed_len=4)
    processor = _DummyProcessor()
    vlm = _DummyVLM()
    _patch_pali(monkeypatch, tokenizer=tok, processor=processor, vlm=vlm)
    if patch_decode_inputs:
        _patch_bounded_vlm_inputs(monkeypatch)

    policy = BOATPaliPolicy.from_io(io, **config_kwargs)
    policy.set_normalization_stats(_stats_for_io(io))
    return policy, vlm


def test_boat_pali_forward_uses_fixed_budget_blockwise_runtime(monkeypatch):
    policy, _vlm = _make_policy(monkeypatch, patch_decode_inputs=False)
    batch = {
        "observation.state": torch.randn(1, 1, 4),
        "action": torch.randn(1, 2, 2),
    }

    loss, info = policy(batch)

    assert loss.shape == ()
    assert isinstance(info, dict)
    assert "debug/train/L_stop" not in info
    assert "adaptive" not in policy.config.action_tokenizer
    assert policy.config.decode_stop_mode == "fixed_budget"
    assert policy._supervise_eos is False
    assert not hasattr(policy, "stop_head")


def test_boat_pali_config_is_fixed_budget_by_default():
    cfg = BOATPaliPolicyConfig.from_dict({"name": "bar_boat_pali"})

    assert cfg.decode_stop_mode == "fixed_budget"
    assert cfg.decode_keep_k is None
    assert cfg.action_tokenizer == {
        "kind": "tokens",
        "name": "boat",
        "pretrained_path": None,
    }


def test_boat_pali_config_rejects_removed_shared_flags():
    with pytest.raises(ValueError, match="not configurable"):
        BOATPaliPolicyConfig.from_dict(
            {"name": "bar_boat_pali", "supervise_eos": False}
        )
    with pytest.raises(ValueError, match="block_schedule is not configurable"):
        BOATPaliPolicyConfig.from_dict(
            {"name": "bar_boat_pali", "block_schedule": [1, 1]}
        )
    with pytest.raises(ValueError, match="unknown_policy_key"):
        BOATPaliPolicyConfig.from_dict(
            {"name": "bar_boat_pali", "unknown_policy_key": True}
        )


def test_boat_pali_config_rejects_invalid_block_ce_weighting():
    with pytest.raises(ValueError, match="block_ce_weighting"):
        BOATPaliPolicyConfig.from_dict({"block_ce_weighting": "sample_mean"})


def test_boat_pali_decode_keep_k_is_supported(monkeypatch):
    policy, vlm = _make_policy(monkeypatch, decode_keep_k=2)
    vlm._generate_action_ids = [
        policy._require_action_token_vocabulary().token_min + i for i in range(4)
    ]

    actions, info = policy.predict_action_chunk(
        {"observation.state": torch.randn(1, 1, 4)}
    )

    assert actions.shape == (1, 2, 2)
    assert info["decode_keep_k"] == 2
    assert info["decode_steps"] == 2
    assert info["block_schedule"] == [1, 1, 2]
    assert info["token_regime"] == "fixed_length"
    assert not any(str(key).startswith("pred_stop") for key in info)


def test_boat_pali_decode_keep_k_calltime_override(monkeypatch):
    policy, vlm = _make_policy(monkeypatch, decode_keep_k=None)
    vlm._generate_action_ids = [
        policy._require_action_token_vocabulary().token_min + i for i in range(4)
    ]

    _actions, info = policy.predict_action_chunk(
        {"observation.state": torch.randn(1, 1, 4)},
        decode_keep_k=1,
    )

    assert info["decode_keep_k"] == 1
    assert info["decode_steps"] == 1


def test_boat_pali_decode_keep_k_validation(monkeypatch):
    with pytest.raises(ValueError, match="decode_keep_k must be >= 1"):
        _make_policy(monkeypatch, decode_keep_k=0)

    with pytest.raises(ValueError, match="decode_keep_k must be <= token_horizon"):
        _make_policy(monkeypatch, decode_keep_k=8)

    policy, _vlm = _make_policy(monkeypatch)
    with pytest.raises(TypeError, match="decode_keep_k must be an int or None"):
        policy.predict_action_chunk(
            {"observation.state": torch.randn(1, 1, 4)},
            decode_keep_k="2",
        )


def test_boat_pali_hydra_accepts_decode_keep_k_override():
    config_dir = Path("src/praxis/configs").resolve()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="train_policy",
            overrides=[
                "policy=bar_boat_pali",
                "policy.decode_keep_k=2",
            ],
        )

    assert "decode_stop_mode" not in cfg.policy
    assert cfg.policy.decode_keep_k == 2
