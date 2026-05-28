"""Tests for policy registry-based factory."""

from __future__ import annotations

import json

import pytest
import torch
from omegaconf import OmegaConf

import praxis.policies.factory as policy_factory
from praxis.codecs.factory import current_codec_path_base
from praxis.core.types import FeatureSpec, FeatureType, NormMode, PolicyIO
from praxis.policies.bar.core.vlm.config import VlmActionTokenPolicyConfig
from praxis.policies.catalog import supported_policy_names
from praxis.policies.core import BasePolicy


class DummyPolicy(BasePolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def from_io(cls, io: PolicyIO, **config_overrides):
        policy = cls()
        policy.set_io(io)
        return policy

    def forward(self, batch: dict[str, torch.Tensor]):
        loss = batch["action"].float().mean() * 0
        return loss, {"loss": float(loss.item())}

    def predict_action(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = 1
        state = obs.get("observation.state")
        if isinstance(state, torch.Tensor) and state.dim() >= 2:
            batch_size = int(state.shape[0])
        device = state.device if isinstance(state, torch.Tensor) else "cpu"
        return torch.zeros((batch_size, 2), dtype=torch.float32, device=device)


class DummyPretrainedPolicy(BasePolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loaded_from: str | None = None

    @classmethod
    def from_pretrained(cls, path_or_hub_id: str, **kwargs):
        del kwargs
        inst = cls()
        inst.loaded_from = path_or_hub_id
        return inst

    @classmethod
    def from_io(cls, io: PolicyIO, **config_overrides):
        del config_overrides
        inst = cls()
        inst.set_io(io)
        return inst

    def forward(self, batch: dict[str, torch.Tensor]):
        loss = batch["action"].float().mean() * 0
        return loss, {"loss": float(loss.item())}

    def predict_action(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        del obs
        return torch.zeros((1, 2), dtype=torch.float32)


class DummyLoadedPolicy(DummyPretrainedPolicy):
    @classmethod
    def from_pretrained(cls, path_or_hub_id: str, **kwargs):
        del kwargs
        inst = cls()
        inst.loaded_from = path_or_hub_id
        inst.set_io(_dummy_io())
        return inst


class _DummyConfig:
    """Minimal config object that supports attribute get/set."""

    PRETRAINED_RUNTIME_OVERRIDE_KEYS = frozenset({"bar", "foo"})

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class DummyLoadedPolicyWithConfig(DummyPretrainedPolicy):
    """Pretrained policy that exposes a mutable config and tracks set_io calls."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = _DummyConfig(
            foo="original",
            bar=0,
            action_tokenizer={
                "kind": "tokens",
                "name": "fast",
                "pretrained_path": "/checkpoint/tok",
            },
        )
        self.set_io_calls: int = 0

    @classmethod
    def from_pretrained(cls, path_or_hub_id: str, **kwargs):
        del kwargs
        inst = cls()
        inst.loaded_from = path_or_hub_id
        inst.set_io(_dummy_io())
        return inst

    def set_io(self, io) -> None:
        super().set_io(io)
        self.set_io_calls = getattr(self, "set_io_calls", 0) + 1


class DummyLoadedPolicyWithVlmConfig(DummyPretrainedPolicy):
    """Pretrained policy exposing the real VLM config fields."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = VlmActionTokenPolicyConfig.from_dict({})
        self.set_io_calls: int = 0

    @classmethod
    def from_pretrained(cls, path_or_hub_id: str, **kwargs):
        del kwargs
        inst = cls()
        inst.loaded_from = path_or_hub_id
        inst.set_io(_dummy_io())
        return inst

    def set_io(self, io) -> None:
        super().set_io(io)
        self.set_io_calls = getattr(self, "set_io_calls", 0) + 1


class DummyLoadedPolicyMismatchedActionDim(DummyPretrainedPolicy):
    @classmethod
    def from_pretrained(cls, path_or_hub_id: str, **kwargs):
        del kwargs
        inst = cls()
        inst.loaded_from = path_or_hub_id
        io = _dummy_io()
        io.action_feature = FeatureSpec(
            key="action",
            type=FeatureType.ACTION,
            shape=(3,),
            norm_mode=NormMode.MIN_MAX,
        )
        inst.set_io(io)
        return inst


class DummyLoadedPolicyCapturingTokenizerBase(DummyPretrainedPolicy):
    @classmethod
    def from_pretrained(cls, path_or_hub_id: str, **kwargs):
        del kwargs
        inst = cls()
        base = current_codec_path_base()
        inst.loaded_from = None if base is None else str(base)
        inst.set_io(_dummy_io())
        return inst


def _dummy_io() -> PolicyIO:
    return PolicyIO(
        obs_features=[
            FeatureSpec(
                key="observation.state",
                type=FeatureType.STATE,
                shape=(2,),
                norm_mode=NormMode.MIN_MAX,
            )
        ],
        action_feature=FeatureSpec(
            key="action",
            type=FeatureType.ACTION,
            shape=(2,),
            norm_mode=NormMode.MIN_MAX,
        ),
        n_obs_steps=1,
        n_action_steps=1,
    )


class TestPolicyFactoryRegistry:
    def test_default_registry_contains_catalog_policies(self):
        policies = set(policy_factory.available_policies())

        assert policies == set(supported_policy_names())
        assert not any(name.startswith(("diffusion_", "flow_")) for name in policies)

    def test_register_policy_and_instantiate(self):
        original = dict(policy_factory._POLICY_REGISTRY)
        try:
            policy_factory.register_policy("dummy", f"{__name__}:DummyPolicy")
            cfg = OmegaConf.create({"name": "dummy", "pretrained_path": None})
            policy = policy_factory.make_policy(cfg, _dummy_io())
            assert isinstance(policy, DummyPolicy)
        finally:
            policy_factory._POLICY_REGISTRY.clear()
            policy_factory._POLICY_REGISTRY.update(original)

    def test_load_pretrained_policy_requires_io(self, monkeypatch):
        monkeypatch.setattr(
            policy_factory, "get_policy_class", lambda _name: DummyPretrainedPolicy
        )
        try:
            policy_factory._load_pretrained_policy(
                "dummy_pretrained", "/tmp/nonexistent"
            )
        except RuntimeError as exc:
            assert "did not load PolicyIO" in str(exc)
        else:
            raise AssertionError(
                "Expected RuntimeError when pretrained policy has no IO loaded."
            )

    def test_load_pretrained_policy_success(self, monkeypatch):
        monkeypatch.setattr(
            policy_factory, "get_policy_class", lambda _name: DummyLoadedPolicy
        )
        policy = policy_factory._load_pretrained_policy("dummy_loaded", "org/repo")
        assert isinstance(policy, DummyLoadedPolicy)
        assert policy.io is not None

    def test_load_pretrained_policy_sets_tokenizer_base_from_checkpoint_repo_root(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(
            policy_factory,
            "get_policy_class",
            lambda _name: DummyLoadedPolicyCapturingTokenizerBase,
        )
        (tmp_path / ".git").mkdir()
        ckpt = tmp_path / "outputs" / "train_policy" / "run" / "pretrained_model"
        ckpt.mkdir(parents=True, exist_ok=True)
        policy = policy_factory._load_pretrained_policy("dummy_loaded", str(ckpt))
        assert isinstance(policy, DummyLoadedPolicyCapturingTokenizerBase)
        assert policy.loaded_from == str(tmp_path.resolve())

    def test_make_policy_pretrained_rejects_action_dim_mismatch(self, monkeypatch):
        monkeypatch.setattr(
            policy_factory,
            "get_policy_class",
            lambda _name: DummyLoadedPolicyMismatchedActionDim,
        )
        cfg = OmegaConf.create(
            {"name": "dummy_loaded", "pretrained_path": "/tmp/nonexistent"}
        )
        with pytest.raises(ValueError, match="Pretrained policy action_dim mismatch"):
            _ = policy_factory.make_policy(cfg, _dummy_io())

    def test_make_policy_from_scratch_requires_pretrained_action_tokenizer(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            policy_factory, "get_policy_class", lambda _name: DummyPolicy
        )
        cfg = OmegaConf.create(
            {
                "name": "bar_fast_pali",
                "pretrained_path": None,
                "action_tokenizer": {
                    "name": "fast",
                    "kind": "tokens",
                    "pretrained_path": None,
                    "bpe_tokenizer_path": "physical-intelligence/fast",
                },
            }
        )
        with pytest.raises(
            ValueError, match="requires action_tokenizer.pretrained_path"
        ):
            _ = policy_factory.make_policy(cfg, _dummy_io())

    def test_make_policy_from_scratch_accepts_pretrained_action_tokenizer(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(
            policy_factory, "get_policy_class", lambda _name: DummyPolicy
        )
        tok_dir = tmp_path / "tok_ok"
        tok_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "stats": {
                    "action": {
                        "min": torch.zeros(2),
                        "max": torch.ones(2),
                    }
                },
                "modes": {"action": "min_max"},
            },
            tok_dir / "norm_stats.pt",
        )
        (tok_dir / "policy_io.json").write_text(
            json.dumps(_dummy_io().to_dict()), encoding="utf-8"
        )
        cfg = OmegaConf.create(
            {
                "name": "bar_fast_pali",
                "pretrained_path": None,
                "action_tokenizer": {
                    "name": "fast",
                    "kind": "tokens",
                    "pretrained_path": str(tok_dir),
                },
            }
        )
        policy = policy_factory.make_policy(cfg, _dummy_io())
        assert isinstance(policy, DummyPolicy)

    def test_make_policy_from_scratch_rejects_missing_tokenizer_norm_stats(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            policy_factory, "get_policy_class", lambda _name: DummyPolicy
        )
        cfg = OmegaConf.create(
            {
                "name": "bar_fast_pali",
                "pretrained_path": None,
                "action_tokenizer": {
                    "name": "fast",
                    "kind": "tokens",
                    "pretrained_path": "/tmp/nonexistent_tokenizer_ckpt",
                },
            }
        )
        with pytest.raises(ValueError, match="missing norm_stats\\.pt"):
            _ = policy_factory.make_policy(cfg, _dummy_io())

    def test_make_policy_from_scratch_rejects_tokenizer_without_action_stats(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(
            policy_factory, "get_policy_class", lambda _name: DummyPolicy
        )
        tok_dir = tmp_path / "tok_bad"
        tok_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "stats": {
                    "observation.state": {
                        "min": torch.zeros(2),
                        "max": torch.ones(2),
                    }
                },
                "modes": {"observation.state": "min_max"},
            },
            tok_dir / "norm_stats.pt",
        )
        (tok_dir / "policy_io.json").write_text(
            json.dumps(_dummy_io().to_dict()), encoding="utf-8"
        )
        cfg = OmegaConf.create(
            {
                "name": "bar_fast_pali",
                "pretrained_path": None,
                "action_tokenizer": {
                    "name": "fast",
                    "kind": "tokens",
                    "pretrained_path": str(tok_dir),
                },
            }
        )
        with pytest.raises(ValueError, match="missing action stats"):
            _ = policy_factory.make_policy(cfg, _dummy_io())

    def test_make_policy_pretrained_applies_cfg_overrides(self, monkeypatch):
        """CLI config overrides must be applied when loading from pretrained_path."""
        monkeypatch.setattr(
            policy_factory,
            "get_policy_class",
            lambda _name: DummyLoadedPolicyWithConfig,
        )
        monkeypatch.setattr(
            policy_factory,
            "_get_explicit_policy_override_keys",
            lambda: {"foo", "bar"},
        )
        cfg = OmegaConf.create(
            {
                "name": "dummy_loaded",
                "pretrained_path": "/tmp/nonexistent",
                "foo": "overridden",
                "bar": 42,
            }
        )
        policy = policy_factory.make_policy(cfg)
        assert isinstance(policy, DummyLoadedPolicyWithConfig)
        # Config fields must be overridden from CLI values
        assert policy.config.foo == "overridden"
        assert policy.config.bar == 42
        # set_io must have been called again (once in from_pretrained, once for override setup)
        assert policy.set_io_calls == 2

    def test_make_policy_pretrained_allows_runtime_n_action_steps_override(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            policy_factory,
            "get_policy_class",
            lambda _name: DummyLoadedPolicyWithVlmConfig,
        )
        monkeypatch.setattr(
            policy_factory,
            "_get_explicit_policy_override_keys",
            lambda: {"n_action_steps"},
        )
        cfg = OmegaConf.create(
            {
                "name": "dummy_loaded",
                "pretrained_path": "/tmp/nonexistent",
                "n_action_steps": 4,
            }
        )

        policy = policy_factory.make_policy(cfg)

        assert isinstance(policy, DummyLoadedPolicyWithVlmConfig)
        assert policy.config.n_action_steps == 4
        assert policy.set_io_calls == 2

    def test_make_policy_pretrained_rejects_vlm_checkpoint_owned_override(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            policy_factory,
            "get_policy_class",
            lambda _name: DummyLoadedPolicyWithVlmConfig,
        )
        monkeypatch.setattr(
            policy_factory,
            "_get_explicit_policy_override_keys",
            lambda: {"action_embed_init"},
        )
        cfg = OmegaConf.create(
            {
                "name": "dummy_loaded",
                "pretrained_path": "/tmp/nonexistent",
                "action_embed_init": "avg",
            }
        )

        with pytest.raises(ValueError, match=r"policy\.action_embed_init"):
            policy_factory.make_policy(cfg)

    def test_make_policy_pretrained_without_explicit_overrides_keeps_checkpoint_config(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            policy_factory,
            "get_policy_class",
            lambda _name: DummyLoadedPolicyWithConfig,
        )
        monkeypatch.setattr(
            policy_factory, "_get_explicit_policy_override_keys", lambda: set()
        )
        cfg = OmegaConf.create(
            {
                "name": "dummy_loaded",
                "pretrained_path": "/tmp/nonexistent",
                "foo": "default_cfg_value",
                "bar": 999,
            }
        )
        policy = policy_factory.make_policy(cfg)
        assert isinstance(policy, DummyLoadedPolicyWithConfig)
        # No explicit policy.* override => preserve checkpoint config values.
        assert policy.config.foo == "original"
        assert policy.config.bar == 0
        # set_io should not be re-run when no overrides were applied.
        assert policy.set_io_calls == 1

    def test_make_policy_pretrained_rejects_runtime_rebuild_override(self, monkeypatch):
        monkeypatch.setattr(
            policy_factory,
            "get_policy_class",
            lambda _name: DummyLoadedPolicyWithConfig,
        )
        monkeypatch.setattr(
            policy_factory,
            "_get_explicit_policy_override_keys",
            lambda: {"action_tokenizer"},
        )
        cfg = OmegaConf.create(
            {
                "name": "dummy_loaded",
                "pretrained_path": "/tmp/nonexistent",
                "action_tokenizer": {
                    "name": "fast",
                    "kind": "tokens",
                    "pretrained_path": "/other/tok",
                },
            }
        )

        with pytest.raises(ValueError, match=r"policy\.action_tokenizer"):
            policy_factory.make_policy(cfg)

    def test_make_policy_pretrained_rejects_unknown_explicit_override(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            policy_factory,
            "get_policy_class",
            lambda _name: DummyLoadedPolicyWithVlmConfig,
        )
        monkeypatch.setattr(
            policy_factory,
            "_get_explicit_policy_override_keys",
            lambda: {"decode_unknown_temperature"},
        )
        cfg = OmegaConf.create(
            {
                "name": "dummy_loaded",
                "pretrained_path": "/tmp/nonexistent",
                "decode_unknown_temperature": 0.7,
            }
        )

        with pytest.raises(ValueError, match=r"policy\.decode_unknown_temperature"):
            policy_factory.make_policy(cfg)

    def test_make_policy_pretrained_validates_explicit_overrides(self, monkeypatch):
        monkeypatch.setattr(
            policy_factory,
            "get_policy_class",
            lambda _name: DummyLoadedPolicyWithVlmConfig,
        )
        monkeypatch.setattr(
            policy_factory,
            "_get_explicit_policy_override_keys",
            lambda: {"block_ce_weighting"},
        )
        cfg = OmegaConf.create(
            {
                "name": "dummy_loaded",
                "pretrained_path": "/tmp/nonexistent",
                "block_ce_weighting": "sample_mean",
            }
        )

        with pytest.raises(ValueError, match="block_ce_weighting"):
            policy_factory.make_policy(cfg)

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("action_embed_init", "avg"),
            ("action_mask_token_template", "<|other_mask|>"),
            ("decode_stop_mode", "eos"),
        ],
    )
    def test_make_policy_pretrained_rejects_vlm_init_time_overrides(
        self, monkeypatch, key, value
    ):
        monkeypatch.setattr(
            policy_factory,
            "get_policy_class",
            lambda _name: DummyLoadedPolicyWithVlmConfig,
        )
        monkeypatch.setattr(
            policy_factory,
            "_get_explicit_policy_override_keys",
            lambda: {key},
        )
        cfg = OmegaConf.create(
            {
                "name": "dummy_loaded",
                "pretrained_path": "/tmp/nonexistent",
                key: value,
            }
        )

        with pytest.raises(ValueError, match=rf"policy\.{key}"):
            policy_factory.make_policy(cfg)
