"""Tests for codec factory registry and construction."""

from __future__ import annotations

import importlib

import pytest
import torch
from omegaconf import OmegaConf

from praxis.codecs.factory import (
    available_codecs,
    codec_path_base,
    current_codec_path_base,
    make_codec,
    register_codec,
)
from praxis.codecs.tokens.acodec.config import ActionCodecTokenizerConfig
from praxis.codecs.tokens.bin.config import BinTokenizerConfig
from praxis.codecs.tokens.boat.config import BOATTokenizerConfig
from praxis.codecs.tokens.core import BaseTokenizer
from praxis.codecs.tokens.fast.config import FASTTokenizerConfig
from praxis.codecs.tokens.oat.config import OATTokenizerConfig
from praxis.codecs.tokens.quest.config import QueSTTokenizerConfig


class DummyTokenizer(BaseTokenizer):
    def __init__(self, config: dict | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = config or {}
        self.weight = torch.nn.Parameter(torch.zeros(1))
        if self.config.get("extra_weight", False):
            self.extra = torch.nn.Parameter(torch.ones(1))
        init_weight = float(self.config.get("init_weight", 0.0))
        with torch.no_grad():
            self.weight.fill_(init_weight)

    @property
    def codebook_size(self) -> int:
        return 8

    def tokenize(self, actions: torch.Tensor) -> torch.Tensor:
        batch_size, horizon, _ = actions.shape
        return torch.zeros(
            (batch_size, horizon), dtype=torch.long, device=actions.device
        )

    def detokenize(self, tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        batch_size, horizon = tokens.shape
        return torch.zeros(
            (batch_size, horizon, 2), dtype=torch.float32, device=tokens.device
        )


class RuntimeStateTokenizer(DummyTokenizer):
    def __init__(self, config: dict | None = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self.runtime_state = self.config.get("runtime_state")

    def load_pretrained_runtime_state(self, source: BaseTokenizer) -> None:
        self.runtime_state = getattr(source, "runtime_state", None)


class TestCodecFactory:
    def test_available_codecs_contains_builtin_types(self):
        assert "tokens/acodec" in available_codecs()
        assert "tokens/boat" in available_codecs()
        assert "tokens/bin" in available_codecs()
        assert "tokens/oat" in available_codecs()
        assert "tokens/quest" in available_codecs()
        assert "latents/acodec" in available_codecs()
        assert "latents/oat" in available_codecs()
        assert "latents/quest" in available_codecs()
        assert "oat" in available_codecs(kind="tokens")
        assert "oat" in available_codecs(kind="latents")

    def test_register_codec_and_make(self):
        factory_mod = importlib.import_module("praxis.codecs.factory")
        register_codec("dummy", f"{__name__}:DummyTokenizer", kind="tokens")
        try:
            tok = make_codec(
                OmegaConf.create({"kind": "tokens", "name": "dummy", "foo": 123})
            )
            assert isinstance(tok, DummyTokenizer)
            assert tok.config.get("foo") == 123
        finally:
            factory_mod._CODEC_REGISTRY.pop(("tokens", "dummy"), None)

    def test_make_codec_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown codec type"):
            make_codec({"kind": "tokens", "name": "does_not_exist"})

    def test_make_codec_requires_kind(self):
        with pytest.raises(ValueError, match="non-empty `kind`"):
            make_codec({"name": "oat"})

    def test_make_codec_pretrained_path_uses_from_pretrained(self, monkeypatch):
        factory_mod = importlib.import_module("praxis.codecs.factory")
        register_codec("dummy", f"{__name__}:DummyTokenizer", kind="tokens")
        try:
            called: dict[str, object] = {}

            @classmethod
            def _fake_from_pretrained(cls, path):
                called["path"] = path
                return cls()

            monkeypatch.setattr(
                DummyTokenizer, "from_pretrained", _fake_from_pretrained
            )
            tok = make_codec(
                {"kind": "tokens", "name": "dummy", "pretrained_path": "/tmp/model"}
            )
            assert isinstance(tok, DummyTokenizer)
            assert called["path"] == "/tmp/model"
        finally:
            factory_mod._CODEC_REGISTRY.pop(("tokens", "dummy"), None)

    def test_make_codec_pretrained_path_resolves_relative_against_context_base(
        self, monkeypatch, tmp_path
    ):
        factory_mod = importlib.import_module("praxis.codecs.factory")
        register_codec("dummy", f"{__name__}:DummyTokenizer", kind="tokens")
        try:
            called: dict[str, object] = {}
            target = tmp_path / "outputs" / "tok" / "pretrained_model"
            target.mkdir(parents=True, exist_ok=True)

            @classmethod
            def _fake_from_pretrained(cls, path):
                called["path"] = path
                return cls()

            monkeypatch.setattr(
                DummyTokenizer, "from_pretrained", _fake_from_pretrained
            )
            assert current_codec_path_base() is None
            with codec_path_base(tmp_path):
                tok = make_codec(
                    {
                        "kind": "tokens",
                        "name": "dummy",
                        "pretrained_path": "outputs/tok/pretrained_model",
                    }
                )
            assert isinstance(tok, DummyTokenizer)
            assert called["path"] == str(target.resolve())
        finally:
            factory_mod._CODEC_REGISTRY.pop(("tokens", "dummy"), None)

    def test_make_codec_pretrained_with_runtime_overrides_transplants_weights(
        self, monkeypatch
    ):
        factory_mod = importlib.import_module("praxis.codecs.factory")
        register_codec("dummy", f"{__name__}:DummyTokenizer", kind="tokens")
        try:

            @classmethod
            def _fake_from_pretrained(cls, path):
                return cls(config={"foo": 1, "init_weight": 3.0})

            monkeypatch.setattr(
                DummyTokenizer, "from_pretrained", _fake_from_pretrained
            )
            tok = make_codec(
                {
                    "name": "dummy",
                    "kind": "tokens",
                    "pretrained_path": "/tmp/model",
                    "foo": 2,
                    "init_weight": 0.0,
                }
            )
            assert isinstance(tok, DummyTokenizer)
            # Runtime config override must win.
            assert tok.config.get("foo") == 2
            # Pretrained weights must be transplanted.
            assert float(tok.weight.item()) == pytest.approx(3.0)
        finally:
            factory_mod._CODEC_REGISTRY.pop(("tokens", "dummy"), None)

    def test_make_codec_pretrained_override_rejects_missing_weights(self, monkeypatch):
        factory_mod = importlib.import_module("praxis.codecs.factory")
        register_codec("dummy", f"{__name__}:DummyTokenizer", kind="tokens")
        try:

            @classmethod
            def _fake_from_pretrained(cls, path):
                return cls(config={"extra_weight": False})

            monkeypatch.setattr(
                DummyTokenizer, "from_pretrained", _fake_from_pretrained
            )
            with pytest.raises(RuntimeError, match="Missing pretrained keys"):
                make_codec(
                    {
                        "name": "dummy",
                        "kind": "tokens",
                        "pretrained_path": "/tmp/model",
                        "extra_weight": True,
                    }
                )
        finally:
            factory_mod._CODEC_REGISTRY.pop(("tokens", "dummy"), None)

    def test_make_codec_pretrained_override_copies_runtime_state(self, monkeypatch):
        factory_mod = importlib.import_module("praxis.codecs.factory")
        register_codec(
            "runtime_state", f"{__name__}:RuntimeStateTokenizer", kind="tokens"
        )
        marker = object()
        try:

            @classmethod
            def _fake_from_pretrained(cls, path):
                return cls(config={"foo": 1, "runtime_state": marker})

            monkeypatch.setattr(
                RuntimeStateTokenizer, "from_pretrained", _fake_from_pretrained
            )
            tok = make_codec(
                {
                    "name": "runtime_state",
                    "kind": "tokens",
                    "pretrained_path": "/tmp/model",
                    "foo": 2,
                }
            )

            assert isinstance(tok, RuntimeStateTokenizer)
            assert tok.runtime_state is marker
        finally:
            factory_mod._CODEC_REGISTRY.pop(("tokens", "runtime_state"), None)

    def test_make_fast_codec_pretrained_path_missing_raises(self, monkeypatch):
        factory_mod = importlib.import_module("praxis.codecs.factory")
        original_fast = factory_mod._CODEC_REGISTRY[("tokens", "fast")]
        factory_mod._CODEC_REGISTRY[("tokens", "fast")] = f"{__name__}:DummyTokenizer"
        try:

            @classmethod
            def _fake_from_pretrained(cls, path):
                raise FileNotFoundError(path)

            monkeypatch.setattr(
                DummyTokenizer, "from_pretrained", _fake_from_pretrained
            )
            with pytest.raises(FileNotFoundError):
                make_codec(
                    {
                        "kind": "tokens",
                        "name": "fast",
                        "pretrained_path": "physical-intelligence/fast",
                    }
                )
        finally:
            factory_mod._CODEC_REGISTRY[("tokens", "fast")] = original_fast

    def test_make_non_fast_pretrained_missing_still_raises(self, monkeypatch):
        factory_mod = importlib.import_module("praxis.codecs.factory")
        register_codec("dummy", f"{__name__}:DummyTokenizer", kind="tokens")
        try:

            @classmethod
            def _fake_from_pretrained(cls, path):
                raise FileNotFoundError(path)

            monkeypatch.setattr(
                DummyTokenizer, "from_pretrained", _fake_from_pretrained
            )
            with pytest.raises(FileNotFoundError):
                make_codec(
                    {
                        "kind": "tokens",
                        "name": "dummy",
                        "pretrained_path": "/tmp/missing",
                    }
                )
        finally:
            factory_mod._CODEC_REGISTRY.pop(("tokens", "dummy"), None)

    def test_make_oat_tokenizer_scaffold(self):
        tok = make_codec(
            {"kind": "tokens", "name": "oat", "action_dim": 8, "max_fit_chunks": 100}
        )
        assert tok.__class__.__name__ == "OATTokenizer"
        assert tok.codebook_size == 1920

    def test_make_boat_tokenizer_scaffold(self):
        tok = make_codec({"kind": "tokens", "name": "boat", "action_dim": 8})
        assert tok.__class__.__name__ == "BOATTokenizer"
        assert tok.codebook_size == 1920
        assert tok.block_schedule == (1, 1, 2, 4, 8)

    def test_make_acodec_tokenizer_scaffold(self):
        tok = make_codec({"kind": "tokens", "name": "acodec", "action_dim": 8})
        assert tok.__class__.__name__ == "ActionCodecTokenizer"
        assert tok.codebook_size == 2048

    def test_make_bin_tokenizer_scaffold(self):
        tok = make_codec({"kind": "tokens", "name": "bin", "action_dim": 8})
        assert tok.__class__.__name__ == "BinTokenizer"
        assert tok.codebook_size == 256

    def test_make_quest_tokenizer_scaffold(self):
        tok = make_codec({"kind": "tokens", "name": "quest", "action_dim": 8})
        assert tok.__class__.__name__ == "QueSTTokenizer"
        assert tok.codebook_size == 1920

    def test_make_oat_latent_scaffold(self):
        codec = make_codec({"kind": "latents", "name": "oat", "action_dim": 8})
        assert codec.__class__.__name__ == "OATAutoencoder"
        assert codec.config.name == "oat"
        assert codec.supports_prefix_decode is True

    def test_make_acodec_latent_scaffold(self):
        codec = make_codec({"kind": "latents", "name": "acodec", "action_dim": 8})
        assert codec.__class__.__name__ == "ActionCodecAutoencoder"
        assert codec.config.name == "acodec"

    def test_make_quest_latent_scaffold(self):
        codec = make_codec({"kind": "latents", "name": "quest", "action_dim": 8})
        assert codec.__class__.__name__ == "QueSTAutoencoder"
        assert codec.config.name == "quest"

    @pytest.mark.parametrize("name", ["acodec", "boat", "oat", "bin", "quest"])
    def test_make_trainable_tokenizer_requires_action_dim_without_pretrained(
        self, name: str
    ):
        with pytest.raises(ValueError, match="requires action_dim"):
            make_codec({"kind": "tokens", "name": name})


@pytest.mark.parametrize(
    "config_cls",
    [
        ActionCodecTokenizerConfig,
        BinTokenizerConfig,
        BOATTokenizerConfig,
        FASTTokenizerConfig,
        OATTokenizerConfig,
        QueSTTokenizerConfig,
    ],
)
def test_tokenizer_configs_reject_unknown_keys(config_cls):
    with pytest.raises(ValueError, match="unknown_key"):
        config_cls.from_dict({"name": "test", "unknown_key": 1})
