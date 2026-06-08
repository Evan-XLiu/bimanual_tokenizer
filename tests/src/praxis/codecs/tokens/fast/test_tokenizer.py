# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for FAST tokenizer (DCT + BPE)."""

from __future__ import annotations

import tempfile
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

from praxis.codecs.tokens.fast.config import FASTTokenizerConfig
from praxis.codecs.tokens.fast.dct import dct, idct
from praxis.core.types import FeatureSpec, FeatureType, NormMode, PolicyIO

if TYPE_CHECKING:
    from praxis.codecs.tokens.fast.tokenizer import FASTTokenizer


# ---------------------------------------------------------------------------
# DCT tests (standalone, no BPE needed)
# ---------------------------------------------------------------------------


class TestDCT:
    def test_roundtrip_identity(self):
        """DCT then IDCT should recover the original signal."""
        x = torch.randn(2, 16, 4)
        reconstructed = idct(dct(x, dim=-2), dim=-2)
        torch.testing.assert_close(reconstructed, x, atol=1e-5, rtol=1e-5)

    def test_roundtrip_1d(self):
        """Works on a 1-D-ish signal along the specified dim."""
        x = torch.randn(32, 3)
        reconstructed = idct(dct(x, dim=0), dim=0)
        torch.testing.assert_close(reconstructed, x, atol=1e-5, rtol=1e-5)

    def test_orthonormality(self):
        """DCT matrix should be orthogonal: C @ C^T = I."""
        from praxis.codecs.tokens.fast.dct import _dct_matrix

        n = 16
        mat = _dct_matrix(n, torch.float64, torch.device("cpu"))
        eye = mat @ mat.t()
        torch.testing.assert_close(
            eye, torch.eye(n, dtype=torch.float64), atol=1e-12, rtol=1e-12
        )

    def test_different_dims(self):
        """DCT along different axes should all roundtrip correctly."""
        x = torch.randn(3, 8, 5)
        for dim in (-2, -1, 0, 1, 2):
            reconstructed = idct(dct(x, dim=dim), dim=dim)
            torch.testing.assert_close(reconstructed, x, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Tokenizer tests (require BPE model download from HF Hub)
# ---------------------------------------------------------------------------


def _have_bpe_model() -> bool:
    """Check if the BPE model can be loaded (requires network + model)."""
    try:
        from transformers import PreTrainedTokenizerFast

        PreTrainedTokenizerFast.from_pretrained("physical-intelligence/fast")
        return True
    except Exception:
        return False


_skip_no_bpe = pytest.mark.skipif(
    not _have_bpe_model(),
    reason="BPE model 'physical-intelligence/fast' not available",
)


def _make_tokenizer(**overrides) -> FASTTokenizer:
    from praxis.codecs.tokens.fast.tokenizer import FASTTokenizer

    base = {"action_dim": 4, "action_horizon": 8, "max_token_length": 128}
    base.update(overrides)
    tok = FASTTokenizer(config=FASTTokenizerConfig(**base))
    tok.set_io(
        PolicyIO(
            obs_features=[],
            action_feature=FeatureSpec(
                key="action",
                type=FeatureType.ACTION,
                shape=(int(base["action_dim"]),),
                norm_mode=NormMode.MIN_MAX,
            ),
            n_obs_steps=0,
            n_action_steps=int(base["action_horizon"]),
        )
    )
    return tok


@_skip_no_bpe
class TestFASTTokenizer:
    def test_tokenize_shape(self):
        tok = _make_tokenizer()
        actions = torch.randn(2, 8, 4)
        tokens = tok.tokenize(actions)
        assert tokens.dtype == torch.long
        assert tokens.shape == (2, 128)

    def test_token_range(self):
        tok = _make_tokenizer()
        actions = torch.randn(3, 8, 4)
        tokens = tok.tokenize(actions)
        assert int(tokens.min()) >= 0
        assert int(tokens.max()) <= tok.codebook_size - 1

    def test_detokenize_shape(self):
        tok = _make_tokenizer()
        actions = torch.randn(2, 8, 4)
        tokens = tok.tokenize(actions)
        reconstructed = tok.detokenize(tokens)
        assert reconstructed.shape == (2, 8, 4)

    def test_roundtrip_accuracy(self):
        """Tokenize then detokenize should produce a reasonable approximation."""
        tok = _make_tokenizer()
        actions = torch.randn(2, 8, 4) * 0.5  # moderate range
        tokens = tok.tokenize(actions)
        reconstructed = tok.detokenize(tokens)
        # BPE is lossy but should be in the right ballpark
        error = (actions - reconstructed).abs().mean()
        assert error < 1.0, f"Mean roundtrip error {error:.4f} too large"

    def test_codebook_size(self):
        tok = _make_tokenizer()
        # The codebook includes the BPE vocabulary plus one pad token.
        assert tok.codebook_size == tok._bpe_vocab_size + 1

    def test_padding(self):
        """Shorter sequences should be padded with pad_token_id."""
        tok = _make_tokenizer(max_token_length=512)
        actions = torch.randn(1, 8, 4)
        tokens = tok.tokenize(actions)
        assert tokens.shape == (1, 512)
        # At least some tokens at the end should be pad
        assert (tokens[0] == tok._pad_token_id).any()

    def test_strict_decode_raises_on_length_mismatch(self):
        """Feeding garbage tokens in strict mode should raise ValueError."""
        tok = _make_tokenizer(strict_decode=True)
        # Craft tokens that will BPE-decode to wrong length
        garbage = torch.zeros(1, 128, dtype=torch.long)
        garbage[0, 0] = 1  # single real token, rest are pad
        with pytest.raises(ValueError, match="BPE decoded"):
            tok.detokenize(garbage)

    def test_strict_decode_override_at_call_site(self):
        """Call-site strict_decode overrides config default."""
        tok = _make_tokenizer(strict_decode=False)  # lenient by default
        garbage = torch.zeros(1, 128, dtype=torch.long)
        garbage[0, 0] = 1
        # Override to strict at call site
        with pytest.raises(ValueError, match="BPE decoded"):
            tok.detokenize(garbage, strict_decode=True)
        # And the reverse: strict config, lenient call
        tok2 = _make_tokenizer(strict_decode=True)
        result = tok2.detokenize(garbage, strict_decode=False)
        assert result.shape == (1, 8, 4)

    def test_lenient_decode_pads_short_output(self):
        """Default lenient mode should pad short BPE output, not raise."""
        tok = _make_tokenizer(strict_decode=False)
        garbage = torch.zeros(1, 128, dtype=torch.long)
        garbage[0, 0] = 1
        result = tok.detokenize(garbage)
        assert result.shape == (1, 8, 4)

    def test_normalization_applied(self):
        tok = _make_tokenizer()
        stats = {
            "action": {
                "min": torch.zeros(4),
                "max": torch.ones(4) * 10.0,
            }
        }
        tok.set_normalization_stats(stats)
        actions = torch.ones(1, 8, 4) * 5.0
        tokens_norm = tok.tokenize(actions)
        # Just verify it runs without error and produces valid tokens
        assert tokens_norm.shape == (1, 128)
        assert int(tokens_norm.min()) >= 0


# ---------------------------------------------------------------------------
# fit() tests (no pretrained BPE needed)
# ---------------------------------------------------------------------------


class TestFASTFit:
    def test_set_io_infers_action_dim_when_unset(self):
        from praxis.codecs.tokens.fast.tokenizer import FASTTokenizer

        tok = FASTTokenizer(
            config=FASTTokenizerConfig(
                action_dim=None,
                action_horizon=8,
                bpe_tokenizer_path=None,
            )
        )
        tok.set_io(
            PolicyIO(
                obs_features=[],
                action_feature=FeatureSpec(
                    key="action",
                    type=FeatureType.ACTION,
                    shape=(4,),
                    norm_mode=NormMode.MIN_MAX,
                ),
                n_obs_steps=0,
                n_action_steps=8,
            )
        )
        assert tok.action_dim == 4

    def test_set_io_raises_on_action_dim_mismatch(self):
        from praxis.codecs.tokens.fast.tokenizer import FASTTokenizer

        tok = FASTTokenizer(
            config=FASTTokenizerConfig(
                action_dim=3,
                action_horizon=8,
                bpe_tokenizer_path=None,
            )
        )
        with pytest.raises(ValueError, match="action_dim mismatch"):
            tok.set_io(
                PolicyIO(
                    obs_features=[],
                    action_feature=FeatureSpec(
                        key="action",
                        type=FeatureType.ACTION,
                        shape=(4,),
                        norm_mode=NormMode.MIN_MAX,
                    ),
                    n_obs_steps=0,
                    n_action_steps=8,
                )
            )

    def _make_unfitted(self, **overrides) -> FASTTokenizer:
        from praxis.codecs.tokens.fast.tokenizer import FASTTokenizer

        base = {
            "action_dim": 4,
            "action_horizon": 8,
            "max_token_length": 128,
            "bpe_tokenizer_path": None,
            "vocab_size": 256,
        }
        base.update(overrides)
        tok = FASTTokenizer(config=FASTTokenizerConfig(**base))
        tok.set_io(
            PolicyIO(
                obs_features=[],
                action_feature=FeatureSpec(
                    key="action",
                    type=FeatureType.ACTION,
                    shape=(int(base["action_dim"]),),
                    norm_mode=NormMode.MIN_MAX,
                ),
                n_obs_steps=0,
                n_action_steps=int(base["action_horizon"]),
            )
        )
        return tok

    def test_no_bpe_raises_on_tokenize(self):
        tok = self._make_unfitted()
        with pytest.raises(RuntimeError, match="BPE tokenizer is not initialized"):
            tok.tokenize(torch.randn(1, 8, 4))

    def test_max_code_seq_len_exposes_decode_budget(self):
        tok = self._make_unfitted(max_token_length=32)
        assert tok.variable_length_codes is True
        assert tok.max_code_seq_len == 32

    def test_no_bpe_raises_on_detokenize(self):
        tok = self._make_unfitted()
        with pytest.raises(RuntimeError, match="BPE tokenizer is not initialized"):
            tok.detokenize(torch.zeros(1, 128, dtype=torch.long))

    def test_lenient_decode_truncates_and_pads_missing_coefficients_as_zero(self):
        class _DecodeBPE:
            def __init__(self, decoded: str) -> None:
                self.decoded = decoded

            def batch_decode(self, ids):
                return [self.decoded for _ in ids]

        tok = self._make_unfitted(action_dim=1, action_horizon=4, min_token=-5)
        tok.scale = 1.0
        tok.config.scale = 1.0
        zero_coeff_code = -tok.min_token
        tok._bpe = _DecodeBPE(
            "".join(chr(v) for v in [zero_coeff_code + 4, zero_coeff_code])
        )
        tok._bpe_vocab_size = 1
        tok._pad_token_id = 1

        tokens = torch.full((1, 8), tok._pad_token_id, dtype=torch.long)
        reconstructed = tok.detokenize(tokens)
        coeffs = torch.tensor([[[4.0], [0.0], [0.0], [0.0]]])
        expected = idct(coeffs, dim=-2)

        torch.testing.assert_close(reconstructed, expected)

        tok._bpe = _DecodeBPE(
            "".join(
                chr(v)
                for v in [
                    zero_coeff_code,
                    zero_coeff_code,
                    zero_coeff_code,
                    zero_coeff_code,
                    zero_coeff_code + 99,
                ]
            )
        )
        reconstructed = tok.detokenize(tokens)
        torch.testing.assert_close(reconstructed, torch.zeros(1, 4, 1))

    def test_fit_enables_tokenize(self):
        tok = self._make_unfitted()
        actions = torch.randn(50, 8, 4) * 0.5
        tok.fit(actions)
        assert tok._bpe is not None
        assert tok._bpe_vocab_size > 0
        tokens = tok.tokenize(actions[:2])
        assert tokens.shape == (2, 128)
        assert tokens.dtype == torch.long

    def test_fit_roundtrip(self):
        tok = self._make_unfitted()
        actions = torch.randn(100, 8, 4) * 0.3
        tok.fit(actions)
        tokens = tok.tokenize(actions[:5])
        recon = tok.detokenize(tokens)
        assert recon.shape == (5, 8, 4)
        # Should be a reasonable approximation
        error = (actions[:5] - recon).abs().mean()
        assert error < 2.0, f"Mean roundtrip error {error:.4f} too large after fit"

    def test_fit_updates_min_token(self):
        tok = self._make_unfitted()
        actions = torch.randn(50, 8, 4) * 0.5
        tok.fit(actions)
        # min_token should have been set from data
        assert tok.min_token == tok.config.min_token

    def test_fit_uses_bytelevel_bpe_configuration(self):
        tok = self._make_unfitted()
        actions = torch.randn(80, 8, 4) * 0.3
        tok.fit(actions)

        assert tok._bpe is not None
        assert "ByteLevel" in str(tok._bpe.backend_tokenizer.pre_tokenizer)
        assert "ByteLevel" in str(tok._bpe.backend_tokenizer.decoder)
        assert tok._bpe.clean_up_tokenization_spaces is False

    def test_fit_raises_when_vocab_too_small_for_token_range(self):
        tok = self._make_unfitted(vocab_size=8)
        actions = torch.tensor(
            [
                [[0.0, 0.0, 0.0, 0.0]] * 8,
                [[50.0, -50.0, 25.0, -25.0]] * 8,
            ],
            dtype=torch.float32,
        )
        with pytest.raises(ValueError, match="vocab_size=8 is too small"):
            tok.fit(actions)

    def test_fit_save_load_roundtrip(self):
        from praxis.codecs.tokens.fast.tokenizer import FASTTokenizer

        tok = self._make_unfitted()
        actions = torch.randn(100, 8, 4) * 0.3
        tok.fit(actions)

        tokens_before = tok.tokenize(actions[:3])

        with tempfile.TemporaryDirectory() as tmpdir:
            tok.save_pretrained(tmpdir, config=tok.config.to_dict())
            tok2 = FASTTokenizer.from_pretrained(tmpdir)

        assert tok2._bpe is not None
        assert tok2._bpe_vocab_size == tok._bpe_vocab_size
        assert tok2.min_token == tok.min_token

        tokens_after = tok2.tokenize(actions[:3])
        assert torch.equal(tokens_before, tokens_after)

    def test_fit_save_load_with_norm_stats(self):
        from praxis.codecs.tokens.fast.tokenizer import FASTTokenizer

        tok = self._make_unfitted()
        stats = {
            "action": {
                "min": torch.zeros(4),
                "max": torch.ones(4) * 10.0,
            }
        }
        tok.set_normalization_stats(stats)
        actions = torch.randn(100, 8, 4) * 0.3
        tok.fit(actions)

        with tempfile.TemporaryDirectory() as tmpdir:
            tok.save_pretrained(tmpdir, config=tok.config.to_dict())
            tok2 = FASTTokenizer.from_pretrained(tmpdir)

        assert tok2.normalize_action is not None
        assert tok2.unnormalize_action is not None
        assert tok2._bpe is not None

    def test_tokenize_unpadded_returns_variable_length_list(self):
        tok = self._make_unfitted()
        actions = torch.randn(30, 8, 4) * 0.3
        tok.fit(actions)
        tokens = tok.tokenize(actions[:4], pad=False)
        assert isinstance(tokens, list)
        assert len(tokens) == 4
        assert all(isinstance(row, torch.Tensor) for row in tokens)
        assert all(row.ndim == 1 for row in tokens)

    def test_tokenize_custom_pad_value(self):
        tok = self._make_unfitted(max_token_length=32)
        actions = torch.randn(30, 8, 4) * 0.3
        tok.fit(actions)
        custom_pad = tok.codebook_size + 7
        padded, lengths = tok.tokenize(
            actions[:3], pad=True, pad_value=custom_pad, return_lengths=True
        )
        assert padded.shape == (3, 32)
        assert lengths.shape == (3,)
        for i in range(3):
            length_i = int(lengths[i].item())
            if length_i < 32:
                assert torch.all(padded[i, length_i:] == custom_pad)
        reconstructed = tok.detokenize(padded, pad_value=custom_pad)
        assert reconstructed.shape == actions[:3].shape


# ---------------------------------------------------------------------------
# Config tests (no BPE needed)
# ---------------------------------------------------------------------------


class TestFASTConfig:
    def test_config_roundtrip(self):
        cfg = FASTTokenizerConfig(action_dim=6, scale=20.0)
        d = cfg.to_dict()
        cfg2 = FASTTokenizerConfig.from_dict(d)
        assert cfg2.action_dim == 6
        assert cfg2.scale == 20.0

    def test_config_rejects_unknown_keys(self):
        with pytest.raises(ValueError, match="unknown_key"):
            FASTTokenizerConfig.from_dict({"action_dim": 3, "unknown_key": 42})

    def test_config_vocab_size_default(self):
        cfg = FASTTokenizerConfig()
        assert cfg.vocab_size == 1024

    def test_config_bpe_path_none(self):
        cfg = FASTTokenizerConfig(bpe_tokenizer_path=None)
        assert cfg.bpe_tokenizer_path is None


def test_from_pretrained_loads_colocated_hub_bpe(monkeypatch):
    from praxis.codecs.tokens.core import BaseTokenizer
    from praxis.codecs.tokens.fast.tokenizer import FASTTokenizer

    fake_bpe = SimpleNamespace(
        vocab_size=11,
        backend_tokenizer=SimpleNamespace(decoder=object()),
    )
    calls = []

    def _fake_base_from_pretrained(cls, *, model_id, **kwargs):
        return cls(config={"action_dim": 4, "bpe_tokenizer_path": None})

    def _fake_bpe_from_pretrained(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return fake_bpe

    monkeypatch.setattr(
        BaseTokenizer,
        "_from_pretrained",
        classmethod(_fake_base_from_pretrained),
    )
    monkeypatch.setattr(
        "praxis.codecs.tokens.fast.tokenizer.PreTrainedTokenizerFast.from_pretrained",
        _fake_bpe_from_pretrained,
    )

    tok = FASTTokenizer._from_pretrained(
        model_id="org/fitted-fast",
        revision="dev",
        local_files_only=True,
        token="secret",
    )

    assert tok._bpe is fake_bpe
    assert tok._bpe_vocab_size == 11
    assert tok._pad_token_id == 11
    assert calls == [
        (
            "org/fitted-fast",
            {
                "revision": "dev",
                "cache_dir": None,
                "force_download": False,
                "proxies": None,
                "resume_download": None,
                "local_files_only": True,
                "token": "secret",
            },
        )
    ]


def test_from_pretrained_propagates_invalid_colocated_bpe(monkeypatch):
    from praxis.codecs.tokens.core import BaseTokenizer
    from praxis.codecs.tokens.fast.tokenizer import FASTTokenizer

    invalid_bpe = SimpleNamespace(
        vocab_size=11,
        backend_tokenizer=SimpleNamespace(decoder=None),
    )

    def _fake_base_from_pretrained(cls, *, model_id, **kwargs):
        del model_id, kwargs
        return cls(config={"action_dim": 4, "bpe_tokenizer_path": None})

    monkeypatch.setattr(
        BaseTokenizer,
        "_from_pretrained",
        classmethod(_fake_base_from_pretrained),
    )
    monkeypatch.setattr(
        "praxis.codecs.tokens.fast.tokenizer.PreTrainedTokenizerFast.from_pretrained",
        lambda *args, **kwargs: invalid_bpe,
    )

    with pytest.raises(RuntimeError, match="no decoder"):
        FASTTokenizer._from_pretrained(model_id="org/fitted-fast")


def test_from_pretrained_prefers_local_bpe_before_configured_bpe_path(
    monkeypatch, tmp_path
):
    from praxis.codecs.tokens.core import BaseTokenizer
    from praxis.codecs.tokens.fast.tokenizer import FASTTokenizer

    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    fake_bpe = SimpleNamespace(
        vocab_size=11,
        backend_tokenizer=SimpleNamespace(decoder=object()),
    )
    calls = []

    def _fake_base_from_pretrained(cls, *, model_id, **kwargs):
        del model_id
        assert kwargs["config"]["bpe_tokenizer_path"] is None
        return cls(config=kwargs["config"])

    def _fake_bpe_from_pretrained(model_id, **kwargs):
        del kwargs
        calls.append(str(model_id))
        if str(model_id) != str(tmp_path):
            raise AssertionError(f"unexpected BPE load from {model_id}")
        return fake_bpe

    monkeypatch.setattr(
        BaseTokenizer,
        "_from_pretrained",
        classmethod(_fake_base_from_pretrained),
    )
    monkeypatch.setattr(
        "praxis.codecs.tokens.fast.tokenizer.PreTrainedTokenizerFast.from_pretrained",
        _fake_bpe_from_pretrained,
    )

    tok = FASTTokenizer._from_pretrained(
        model_id=tmp_path,
        config={
            "action_dim": 4,
            "action_horizon": 8,
            "bpe_tokenizer_path": "remote/fast-bpe",
        },
        local_files_only=True,
    )

    assert tok._bpe is fake_bpe
    assert tok.config.bpe_tokenizer_path is None
    assert calls == [str(tmp_path)]


def test_factory_runtime_rebuild_preserves_fitted_fast_metadata(monkeypatch):
    from praxis.codecs.factory import make_codec
    from praxis.codecs.tokens.fast.tokenizer import FASTTokenizer

    source_bpe = SimpleNamespace(
        vocab_size=17,
        backend_tokenizer=SimpleNamespace(decoder=object()),
    )
    target_bpe = SimpleNamespace(
        vocab_size=23,
        backend_tokenizer=SimpleNamespace(decoder=object()),
    )
    source = FASTTokenizer(
        config={
            "action_dim": 4,
            "action_horizon": 6,
            "max_token_length": 42,
            "strict_decode": True,
            "scale": 12.5,
            "min_token": -7,
            "vocab_size": 17,
            "bpe_tokenizer_path": None,
        }
    )
    source._set_bpe(source_bpe)
    source.config.bpe_tokenizer_path = None

    @classmethod
    def _fake_from_pretrained(cls, path):
        del path
        return source

    monkeypatch.setattr(FASTTokenizer, "from_pretrained", _fake_from_pretrained)
    monkeypatch.setattr(
        "praxis.codecs.tokens.fast.tokenizer.PreTrainedTokenizerFast.from_pretrained",
        lambda *args, **kwargs: target_bpe,
    )

    tok = make_codec(
        {
            "kind": "tokens",
            "name": "fast",
            "pretrained_path": "/tmp/fitted-fast",
            "action_dim": 4,
            "bpe_tokenizer_path": "physical-intelligence/fast",
        }
    )

    assert tok._bpe is source_bpe
    assert tok._bpe_vocab_size == 17
    assert tok.action_horizon == 6
    assert tok.max_token_length == 42
    assert tok.strict_decode is True
    assert tok.scale == 12.5
    assert tok.min_token == -7
    assert tok.config.action_horizon == 6
    assert tok.config.max_token_length == 42
    assert tok.config.strict_decode is True
    assert tok.config.scale == 12.5
    assert tok.config.min_token == -7
    assert tok.config.vocab_size == 17
    assert tok.config.bpe_tokenizer_path is None


def test_factory_rejects_fitted_fast_metadata_override(monkeypatch):
    from praxis.codecs.factory import make_codec
    from praxis.codecs.tokens.fast.tokenizer import FASTTokenizer

    source_bpe = SimpleNamespace(
        vocab_size=17,
        backend_tokenizer=SimpleNamespace(decoder=object()),
    )
    source = FASTTokenizer(
        config={
            "action_dim": 4,
            "action_horizon": 6,
            "max_token_length": 42,
            "strict_decode": True,
            "bpe_tokenizer_path": None,
        }
    )
    source._set_bpe(source_bpe)

    @classmethod
    def _fake_from_pretrained(cls, path):
        del cls, path
        return source

    monkeypatch.setattr(FASTTokenizer, "from_pretrained", _fake_from_pretrained)

    with pytest.raises(ValueError, match=r"tokenizer\.max_token_length"):
        make_codec(
            {
                "kind": "tokens",
                "name": "fast",
                "pretrained_path": "/tmp/fitted-fast",
                "max_token_length": 128,
            }
        )


@_skip_no_bpe
class TestFASTFactory:
    def test_factory_available(self):
        from praxis.codecs.factory import available_codecs

        assert "fast" in available_codecs(kind="tokens")

    def test_factory_make(self):
        from praxis.codecs.factory import make_codec

        tok = make_codec(
            {"kind": "tokens", "name": "fast", "action_dim": 4, "action_horizon": 8}
        )
        assert tok.__class__.__name__ == "FASTTokenizer"
        assert tok.codebook_size > 0
