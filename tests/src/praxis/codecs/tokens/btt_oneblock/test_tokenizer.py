"""The one-block interface must not change BTT's autoencoder or weights."""

import pytest
import torch

from praxis.codecs.factory import available_codecs, make_codec
from praxis.codecs.tokens.btt import BTTEncoderConfig, BTTTokenizer
from praxis.codecs.tokens.btt_oneblock import BTTOneBlockConfig, BTTOneBlockTokenizer
from praxis.core.types import FeatureSpec, FeatureType, NormMode, PolicyIO


def tiny_config():
    return {
        "model_dim": 32,
        "num_heads": 4,
        "decoder_nhead": 4,
        "decoder_nlayer": 1,
        "dropout": 0.0,
    }


def test_same_weights_same_tokens_reconstruction_and_training_gradients():
    torch.manual_seed(42)
    standard = BTTTokenizer(BTTEncoderConfig(**tiny_config()))
    torch.manual_seed(42)
    single = BTTOneBlockTokenizer(BTTOneBlockConfig(**tiny_config()))
    assert standard.block_schedule == (8, 8)
    assert single.block_schedule == (16,)
    assert standard.state_dict().keys() == single.state_dict().keys()
    for key, value in standard.state_dict().items():
        torch.testing.assert_close(value, single.state_dict()[key], rtol=0, atol=0)
    actions = torch.randn(2, 32, 14)
    for model in (standard, single):
        model.eval()
    tokens = standard.tokenize(actions)
    assert torch.equal(tokens, single.tokenize(actions))
    left, right = single.encode(actions)
    assert torch.equal(tokens, torch.cat((left, right), dim=1))
    torch.testing.assert_close(
        standard.detokenize(tokens), single.detokenize(tokens), rtol=0, atol=0
    )
    standard.autoencode(actions).square().mean().backward()
    single.autoencode(actions).square().mean().backward()
    for (name, p), (other_name, q) in zip(
        standard.named_parameters(), single.named_parameters(), strict=True
    ):
        assert name == other_name
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


def test_factory_and_save_load_preserve_single_block_identity(tmp_path):
    assert "btt_oneblock" in available_codecs(kind="tokens")
    codec = make_codec({"kind": "tokens", "name": "btt_oneblock", **tiny_config()})
    codec.set_io(
        PolicyIO(
            action_feature=FeatureSpec(
                key="action", type=FeatureType.ACTION, shape=(14,)
            )
        )
    )
    codec.set_normalization_stats(
        {"action": {"min": torch.full((14,), -2.0), "max": torch.full((14,), 2.0)}},
        modes={"action": NormMode.MIN_MAX},
    )
    codec.save_pretrained(tmp_path)
    restored = make_codec(
        {"kind": "tokens", "name": "btt_oneblock", "pretrained_path": str(tmp_path)}
    )
    assert isinstance(restored, BTTOneBlockTokenizer)
    assert restored.block_schedule == (16,)
    assert restored.config.name == "btt_oneblock"
    assert restored.io == codec.io
    assert restored.codebook_size == 1920
    assert restored.max_code_seq_len == 16
    for key, value in codec.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)


def test_wrong_checkpoint_identity_requires_explicit_conversion():
    with pytest.raises(ValueError, match="explicitly converted"):
        BTTOneBlockConfig(name="btt")
