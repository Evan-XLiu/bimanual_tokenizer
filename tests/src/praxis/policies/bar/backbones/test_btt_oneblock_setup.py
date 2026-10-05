"""All three BAR backbones can select the isolated one-block BTT codec."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from praxis.codecs.factory import make_codec
from praxis.policies.factory import get_policy_class


@pytest.mark.parametrize("backbone", ["pali", "qwen", "smol"])
def test_registered_policy_and_hydra_alias(backbone):
    name = f"bar_btt_oneblock_{backbone}"
    cls = get_policy_class(name)
    assert cls.CONFIG_CLASS().action_tokenizer["name"] == "btt_oneblock"
    with initialize_config_dir(
        config_dir=str(Path("src/praxis/configs").resolve()), version_base=None
    ):
        cfg = compose(config_name="train_policy", overrides=[f"policy={name}"])
    assert cfg.policy.name == name
    assert cfg.policy.action_tokenizer.name == "btt_oneblock"
    assert cfg.policy.freeze_action_tokenizer
    assert cfg.policy.block_ce_weighting == "block_mean"


def test_tokenizer_hydra_config_inherits_exact_standard_values():
    with initialize_config_dir(
        config_dir=str(Path("src/praxis/configs").resolve()), version_base=None
    ):
        baseline = compose(
            config_name="train_codecs", overrides=["codecs@codec=tokens/btt"]
        )
        single = compose(
            config_name="train_codecs", overrides=["codecs@codec=tokens/btt_oneblock"]
        )
    a = OmegaConf.to_container(baseline.codec, resolve=True)
    b = OmegaConf.to_container(single.codec, resolve=True)
    assert a.pop("name") == "btt"
    assert b.pop("name") == "btt_oneblock"
    assert a == b
    codec = make_codec(single.codec)
    assert codec.block_schedule == (16,)
    assert sum(p.numel() for p in codec.parameters()) == 7382802
