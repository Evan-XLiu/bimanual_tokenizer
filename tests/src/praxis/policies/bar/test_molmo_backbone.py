"""BAR+BTT Molmo registry and native input-path checks."""

from pathlib import Path
import re
from types import SimpleNamespace

import torch
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from praxis.policies.bar.backbones.molmo.molmo_base import (
    MolmoPolicyBase,
    _molmo_channels_last_observations,
)
from praxis.policies.bar.variants import BTTMolmoPolicy, BTTMolmoPolicyConfig
from praxis.policies.factory import get_policy_class
from .backbones.pali._test_utils import (
    _DummyFASTTokenizer, _DummyFixedTokenizer, _DummyProcessor, _DummyVLM,
    _make_io, _stats_for_io, BasicPrefixAdapter,
)


@pytest.mark.parametrize("codec", ["btt", "oat", "fast"])
def test_molmo_codec_initialization_and_training_forward(monkeypatch, codec):
    policy_cls = get_policy_class(f"bar_{codec}_molmo")
    tokenizer = _DummyFASTTokenizer() if codec == "fast" else _DummyFixedTokenizer()
    vlm, processor = _DummyVLM(), _DummyProcessor()
    monkeypatch.setattr(
        "praxis.policies.bar.core.vlm.base.make_codec", lambda cfg: tokenizer
    )
    def load(self, *args):
        self.vlm, self.processor = vlm, processor
    monkeypatch.setattr(MolmoPolicyBase, "_load_vlm", load)
    def build_inputs(self, obs, *, action_token_strings=None, instructions=None):
        rows = [[0, 0] + [
            self.processor.tokenizer.convert_tokens_to_ids(token)
            for token in re.findall(r"<\|[^|]+\|>", row)
        ] for row in action_token_strings]
        input_ids = torch.tensor(rows, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
    monkeypatch.setattr(MolmoPolicyBase, "_build_vlm_inputs", build_inputs)
    monkeypatch.setattr(
        MolmoPolicyBase, "_build_vlm_prefix_adapter",
        lambda self: BasicPrefixAdapter(self.vlm),
    )
    io = _make_io()
    policy = policy_cls.from_io(io, vlm_dtype="float32", gradient_checkpointing=False)
    policy.set_normalization_stats(_stats_for_io(io))
    assert policy._PREFIX_ATTENTION == "causal"
    assert policy.config.freeze_action_tokenizer
    assert not policy.action_tokenizer.training
    if codec == "fast":
        assert policy.max_decode_tokens == tokenizer.max_code_seq_len + 1
        assert policy._supervise_eos
    loss, metrics = policy({
        "action": torch.randn(2, 2, 2),
        "observation.state": torch.randn(2, 1, 4),
    })
    assert torch.isfinite(loss)


def test_btt_molmo_hydra_and_registry() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(Path("src/praxis/configs").resolve()), version_base=None
    ):
        cfg = compose(config_name="train_policy", overrides=["policy=bar_btt_molmo"])
    assert get_policy_class(cfg.policy.name) is BTTMolmoPolicy
    assert cfg.policy.vlm_model_name == "allenai/Molmo2-ER"
    assert cfg.policy.action_tokenizer.name == "btt"
    assert cfg.policy.state_prefix_encoding == "discrete_text"
    assert cfg.policy.action_horizon == 32
    assert cfg.policy.n_action_steps == 16
    assert BTTMolmoPolicyConfig().state_prefix_num_bins == 256


def test_molmo_bar_inputs_include_current_state_and_action_targets(monkeypatch) -> None:
    captured = {}

    def fake_render(**kwargs):
        captured.update(kwargs)
        return ["rendered"]

    def fake_processor(**kwargs):
        captured["image_batch"] = kwargs["image_batch"]
        captured["text"] = kwargs["text"]
        return SimpleNamespace(
            to=lambda device: {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
                "pixel_values": torch.ones(1, 2, dtype=torch.float32),
            }
        )

    monkeypatch.setattr(
        "praxis.policies.bar.backbones.molmo.molmo_base.render_chat_texts",
        fake_render,
    )
    monkeypatch.setattr(
        "praxis.policies.bar.backbones.molmo.molmo_base.run_molmo_processor",
        fake_processor,
    )
    policy = MolmoPolicyBase.__new__(MolmoPolicyBase)
    torch.nn.Module.__init__(policy)
    policy.config = BTTMolmoPolicyConfig(vlm_dtype="float32")
    policy.io = SimpleNamespace(
        image_features=[SimpleNamespace(key="camera", shape=(3, 4, 4))],
        state_features=[SimpleNamespace(key="state", shape=(14,))],
    )
    class FakeVlm(torch.nn.Linear):
        @property
        def device(self):
            return self.weight.device

    policy.vlm = FakeVlm(1, 1)
    policy.processor = SimpleNamespace()
    obs = {
        "camera": torch.zeros(1, 4, 4, 3),
        "state": torch.zeros(1, 14),
    }
    result = policy._build_vlm_inputs(
        obs, action_token_strings=["<|action_1|>"], instructions=["box"]
    )
    assert captured["state_texts"] == ["State: " + " ".join(["128"] * 14)]
    assert captured["action_token_strings"] == ["<|action_1|>"]
    assert captured["include_system"] is False
    assert captured["image_batch"].shape == (1, 1, 4, 4, 3)
    assert result["pixel_values"].dtype == torch.float32


def test_molmo_camera_layout_handles_tiny_chw_and_three_real_sizes() -> None:
    features = [
        SimpleNamespace(key="top", shape=(3, 360, 640)),
        SimpleNamespace(key="left", shape=(3, 480, 640)),
        SimpleNamespace(key="right", shape=(3, 480, 640)),
    ]
    obs = {
        "top": torch.full((1, 2, 3, 360, 640), 0.25),
        "left": torch.full((1, 2, 480, 640, 3), 0.5),
        "right": torch.full((1, 2, 3, 480, 640), 0.75),
    }
    converted = _molmo_channels_last_observations(obs, features)
    assert converted["top"].shape == (1, 2, 360, 640, 3)
    assert converted["left"].shape == (1, 2, 480, 640, 3)
    assert converted["right"].shape == (1, 2, 480, 640, 3)
    assert converted["top"][0, -1, 0, 0].tolist() == [0.25] * 3

    tiny = _molmo_channels_last_observations(
        {"camera": torch.tensor([[[[0.0]], [[0.5]], [[1.0]]]])},
        [SimpleNamespace(key="camera", shape=(3, 1, 1))],
    )
    assert tiny["camera"].shape == (1, 1, 1, 3)
    assert tiny["camera"][0, 0, 0].tolist() == [0.0, 0.5, 1.0]
