# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for BaseTokenizer contract behavior."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from praxis.codecs.core import BaseCodec
from praxis.codecs.tokens.core import BaseTokenizer
from praxis.core.checkpoint_artifacts import NORM_STATS_FILENAME, POLICY_IO_FILENAME
from praxis.core.types import FeatureSpec, FeatureType, NormMode, PolicyIO


class _AlgoTokenizer(BaseTokenizer):
    @property
    def codebook_size(self) -> int:
        return 256

    @property
    def max_code_seq_len(self) -> int:
        return 4

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


class _TrainableTokenizer(_AlgoTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2)

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        pred = self.proj(batch[self.action_key])
        loss = ((pred - batch[self.action_key]) ** 2).mean()
        return loss, {"loss": float(loss.item())}


def _tokenizer_io() -> PolicyIO:
    return PolicyIO(
        obs_features=[],
        action_feature=FeatureSpec(
            key="action",
            type=FeatureType.ACTION,
            shape=(2,),
            norm_mode=NormMode.MIN_MAX,
        ),
        n_obs_steps=0,
        n_action_steps=1,
    )


class TestBaseTokenizer:
    def test_default_flags_for_algo_tokenizer(self):
        tok = _AlgoTokenizer()
        assert isinstance(tok, BaseCodec)
        assert tok.variable_length_codes is False
        assert tok.max_code_seq_len == 4
        assert tok.block_schedule is None
        assert tok.supports_prefix_decode is False

    def test_non_learning_tokenizer_has_empty_optim_params(self):
        tok = _AlgoTokenizer()
        assert tok.get_optim_params() == []

    def test_trainable_tokenizer_default_optim_params(self):
        tok = _TrainableTokenizer()
        groups = tok.get_optim_params()
        assert len(groups) == 1
        assert "params" in groups[0]
        assert len(list(groups[0]["params"])) > 0

    def test_forward_raises_for_non_trainable(self):
        tok = _AlgoTokenizer()
        with pytest.raises(NotImplementedError, match="does not implement forward"):
            tok.forward({"action": torch.zeros((1, 4, 2), dtype=torch.float32)})

    def test_plain_tokenizer_does_not_claim_latent_codec_api(self):
        tok = _AlgoTokenizer()
        assert not hasattr(tok, "encode")
        assert not hasattr(tok, "decode")

    def test_set_normalization_stats_builds_action_normalizers(self):
        tok = _AlgoTokenizer()
        tok.set_normalization_stats(
            {
                "action": {
                    "min": torch.tensor([-1.0, -1.0]),
                    "max": torch.tensor([1.0, 1.0]),
                }
            }
        )
        assert tok.normalize_action is not None
        assert tok.unnormalize_action is not None

    def test_set_normalization_stats_ignores_invalid_action_stats(self):
        tok = _AlgoTokenizer()
        tok.set_normalization_stats({"action": {"foo": torch.tensor([0.0, 0.0])}})
        assert tok.normalize_action is None
        assert tok.unnormalize_action is None

    def test_set_io_updates_action_key(self):
        tok = _TrainableTokenizer()
        io = PolicyIO(
            obs_features=[],
            action_feature=FeatureSpec(
                key="act",
                type=FeatureType.ACTION,
                shape=(2,),
                norm_mode=NormMode.MIN_MAX,
            ),
            n_obs_steps=0,
            n_action_steps=1,
        )
        tok.set_io(io)
        loss, info = tok.forward({"act": torch.zeros((1, 4, 2), dtype=torch.float32)})
        assert loss.ndim == 0

    def test_set_io_remaps_existing_action_normalizer_stats_to_new_key(self):
        tok = _AlgoTokenizer()
        tok.set_normalization_stats(
            {
                "action": {
                    "min": torch.tensor([-1.0, -2.0]),
                    "max": torch.tensor([1.0, 2.0]),
                    "mean": torch.tensor([0.5, -0.5]),
                    "std": torch.tensor([0.25, 0.5]),
                }
            },
            modes={"action": NormMode.MEAN_STD},
        )

        x = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)
        before = tok.normalize_action({"action": x})["action"]

        io = PolicyIO(
            obs_features=[],
            action_feature=FeatureSpec(
                key="robot_action",
                type=FeatureType.ACTION,
                shape=(2,),
                norm_mode=NormMode.MIN_MAX,
            ),
            n_obs_steps=0,
            n_action_steps=1,
        )
        tok.set_io(io)

        assert tok.action_key == "robot_action"
        assert tok.normalize_action is not None
        assert tok.unnormalize_action is not None
        assert tok.normalize_action.keys == ["robot_action"]
        after = tok.normalize_action({"robot_action": x})["robot_action"]
        torch.testing.assert_close(after, before)

    def test_save_pretrained_pushes_after_sidecars_are_written(
        self, tmp_path: Path, monkeypatch
    ):
        tok = _AlgoTokenizer()
        tok.set_io(_tokenizer_io())
        calls = []

        class _FakeHfApi:
            def __init__(self, *, token=None):
                calls.append(("init", token))

            def create_repo(self, *, repo_id, private=None, exist_ok=True):
                calls.append(("create_repo", repo_id, private, exist_ok))
                return SimpleNamespace(repo_id=repo_id)

            def upload_folder(self, *, folder_path, **kwargs):
                assert Path(folder_path) == tmp_path
                assert (tmp_path / POLICY_IO_FILENAME).exists()
                assert (tmp_path / NORM_STATS_FILENAME).exists()
                calls.append(("upload_folder", kwargs))
                return "https://huggingface.co/org/tok/commit/abc"

        monkeypatch.setattr("praxis.core.checkpoint_artifacts.HfApi", _FakeHfApi)

        result = tok.save_pretrained(
            tmp_path,
            push_to_hub=True,
            repo_id="org/tok",
            commit_message="publish",
        )

        assert result == "https://huggingface.co/org/tok/commit/abc"
        assert calls == [
            ("init", None),
            ("create_repo", "org/tok", None, True),
            (
                "upload_folder",
                {
                    "repo_id": "org/tok",
                    "repo_type": "model",
                    "commit_message": "publish",
                    "revision": None,
                    "create_pr": None,
                    "allow_patterns": None,
                    "ignore_patterns": None,
                    "delete_patterns": None,
                },
            ),
        ]

    def test_from_pretrained_forwards_hub_options_to_sidecars(
        self, tmp_path: Path, monkeypatch
    ):
        io_path = tmp_path / POLICY_IO_FILENAME
        io_path.write_text(json.dumps(_tokenizer_io().to_dict()) + "\n")
        norm_path = tmp_path / NORM_STATS_FILENAME
        torch.save({"stats": {}, "modes": {}}, norm_path)
        resolved = []
        mixin_kwargs = {}

        def _fake_resolve_sidecar_path(model_id, filename, *, hub_options=None):
            resolved.append((model_id, filename, hub_options))
            return io_path if filename == POLICY_IO_FILENAME else norm_path

        def _fake_mixin_from_pretrained(cls, *, model_id, **kwargs):
            mixin_kwargs.update(kwargs)
            return cls()

        monkeypatch.setattr(
            "praxis.codecs.core.base.resolve_sidecar_path",
            _fake_resolve_sidecar_path,
        )
        monkeypatch.setattr(
            PyTorchModelHubMixin,
            "_from_pretrained",
            classmethod(_fake_mixin_from_pretrained),
        )

        model = _AlgoTokenizer._from_pretrained(
            model_id="org/tok",
            revision="dev",
            cache_dir=tmp_path / "cache",
            local_files_only=True,
            token="secret",
        )

        assert model.io is not None
        assert mixin_kwargs["revision"] == "dev"
        assert mixin_kwargs["cache_dir"] == tmp_path / "cache"
        assert mixin_kwargs["local_files_only"] is True
        assert mixin_kwargs["token"] == "secret"
        assert [item[1] for item in resolved] == [
            POLICY_IO_FILENAME,
            NORM_STATS_FILENAME,
        ]
        for _model_id, _filename, hub_options in resolved:
            assert hub_options is not None
            assert hub_options.revision == "dev"
            assert hub_options.cache_dir == tmp_path / "cache"
            assert hub_options.local_files_only is True
            assert hub_options.token == "secret"
