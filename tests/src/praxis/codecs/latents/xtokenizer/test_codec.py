# SPDX-FileCopyrightText: 2026 Xiaoshen Han
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the trainable X-Tokenizer codec."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor

from praxis.codecs.latents.xtokenizer import (
    XTokenizerCodec,
    XTokenizerModelConfig,
)
from praxis.codecs.latents.xtokenizer.action_space import (
    to_observation_relative_actions,
)
from praxis.codecs.latents.xtokenizer.codec import (
    _masked_dct_l1,
    _masked_smoothness_l1,
    _multi_rotation_geodesic_loss,
)
from praxis.core.types import FeatureSpec, FeatureType, NormMode, PolicyIO
from praxis.training.core.training_context import TrainingContext


def _tiny_config(**overrides: object) -> XTokenizerModelConfig:
    values: dict[str, Any] = {
        "action_dim": 10,
        "action_horizon": 32,
        "compression_ratio": 4,
        "hidden_dim": 16,
        "latent_dim": 16,
        "encoder_num_heads": 4,
        "encoder_depth": 1,
        "decoder_num_heads": 4,
        "decoder_depth": 1,
        "mam_num_heads": 4,
        "mam_depth": 2,
        "prediction_num_heads": 4,
        "prediction_depth": 2,
        "dropout": 0.0,
        "codebook_size": 8,
        "num_quantizers": 4,
        "ema_decay": 0.8,
        "dead_code_threshold": 0,
        "embodiment_vocab_size": 4,
        "vl_feature_dim": 12,
        "mam_mask_probability": 1.0,
        "mam_warmup_epochs": 0,
        "mam_warmup_steps": 0,
        "dct_weight": 0.1,
        "smoothness_weight": 0.1,
    }
    values.update(overrides)
    return XTokenizerModelConfig.from_dict(values)


def _batch(*, requires_grad_target: bool = False) -> dict[str, Tensor]:
    torch.manual_seed(7)
    actions = torch.randn(2, 32, 10)
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    actions[:, :, 3:9] = identity
    actions[1, 24:] = float("nan")
    action_is_pad = torch.zeros(2, 32, dtype=torch.bool)
    action_is_pad[1, 24:] = True

    vl_features = torch.randn(2, 8, 3, 12)
    vl_features[1, 6:] = float("nan")
    vl_feature_is_pad = torch.zeros(2, 8, 3, dtype=torch.bool)
    vl_feature_is_pad[1, 6:] = True
    next_vl_features = torch.randn(2, 12)
    next_vl_features.requires_grad_(requires_grad_target)

    observation_state = torch.zeros(2, 10)
    observation_state[:, 3:9] = identity
    return {
        "action": actions,
        "observation.state": observation_state,
        "robot_type": torch.tensor([0, 1]),
        "action_is_pad": action_is_pad,
        "vl_features": vl_features,
        "vl_feature_is_pad": vl_feature_is_pad,
        "next_vl_features": next_vl_features,
    }


def _canonical_stats_identity(
    *,
    predict_min: Tensor,
    predict_max: Tensor,
    observation_min: Tensor,
    observation_max: Tensor,
) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "format_version": 3,
        "dataset_repo_id": "org/bridge",
        "dataset_revision": "a" * 40,
        "robot_type": "WidowX",
        "action_layout": {
            "convention": "absolute",
            "dof_dims": {
                "follow_left_ee_cartesian_pos": 3,
                "follow_left_ee_rotation_6D": 6,
                "follow_left_gripper": 1,
            },
            "dof_order": [
                "follow_left_ee_cartesian_pos",
                "follow_left_ee_rotation_6D",
                "follow_left_gripper",
            ],
        },
        "normalization": {
            "mode": "quantile_min_max",
            "quantile_low": 0.001,
            "quantile_high": 0.999,
            "clamp": True,
            "stats_horizon": 32,
            "predict_action": {
                "space": "observation_relative",
                "q001": predict_min.tolist(),
                "q999": predict_max.tolist(),
            },
            "observation_state": {
                "space": "absolute",
                "q001": observation_min.tolist(),
                "q999": observation_max.tolist(),
            },
        },
    }
    payload = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    identity["content_fingerprint"] = hashlib.sha256(payload).hexdigest()
    return identity


def _has_nonzero_grad(module: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
        for parameter in module.parameters()
    )


def test_codec_forward_reports_finite_enabled_metrics_and_updates_all_paths() -> None:
    codec = XTokenizerCodec(config=_tiny_config())
    batch = _batch(requires_grad_target=True)
    initial_codebooks = codec.quantizer.codebooks.clone()
    quantizer_indices: list[Tensor] = []
    mam_inputs: list[Tensor] = []
    quantizer_hook = codec.quantizer.register_forward_hook(
        lambda _module, _args, output: quantizer_indices.append(
            output.indices.detach().clone()
        )
    )
    mam_hook = codec.mam_head.register_forward_pre_hook(
        lambda _module, args: mam_inputs.append(args[0].detach().clone())
    )

    loss, metrics = codec(batch)
    quantizer_hook.remove()
    mam_hook.remove()

    expected_metrics = {
        "action_l1_loss",
        "action_l1_loss_weighted",
        "rotation_geodesic_loss",
        "rotation_geodesic_loss_weighted",
        "commitment_loss",
        "commitment_loss_weighted",
        "dct_loss",
        "dct_loss_weighted",
        "smoothness_loss",
        "smoothness_loss_weighted",
        "mam_loss",
        "mam_loss_weighted",
        "alignment_global_loss",
        "alignment_local_loss",
        "alignment_loss",
        "alignment_loss_weighted",
        "prediction_loss",
        "prediction_loss_weighted",
        "codebook_perplexity",
        "codebook_perplexity_q0",
        "codebook_perplexity_q1",
        "codebook_perplexity_q2",
        "codebook_perplexity_q3",
        "total_loss",
    }
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert expected_metrics <= metrics.keys()
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert len(quantizer_indices) == 1
    assert len(mam_inputs) == 1
    assert torch.equal(mam_inputs[0], quantizer_indices[0][..., 0])
    assert not torch.equal(codec.quantizer.codebooks, initial_codebooks)

    loss.backward()

    assert _has_nonzero_grad(codec.encoder)
    assert _has_nonzero_grad(codec.decoder)
    assert _has_nonzero_grad(codec.mam_head)
    assert _has_nonzero_grad(codec.alignment_projection)
    assert _has_nonzero_grad(codec.prediction_head)
    assert batch["next_vl_features"].grad is None


def test_codec_initializes_configured_infonce_scale() -> None:
    codec = XTokenizerCodec(config=_tiny_config(infonce_scale_init=12.5))

    assert codec.logit_scale.exp().item() == pytest.approx(12.5)


def test_codec_normalizes_actions_and_anchor_before_encoding() -> None:
    codec = XTokenizerCodec(
        config=_tiny_config(
            mam_weight=0.0,
            alignment_weight=0.0,
            prediction_weight=0.0,
            dct_weight=0.0,
            smoothness_weight=0.0,
        )
    )
    codec.eval()
    relative_min = torch.full((10,), -2.0)
    relative_max = torch.full((10,), 2.0)
    observation_min = torch.full((10,), 10.0)
    observation_max = torch.full((10,), 20.0)
    codec.set_xtokenizer_normalization_stats(
        predict_stats={
            "action": {"min": relative_min, "max": relative_max},
        },
        observation_stats={
            "observation.state": {
                "min": observation_min,
                "max": observation_max,
            },
        },
        canonical_stats_identity=_canonical_stats_identity(
            predict_min=relative_min,
            predict_max=relative_max,
            observation_min=observation_min,
            observation_max=observation_max,
        ),
        stats_horizon=32,
    )
    batch = _batch()
    batch["action"][1, 24:] = 0.0
    batch["action"][0, 0, 0] = 100.0
    batch["observation.state"][0, 0] = 25.0
    seen: list[tuple[Tensor, Tensor]] = []
    hook = codec.encoder.register_forward_pre_hook(
        lambda _module, args: seen.append(
            (args[0].detach().clone(), args[1].detach().clone())
        )
    )

    codec(batch)
    hook.remove()

    assert len(seen) == 1
    relative_actions = to_observation_relative_actions(
        batch["action"],
        batch["observation.state"].unsqueeze(1),
        position_slice=codec.position_slice,
        rotation_slice=codec.rotation_slice,
    )
    expected_actions = (
        2.0 * (relative_actions - relative_min) / (relative_max - relative_min) - 1.0
    ).clamp(-1.0, 1.0)
    expected_actions = expected_actions.masked_fill(
        batch["action_is_pad"].unsqueeze(-1),
        0,
    )
    expected_anchor = (
        2.0
        * (batch["observation.state"] - observation_min)
        / (observation_max - observation_min)
        - 1.0
    ).clamp(-1.0, 1.0)
    assert torch.allclose(seen[0][0], expected_actions)
    assert torch.allclose(seen[0][1], expected_anchor.unsqueeze(1))
    assert seen[0][0].amin().item() >= -1.0
    assert seen[0][0].amax().item() <= 1.0
    assert seen[0][1].amin().item() >= -1.0
    assert seen[0][1].amax().item() <= 1.0
    assert codec.normalize_action is not codec.normalize_observation
    assert codec.unnormalize_action is not codec.unnormalize_observation
    checkpoint_stats, checkpoint_modes = codec._export_norm_stats_for_checkpoint()
    assert set(checkpoint_stats) == {"action", "observation.state"}
    assert checkpoint_modes == {
        "action": NormMode.MIN_MAX.value,
        "observation.state": NormMode.MIN_MAX.value,
    }


def test_codec_generic_mean_std_stats_preserve_base_anchor_fallback() -> None:
    codec = XTokenizerCodec(config=_tiny_config()).eval()
    mean = torch.linspace(-1.0, 1.0, 10)
    std = torch.linspace(0.5, 1.4, 10)
    codec.set_normalization_stats(
        {"action": {"mean": mean, "std": std}},
        modes={"action": NormMode.MEAN_STD},
    )
    batch = _batch()
    actions = batch["action"].nan_to_num()
    actions[0, 0, 0] = 20.0
    anchor = batch["observation.state"]
    seen: list[tuple[Tensor, Tensor]] = []
    hook = codec.encoder.register_forward_pre_hook(
        lambda _module, args: seen.append(
            (args[0].detach().clone(), args[1].detach().clone())
        )
    )

    codec.encode(actions, anchor=anchor, robot_type=batch["robot_type"])
    hook.remove()

    relative = to_observation_relative_actions(
        actions,
        anchor.unsqueeze(1),
        position_slices=codec.position_slices,
        rotation_slices=codec.rotation_slices,
    )
    assert len(seen) == 1
    torch.testing.assert_close(seen[0][0], (relative - mean) / std)
    torch.testing.assert_close(seen[0][1], ((anchor - mean) / std).unsqueeze(1))
    assert seen[0][0].amax().item() > 1.0
    assert codec.normalize_observation is None
    assert codec.unnormalize_observation is None
    checkpoint_stats, checkpoint_modes = codec._export_norm_stats_for_checkpoint()
    assert set(checkpoint_stats) == {"action"}
    assert checkpoint_modes == {"action": NormMode.MEAN_STD.value}
    assert set(codec._normalization_checkpoint_payload()) == {"stats", "modes"}


def test_codec_generic_two_key_stats_do_not_activate_paper_roles() -> None:
    codec = XTokenizerCodec(config=_tiny_config()).eval()
    action_stats = {
        "mean": torch.zeros(10),
        "std": torch.full((10,), 2.0),
    }
    observation_stats = {
        "mean": torch.full((10,), 100.0),
        "std": torch.full((10,), 0.25),
    }

    codec.set_normalization_stats(
        {
            "action": action_stats,
            "observation.state": observation_stats,
        },
        modes={
            "action": NormMode.MEAN_STD,
            "observation.state": NormMode.MEAN_STD,
        },
    )

    anchor = _batch()["observation.state"]
    normalized_anchor = codec._normalize_anchor(anchor.unsqueeze(1))
    checkpoint_stats, checkpoint_modes = codec._export_norm_stats_for_checkpoint()
    torch.testing.assert_close(normalized_anchor, anchor.unsqueeze(1) / 2.0)
    assert codec.normalize_observation is None
    assert codec.unnormalize_observation is None
    assert set(checkpoint_stats) == {"action"}
    assert checkpoint_modes == {"action": NormMode.MEAN_STD.value}
    assert set(codec._normalization_checkpoint_payload()) == {"stats", "modes"}


def test_codec_dual_normalization_payload_restores_across_dtype_transition(
    tmp_path: Path,
) -> None:
    predict_min = torch.full((10,), -2.0)
    predict_max = torch.full((10,), 2.0)
    observation_min = torch.full((10,), -20.0)
    observation_max = torch.full((10,), 20.0)
    source = XTokenizerCodec(config=_tiny_config()).eval()
    source.set_xtokenizer_normalization_stats(
        predict_stats={
            "action": {"min": predict_min, "max": predict_max},
        },
        observation_stats={
            "observation.state": {
                "min": observation_min,
                "max": observation_max,
            },
        },
        canonical_stats_identity=_canonical_stats_identity(
            predict_min=predict_min,
            predict_max=predict_max,
            observation_min=observation_min,
            observation_max=observation_max,
        ),
        stats_horizon=32,
    )
    source.set_io(
        PolicyIO(
            obs_features=[
                FeatureSpec(
                    key="observation.state",
                    type=FeatureType.STATE,
                    shape=(10,),
                    norm_mode=NormMode.MIN_MAX,
                )
            ],
            action_feature=FeatureSpec(
                key="action",
                type=FeatureType.ACTION,
                shape=(10,),
                norm_mode=NormMode.MIN_MAX,
            ),
            n_action_steps=32,
        )
    )
    source.save_pretrained(tmp_path)
    payload = torch.load(
        tmp_path / "norm_stats.pt",
        map_location="cpu",
        weights_only=True,
    )

    restored = XTokenizerCodec.from_pretrained(
        tmp_path,
        local_files_only=True,
    ).eval()
    restored.to(dtype=torch.float64)

    assert payload["normalization_kind"] == "xtokenizer_dual_role_v1"
    assert payload["stats_horizon"] == 32
    assert payload["canonical_stats_identity"] == _canonical_stats_identity(
        predict_min=predict_min,
        predict_max=predict_max,
        observation_min=observation_min,
        observation_max=observation_max,
    )
    assert payload["modes"] == {
        "action": NormMode.MIN_MAX.value,
        "observation.state": NormMode.MIN_MAX.value,
    }
    assert restored.normalize_action is not None
    assert restored.normalize_observation is not None
    torch.testing.assert_close(
        restored.normalize_action.stats["action"]["min"],
        predict_min,
    )
    torch.testing.assert_close(
        restored.normalize_observation.stats["observation.state"]["min"],
        observation_min,
    )
    batch = _batch()
    actions = batch["action"].nan_to_num().double()
    anchor = batch["observation.state"].double()
    robot_type = batch["robot_type"]
    seen: list[tuple[Tensor, Tensor]] = []
    encoder_hook = restored.encoder.register_forward_pre_hook(
        lambda _module, args: seen.append(
            (args[0].detach().clone(), args[1].detach().clone())
        )
    )
    indices = restored.encode(actions, anchor=anchor, robot_type=robot_type)
    encoder_hook.remove()

    normalized_relative = torch.zeros_like(actions)
    normalized_relative[..., 3:9] = torch.tensor(
        [0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
        dtype=torch.float64,
    )
    decoder_hook = restored.decoder.register_forward_hook(
        lambda _module, _args, _output: normalized_relative
    )
    decoded = restored.decode(indices, anchor=anchor, robot_type=robot_type)
    decoder_hook.remove()

    assert seen[0][0].dtype is torch.float64
    assert seen[0][1].dtype is torch.float64
    assert decoded.dtype is torch.float64
    assert torch.isfinite(decoded).all()
    torch.testing.assert_close(decoded[..., :3], anchor[:, None, :3].expand(-1, 32, -1))
    torch.testing.assert_close(
        decoded[..., 3:9],
        anchor[:, None, 3:9].expand(-1, 32, -1),
        atol=1e-6,
        rtol=1e-6,
    )


def test_codec_dual_normalization_payload_rejects_unknown_marker() -> None:
    codec = XTokenizerCodec(config=_tiny_config()).eval()

    with pytest.raises(ValueError, match="normalization_kind.*unknown_v2"):
        codec._load_normalization_checkpoint_payload(
            {
                "normalization_kind": "unknown_v2",
                "stats": {},
                "modes": {},
            }
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda payload: payload["stats"].__setitem__(
                "action", payload["stats"]["observation.state"]
            ),
            "predict_action",
        ),
        (
            lambda payload: payload["stats"]["action"]["min"].__setitem__(
                0, torch.tensor(-1.5)
            ),
            "predict_action",
        ),
        (lambda payload: payload.pop("canonical_stats_identity"), "identity"),
        (lambda payload: payload.update(stats_horizon=16), "stats_horizon"),
        (
            lambda payload: payload["stats"]["action"]["max"].__setitem__(
                0, torch.tensor(float("inf"))
            ),
            "finite",
        ),
    ],
)
def test_codec_rejects_unauthenticated_marked_checkpoint_normalization(
    mutation: Any,
    match: str,
) -> None:
    predict_min = torch.full((10,), -2.0)
    predict_max = torch.full((10,), 2.0)
    observation_min = torch.full((10,), -20.0)
    observation_max = torch.full((10,), 20.0)
    source = XTokenizerCodec(config=_tiny_config()).eval()
    source.set_xtokenizer_normalization_stats(
        predict_stats={"action": {"min": predict_min, "max": predict_max}},
        observation_stats={
            "observation.state": {
                "min": observation_min,
                "max": observation_max,
            }
        },
        canonical_stats_identity=_canonical_stats_identity(
            predict_min=predict_min,
            predict_max=predict_max,
            observation_min=observation_min,
            observation_max=observation_max,
        ),
        stats_horizon=32,
    )
    payload = copy.deepcopy(source._normalization_checkpoint_payload())
    mutation(payload)

    with pytest.raises(ValueError, match=match):
        XTokenizerCodec(config=_tiny_config())._load_normalization_checkpoint_payload(
            payload
        )


def test_codec_rejects_checkpoint_identity_replacing_active_artifact() -> None:
    active = XTokenizerCodec(config=_tiny_config()).eval()
    active_predict_min = torch.full((10,), -2.0)
    active_predict_max = torch.full((10,), 2.0)
    active_observation_min = torch.full((10,), -20.0)
    active_observation_max = torch.full((10,), 20.0)
    active.set_xtokenizer_normalization_stats(
        predict_stats={
            "action": {"min": active_predict_min, "max": active_predict_max}
        },
        observation_stats={
            "observation.state": {
                "min": active_observation_min,
                "max": active_observation_max,
            }
        },
        canonical_stats_identity=_canonical_stats_identity(
            predict_min=active_predict_min,
            predict_max=active_predict_max,
            observation_min=active_observation_min,
            observation_max=active_observation_max,
        ),
        stats_horizon=32,
    )

    replacement = XTokenizerCodec(config=_tiny_config()).eval()
    replacement_predict_min = torch.full((10,), -3.0)
    replacement_predict_max = torch.full((10,), 3.0)
    replacement_observation_min = torch.full((10,), -30.0)
    replacement_observation_max = torch.full((10,), 30.0)
    replacement.set_xtokenizer_normalization_stats(
        predict_stats={
            "action": {
                "min": replacement_predict_min,
                "max": replacement_predict_max,
            }
        },
        observation_stats={
            "observation.state": {
                "min": replacement_observation_min,
                "max": replacement_observation_max,
            }
        },
        canonical_stats_identity=_canonical_stats_identity(
            predict_min=replacement_predict_min,
            predict_max=replacement_predict_max,
            observation_min=replacement_observation_min,
            observation_max=replacement_observation_max,
        ),
        stats_horizon=32,
    )

    with pytest.raises(ValueError, match="active canonical statistics identity"):
        active._load_normalization_checkpoint_payload(
            copy.deepcopy(replacement._normalization_checkpoint_payload())
        )


def test_codec_rejects_unmarked_checkpoint_replacing_active_artifact() -> None:
    active = XTokenizerCodec(config=_tiny_config()).eval()
    predict_min = torch.full((10,), -2.0)
    predict_max = torch.full((10,), 2.0)
    observation_min = torch.full((10,), -20.0)
    observation_max = torch.full((10,), 20.0)
    active.set_xtokenizer_normalization_stats(
        predict_stats={"action": {"min": predict_min, "max": predict_max}},
        observation_stats={
            "observation.state": {
                "min": observation_min,
                "max": observation_max,
            }
        },
        canonical_stats_identity=_canonical_stats_identity(
            predict_min=predict_min,
            predict_max=predict_max,
            observation_min=observation_min,
            observation_max=observation_max,
        ),
        stats_horizon=32,
    )

    with pytest.raises(ValueError, match="active canonical statistics identity"):
        active._load_normalization_checkpoint_payload(
            {
                "stats": {},
                "modes": {},
            }
        )


def test_inference_module_does_not_eagerly_import_optional_cache_dependencies() -> None:
    code = """
import builtins
import sys
import types
from pathlib import Path

import praxis

package = types.ModuleType("praxis.codecs.latents.xtokenizer")
package.__path__ = [
    str(Path(praxis.__file__).parent / "codecs" / "latents" / "xtokenizer")
]
sys.modules[package.__name__] = package

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name in {
        "praxis.codecs.latents.xtokenizer.cache",
        "praxis.data.xtokenizer_vl",
    }:
        raise AssertionError(f"eager optional import: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import praxis.codecs.latents.xtokenizer.inference
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_inference_public_api_lazily_exports_committed_cache_helpers() -> None:
    code = """
import sys

from praxis.codecs.latents.xtokenizer import inference

cache_module_name = "praxis.codecs.latents.xtokenizer.cache"
cache_names = {
    "XTokenizerCacheAdapter",
    "load_xtokenizer_cache_adapter",
}
assert cache_names.issubset(inference.__all__)
assert cache_module_name not in sys.modules

adapter = inference.XTokenizerCacheAdapter
assert cache_module_name in sys.modules
cache_module = sys.modules[cache_module_name]
assert adapter is cache_module.XTokenizerCacheAdapter

namespace = {}
exec("from praxis.codecs.latents.xtokenizer.inference import *", namespace)
assert namespace["XTokenizerCacheAdapter"] is cache_module.XTokenizerCacheAdapter
assert (
    namespace["load_xtokenizer_cache_adapter"]
    is cache_module.load_xtokenizer_cache_adapter
)
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_codec_encodes_actions_relative_to_the_observation_anchor() -> None:
    codec = XTokenizerCodec(config=_tiny_config()).eval()
    anchor = torch.tensor([[1.0, 2.0, 3.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.25]])
    actions = torch.tensor(
        [[[1.5, 2.25, 2.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.75]]]
    ).expand(-1, 32, -1)
    seen: list[Tensor] = []
    hook = codec.encoder.register_forward_pre_hook(
        lambda _module, args: seen.append(args[0].detach().clone())
    )

    codec.encode(actions, anchor=anchor, robot_type=torch.zeros(1, dtype=torch.long))
    hook.remove()

    expected = actions.clone()
    expected[..., :3] = torch.tensor([0.5, 0.25, -1.0])
    expected[..., 3:9] = torch.tensor([0.0, -1.0, 0.0, 1.0, 0.0, 0.0])
    assert len(seen) == 1
    assert torch.allclose(seen[0], expected, atol=1e-6)


def test_codec_reanchors_relative_decoder_outputs_to_absolute_actions() -> None:
    codec = XTokenizerCodec(config=_tiny_config()).eval()
    predict_min = torch.full((10,), -2.0)
    predict_max = torch.full((10,), 2.0)
    observation_min = torch.full((10,), -10.0)
    observation_max = torch.full((10,), 10.0)
    codec.set_xtokenizer_normalization_stats(
        predict_stats={
            "action": {"min": predict_min, "max": predict_max},
        },
        observation_stats={
            "observation.state": {
                "min": observation_min,
                "max": observation_max,
            },
        },
        canonical_stats_identity=_canonical_stats_identity(
            predict_min=predict_min,
            predict_max=predict_max,
            observation_min=observation_min,
            observation_max=observation_max,
        ),
        stats_horizon=32,
    )
    anchor = torch.tensor([[1.0, 2.0, 3.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.25]])
    normalized_relative = torch.tensor(
        [[[0.25, 0.125, -0.5, 0.0, -0.5, 0.0, 0.5, 0.0, 0.0, 0.375]]]
    ).expand(-1, 32, -1)
    seen_anchor: list[Tensor] = []
    pre_hook = codec.decoder.register_forward_pre_hook(
        lambda _module, args: seen_anchor.append(args[1].detach().clone())
    )
    hook = codec.decoder.register_forward_hook(
        lambda _module, _args, _output: normalized_relative
    )
    indices = torch.zeros(
        1,
        codec.latent_horizon,
        codec.num_quantizers,
        dtype=torch.long,
    )

    actions = codec.decode(
        indices,
        anchor=anchor,
        robot_type=torch.zeros(1, dtype=torch.long),
    )
    pre_hook.remove()
    hook.remove()

    expected_normalized_anchor = (
        2.0 * (anchor - observation_min) / (observation_max - observation_min) - 1.0
    )
    expected = normalized_relative * 2.0
    expected[..., :3] = torch.tensor([1.5, 2.25, 2.0])
    expected[..., 3:9] = torch.tensor([-1.0, 0.0, 0.0, 0.0, -1.0, 0.0])
    assert len(seen_anchor) == 1
    assert torch.allclose(
        seen_anchor[0],
        expected_normalized_anchor.unsqueeze(1),
    )
    assert torch.allclose(actions, expected, atol=1e-6)


def test_codec_encode_zeros_invalid_dofs_before_the_encoder() -> None:
    codec = XTokenizerCodec(config=_tiny_config()).eval()
    batch = _batch()
    batch["action"][:, :, -1] = 1.0e6
    dof_mask = torch.ones_like(batch["action"], dtype=torch.bool)
    dof_mask[:, :, -1] = False
    seen: list[Tensor] = []
    hook = codec.encoder.register_forward_pre_hook(
        lambda _module, args: seen.append(args[0].detach().clone())
    )

    codec.encode(
        batch["action"].nan_to_num(),
        anchor=batch["observation.state"],
        robot_type=batch["robot_type"],
        action_is_pad=batch["action_is_pad"],
        action_dof_mask=dof_mask,
    )
    hook.remove()

    assert len(seen) == 1
    assert torch.count_nonzero(seen[0][..., -1]) == 0


def test_codec_decode_zeros_invalid_dofs_after_reanchoring() -> None:
    codec = XTokenizerCodec(config=_tiny_config()).eval()
    batch = _batch()
    indices = torch.zeros(
        batch["action"].shape[0],
        codec.latent_horizon,
        codec.num_quantizers,
        dtype=torch.long,
    )
    dof_mask = torch.ones_like(batch["action"], dtype=torch.bool)
    dof_mask[:, :, -1] = False
    with torch.no_grad():
        codec.decoder.output_projection.bias[-1] = 100.0

    decoded = codec.decode(
        indices,
        anchor=batch["observation.state"],
        robot_type=batch["robot_type"],
        action_is_pad=batch["action_is_pad"],
        action_dof_mask=dof_mask,
    )

    assert torch.count_nonzero(decoded[..., -1]) == 0


def test_codec_rotation_loss_uses_selected_min_max_normalization_mode() -> None:
    codec = XTokenizerCodec(
        config=_tiny_config(
            reconstruction_weight=0.0,
            commitment_weight=0.0,
            dct_weight=0.0,
            smoothness_weight=0.0,
            mam_weight=0.0,
            alignment_weight=0.0,
            prediction_weight=0.0,
        )
    )
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    z_90 = torch.tensor([0.0, -1.0, 0.0, 1.0, 0.0, 0.0])
    mean = torch.zeros(10)
    mean[3:9] = identity
    stats = {
        "mean": mean,
        "std": torch.full((10,), 0.1),
        "min": -torch.ones(10),
        "max": torch.ones(10),
    }
    codec.set_normalization_stats(
        {"action": stats, "observation.state": stats},
        modes={
            "action": NormMode.MIN_MAX,
            "observation.state": NormMode.MIN_MAX,
        },
    )
    with torch.no_grad():
        codec.decoder.output_projection.weight.zero_()
        codec.decoder.output_projection.bias.zero_()
        codec.decoder.output_projection.bias[3:9] = z_90

    _, metrics = codec(_batch())

    assert metrics["rotation_geodesic_loss"] == pytest.approx(0.5, abs=2e-4)


def test_multi_rotation_loss_weights_only_valid_named_rotation_groups() -> None:
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    z_90 = torch.tensor([0.0, -1.0, 0.0, 1.0, 0.0, 0.0])
    target = torch.zeros(1, 1, 12)
    reconstruction = torch.zeros_like(target)
    target[..., :6] = identity
    target[..., 6:12] = identity
    reconstruction[..., :6] = torch.nan
    reconstruction[..., 6:12] = z_90
    dof_valid = torch.ones_like(target, dtype=torch.bool)
    dof_valid[..., :6] = False

    loss = _multi_rotation_geodesic_loss(
        reconstruction,
        target,
        {},
        dof_valid,
        normalization_mode=None,
        rotation_slices=(slice(0, 6), slice(6, 12)),
    )

    assert loss.item() == pytest.approx(0.5, abs=2e-4)


def test_smoothness_ignores_rotation_only_changes_on_valid_adjacent_rows() -> None:
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    z_90 = torch.tensor([0.0, -1.0, 0.0, 1.0, 0.0, 0.0])
    reconstruction = torch.zeros(1, 4, 10)
    reconstruction[0, 0, 3:9] = identity
    reconstruction[0, 1, 3:9] = z_90
    reconstruction[0, 2, 0] = 100.0
    reconstruction[0, 3, 0] = -100.0
    reconstruction.requires_grad_()
    target = torch.zeros_like(reconstruction)
    target[..., 3:9] = identity
    valid_mask = torch.tensor([[True, True, False, True]])

    loss = _masked_smoothness_l1(
        reconstruction,
        target,
        valid_mask,
        scalar_channel_indices=(0, 1, 2, 9),
    )

    assert loss.item() == 0.0
    loss.backward()
    assert reconstruction.grad is not None
    assert torch.isfinite(reconstruction.grad).all()


def test_smoothness_excludes_reordered_named_rotation_channels_only() -> None:
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    z_90 = torch.tensor([0.0, -1.0, 0.0, 1.0, 0.0, 0.0])
    reconstruction = torch.zeros(1, 2, 10)
    target = torch.zeros_like(reconstruction)
    reconstruction[..., 4:10] = torch.stack((identity, z_90))
    target[..., 4:10] = identity
    valid_mask = torch.ones(1, 2, dtype=torch.bool)

    rotation_only = _masked_smoothness_l1(
        reconstruction,
        target,
        valid_mask,
        scalar_channel_indices=(0, 1, 2, 3),
    )
    reconstruction[0, 1, 0] = 2.0
    scalar_change = _masked_smoothness_l1(
        reconstruction,
        target,
        valid_mask,
        scalar_channel_indices=(0, 1, 2, 3),
    )

    assert rotation_only.item() == 0.0
    assert scalar_change.item() == pytest.approx(0.5)


def test_dct_loss_applies_per_dof_mask() -> None:
    reconstruction = torch.tensor([[[1.0, 1000.0], [3.0, 2000.0]]])
    target = torch.zeros_like(reconstruction)
    temporal_valid = torch.ones(1, 2, dtype=torch.bool)
    dof_valid = torch.tensor([[[True, False], [True, False]]])

    masked = _masked_dct_l1(
        reconstruction,
        target,
        temporal_valid,
        dof_valid_mask=dof_valid,
    )
    first_channel_only = _masked_dct_l1(
        reconstruction[..., :1],
        target[..., :1],
        temporal_valid,
    )

    torch.testing.assert_close(masked, first_channel_only)


def test_smoothness_loss_applies_per_dof_mask() -> None:
    reconstruction = torch.tensor([[[0.0, 0.0], [2.0, 1000.0]]])
    target = torch.zeros_like(reconstruction)
    temporal_valid = torch.ones(1, 2, dtype=torch.bool)
    dof_valid = torch.tensor([[[True, False], [True, False]]])

    loss = _masked_smoothness_l1(
        reconstruction,
        target,
        temporal_valid,
        scalar_channel_indices=(0, 1),
        dof_valid_mask=dof_valid,
    )

    assert loss.item() == pytest.approx(2.0)


def test_codec_mam_warmup_uses_training_context_boundary() -> None:
    codec = XTokenizerCodec(
        config=_tiny_config(
            mam_warmup_steps=2,
            alignment_weight=0.0,
            prediction_weight=0.0,
            dct_weight=0.0,
            smoothness_weight=0.0,
        )
    )
    batch = _batch()

    codec.set_training_context(TrainingContext(global_step=1))
    _, warmup_metrics = codec(batch)
    assert "mam_loss" not in warmup_metrics

    codec.set_training_context(TrainingContext(global_step=2))
    _, active_metrics = codec(batch)
    assert "mam_loss" in active_metrics
    assert torch.isfinite(torch.tensor(active_metrics["mam_loss"]))


def test_codec_mam_epoch_warmup_starts_at_epoch_zero() -> None:
    codec = XTokenizerCodec(
        config=_tiny_config(
            mam_warmup_epochs=2,
            alignment_weight=0.0,
            prediction_weight=0.0,
            dct_weight=0.0,
            smoothness_weight=0.0,
        )
    )
    batch = _batch()

    _, initial_metrics = codec(batch)
    assert "mam_loss" not in initial_metrics

    codec.set_training_epoch(2)
    _, active_metrics = codec(batch)
    assert "mam_loss" in active_metrics


def test_codec_disabled_auxiliary_heads_do_not_require_sidecar_fields() -> None:
    codec = XTokenizerCodec(
        config=_tiny_config(
            mam_weight=0.0,
            alignment_weight=0.0,
            prediction_weight=0.0,
            dct_weight=0.0,
            smoothness_weight=0.0,
        )
    )
    batch = _batch()
    for key in ("vl_features", "vl_feature_is_pad", "next_vl_features"):
        del batch[key]

    loss, metrics = codec(batch)

    assert torch.isfinite(loss)
    assert "mam_loss" not in metrics
    assert "alignment_loss" not in metrics
    assert "prediction_loss" not in metrics


def test_codec_encode_decode_and_minimal_inference_export(tmp_path: Path) -> None:
    codec = XTokenizerCodec(config=_tiny_config())
    codec.train()
    codec(_batch())
    codec.eval()
    batch = _batch()
    valid_mask = ~batch["action_is_pad"]

    indices = codec.encode(
        batch["action"],
        anchor=batch["observation.state"],
        robot_type=batch["robot_type"],
        action_mask=valid_mask,
    )
    decoded = codec.decode(
        indices,
        anchor=batch["observation.state"],
        robot_type=batch["robot_type"],
        action_mask=valid_mask,
    )

    assert indices.shape == (2, 8, 4)
    assert indices.dtype == torch.long
    assert decoded.shape == (2, 32, 10)
    codec.export_inference_artifact(tmp_path)

    payload = torch.load(
        tmp_path / "xtokenizer_inference.pt",
        map_location="cpu",
        weights_only=True,
    )
    state_keys = set(payload["state_dict"])
    assert any(key.startswith("encoder.") for key in state_keys)
    assert any(key.startswith("quantizer.") for key in state_keys)
    assert any(key.startswith("decoder.") for key in state_keys)
    assert not any("mam" in key for key in state_keys)
    assert not any("alignment" in key for key in state_keys)
    assert not any("prediction" in key for key in state_keys)


def test_codec_refuses_to_export_uninitialized_kmeans_codebooks(
    tmp_path: Path,
) -> None:
    codec = XTokenizerCodec(config=_tiny_config(kmeans_init=True)).eval()

    with pytest.raises(RuntimeError, match="initialized|k-means"):
        codec.export_inference_artifact(tmp_path)


def test_codec_reconstruct_with_info_returns_encoded_indices() -> None:
    codec = XTokenizerCodec(config=_tiny_config())
    codec.eval()
    batch = _batch()

    reconstruction, info = codec.reconstruct_with_info(
        batch["action"],
        batch=batch,
        action_key="action",
    )

    assert reconstruction.shape == batch["action"].shape
    assert info["xtokenizer_indices"].shape == (2, 8, 4)
    assert info["xtokenizer_indices"].dtype == torch.long


def test_codec_requires_enabled_sidecar_inputs() -> None:
    codec = XTokenizerCodec(config=_tiny_config())
    batch = _batch()
    del batch["vl_features"]

    with pytest.raises(KeyError, match="vl_features"):
        codec(batch)
