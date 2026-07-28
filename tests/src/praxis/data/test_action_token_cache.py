# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from praxis.data.action_token_cache import (
    ActionTokenCache,
    CachedActionTokenDataset,
    sha256_float32_matrix,
)
from praxis.data.lerobot_adapter import write_lerobot_cache_provenance
from praxis.data.xtokenizer_vl import vl_sidecar_generation_fingerprint


class _Dataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"action": torch.tensor([float(index)])}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_ints(values: list[int]) -> str:
    payload = np.ascontiguousarray(values, dtype=np.int64).tobytes()
    return hashlib.sha256(payload).hexdigest()


def _canonical_stats_identity() -> dict[str, Any]:
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
                "q001": [-2.0] * 10,
                "q999": [2.0] * 10,
            },
            "observation_state": {
                "space": "absolute",
                "q001": [-20.0] * 10,
                "q999": [20.0] * 10,
            },
        },
    }
    encoded = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    identity["content_fingerprint"] = hashlib.sha256(encoded).hexdigest()
    return identity


def _vl_sidecar_identity() -> dict[str, Any]:
    identity: dict[str, Any] = {
        "format_version": 3,
        "dataset_repo_id": "org/bridge",
        "dataset_revision": "a" * 40,
        "extractor_revision": "b" * 40,
        "model_id": "org/vl-model",
        "model_revision": "c" * 40,
        "layer": -2,
        "stride": 4,
        "feature_dim": 16,
        "episode_indices": [0],
        "episode_lengths": [33],
        "image_keys": ["observation.images.image"],
        "global_instruction": "complete the task",
        "local_subtask": "reach",
        "synthetic": False,
        "features_sha256": "d" * 64,
        "next_features_sha256": "e" * 64,
        "valid_sha256": "f" * 64,
    }
    identity["generation_fingerprint"] = vl_sidecar_generation_fingerprint(identity)
    return identity


def _write_cache(tmp_path: Path, *, native: bool = False) -> Path:
    root = tmp_path / ("train" if native else "legacy")
    root.mkdir()
    tokens_path = root / "tokens.npy"
    num_samples = 33 if native else 2
    np.save(
        tokens_path,
        np.arange(
            num_samples * 32,
            dtype=np.uint16 if native else np.int64,
        ).reshape(num_samples, 8, 4),
    )
    manifest: dict[str, Any] = {
        "format_version": 2,
        "tokenizer": "x-tokenizer",
        "action_semantics": {
            "public_action": "absolute",
            "tokenized_action": "observation_relative",
            "observation_anchor": "absolute",
            "relative_reference": "single_pre_chunk_observation",
        },
        "robot_type_id": 7,
        "codebook_size": 2048,
        "num_quantizers": 4,
        "compression_ratio": 4,
        "action_horizon": 32,
        "token_order": "time_major",
        "num_samples": num_samples,
        "num_valid_samples": 2,
        "tokens_file": "tokens.npy",
        "tokens_sha256": _sha256_file(tokens_path),
        "split": "train",
        "split_seed": 42,
        "val_ratio": 0.05,
        "episode_ids_sha256": _sha256_ints([0]),
        "global_indices_sha256": _sha256_ints(list(range(num_samples))),
        "first_global_index": 0,
        "last_global_index": num_samples - 1,
        "active_dofs": [
            "follow_left_ee_cartesian_pos",
            "follow_left_ee_rotation_6D",
            "follow_left_gripper",
        ],
        "inactive_normalized_fill": 0.0,
        "tail_token_fill": 0,
        "gripper_clip": [0.0, 1.0],
        "max_valid_windows": None,
    }
    if native:
        identity = _canonical_stats_identity()
        normalization = {
            "kind": "xtokenizer_dual_role_v1",
            "roles": {
                "predict_action": {
                    "key": "action",
                    "space": "observation_relative",
                },
                "observation_state": {
                    "key": "observation.state",
                    "space": "absolute",
                },
            },
            "stats": {
                "action": {"min": [-2.0] * 10, "max": [2.0] * 10},
                "observation.state": {
                    "min": [-20.0] * 10,
                    "max": [20.0] * 10,
                },
            },
            "modes": {"action": "min_max", "observation.state": "min_max"},
            "identity": {
                "format_version": 3,
                "content_fingerprint": identity["content_fingerprint"],
                "stats_horizon": 32,
            },
        }
        normalization_path = tmp_path / "normalization.json"
        normalization_path.write_text(json.dumps(normalization))
        manifest.update(
            {
                "normalization_source": "inference_artifact",
                "content_fingerprint": identity["content_fingerprint"],
                "stats_horizon": 32,
                "canonical_stats_identity": identity,
                "dataset_repo_id": "org/bridge",
                "dataset_revision": "a" * 40,
                "vl_sidecar_identity": _vl_sidecar_identity(),
                "artifact_sha256": "b" * 64,
                "source_checkpoint_sha256": "c" * 64,
                "normalization_file": "../normalization.json",
                "normalization_sha256": _sha256_file(normalization_path),
                "source_action_sha256": sha256_float32_matrix(
                    np.arange(33 * 7, dtype=np.float32).reshape(33, 7)
                ),
                "source_observation_state_sha256": sha256_float32_matrix(
                    np.arange(33 * 7, dtype=np.float32).reshape(33, 7) / 10.0
                ),
            }
        )
    else:
        manifest.update(
            {
                "source_revision": "9733990b16a07d51fb8b5d305ad8c16abb57aef2",
                "hf_revision": "bb5a47a36079d085d55c303fdf73223bd3dedd1d",
                "checkpoint_sha256": "a" * 64,
                "dataset_repo_id": "chaoqi-liu/bridge_single_view",
                "statistics_sha256": "d" * 64,
                "statistics_key": "WidowX",
            }
        )
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


class _NativeDataset(_Dataset):
    def __init__(
        self,
        *,
        repo_id: str = "org/bridge",
        revision: str = "a" * 40,
        indices: list[int] | None = None,
        episode_indices: list[int] | None = None,
        actions: np.ndarray | None = None,
        states: np.ndarray | None = None,
        action_horizon: int = 32,
        root: Path | None = None,
    ) -> None:
        self.io = SimpleNamespace(n_action_steps=action_horizon)
        self.lerobot_dataset = SimpleNamespace(
            repo_id=repo_id,
            revision=revision,
            root=root,
            hf_dataset={
                "index": list(range(33)) if indices is None else indices,
                "episode_index": (
                    [0] * 33 if episode_indices is None else episode_indices
                ),
                "action": (
                    np.arange(33 * 7, dtype=np.float32).reshape(33, 7)
                    if actions is None
                    else actions
                ),
                "observation.state": (
                    np.arange(33 * 7, dtype=np.float32).reshape(33, 7) / 10.0
                    if states is None
                    else states
                ),
            },
        )

    def __len__(self) -> int:
        return len(self.lerobot_dataset.hf_dataset["index"])


def _native_dataset(tmp_path: Path, **kwargs: Any) -> _NativeDataset:
    root = tmp_path / "dataset"
    write_lerobot_cache_provenance(
        root=root,
        repo_id="org/bridge",
        revision="a" * 40,
    )
    return _NativeDataset(root=root, **kwargs)


def _wrap_native(
    dataset: _NativeDataset,
    cache: ActionTokenCache,
    *,
    expected_split: str = "train",
    expected_split_seed: int = 42,
    expected_val_ratio: float = 0.05,
    expected_artifact_sha256: str = "b" * 64,
) -> CachedActionTokenDataset:
    return CachedActionTokenDataset(
        dataset,
        cache,
        expected_split=expected_split,
        expected_split_seed=expected_split_seed,
        expected_val_ratio=expected_val_ratio,
        expected_artifact_sha256=expected_artifact_sha256,
    )


def test_cache_preserves_structured_rvq_tokens_and_flattens_time_major(
    tmp_path,
) -> None:
    root = _write_cache(tmp_path)

    cache = ActionTokenCache.open(root)

    assert cache.tokens.shape == (2, 8, 4)
    assert cache.token_ids(1).tolist() == list(range(32, 64))
    assert not hasattr(cache, "block_schedule")


def test_cache_rejects_manifest_shape_mismatch(tmp_path) -> None:
    root = _write_cache(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["num_quantizers"] = 3
    (root / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="token shape"):
        ActionTokenCache.open(root)


def test_cache_rejects_legacy_absolute_trained_tokens(tmp_path) -> None:
    root = _write_cache(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["format_version"] = 1
    manifest.pop("action_semantics")
    (root / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="format_version"):
        ActionTokenCache.open(root)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda manifest: manifest.pop("content_fingerprint"), "content_fingerprint"),
        (
            lambda manifest: manifest.update(content_fingerprint="not-a-digest"),
            "content_fingerprint",
        ),
        (lambda manifest: manifest.pop("stats_horizon"), "stats_horizon"),
        (lambda manifest: manifest.update(stats_horizon=16), "stats_horizon"),
        (
            lambda manifest: manifest["canonical_stats_identity"].update(
                format_version=1
            ),
            "format_version",
        ),
    ],
)
def test_native_cache_requires_v3_statistics_identity(
    tmp_path,
    mutation,
    match: str,
) -> None:
    root = _write_cache(tmp_path, native=True)
    manifest = json.loads((root / "manifest.json").read_text())
    mutation(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=match):
        ActionTokenCache.open(root)


def test_native_cache_accepts_matching_v3_statistics_identity(tmp_path) -> None:
    root = _write_cache(tmp_path, native=True)

    cache = ActionTokenCache.open(root)

    assert (
        cache.manifest["content_fingerprint"]
        == (_canonical_stats_identity()["content_fingerprint"])
    )
    assert cache.manifest["stats_horizon"] == cache.manifest["action_horizon"]


def test_dataset_attaches_cached_targets_without_mutating_source(tmp_path) -> None:
    root = _write_cache(tmp_path)
    source = _Dataset()
    dataset = CachedActionTokenDataset(source, ActionTokenCache.open(root))

    sample = dataset[1]

    assert "action_token_ids" not in source[1]
    assert sample["action_token_ids"].tolist() == list(range(32, 64))
    assert sample["action_token_length"].item() == 32


def test_cached_dataset_dataloader_keeps_cache_wrapper(tmp_path) -> None:
    root = _write_cache(tmp_path)
    dataset = CachedActionTokenDataset(_Dataset(), ActionTokenCache.open(root))

    loader = dataset.make_dataloader(
        batch_size=2,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
        use_episode_aware_sampler=False,
        drop_last=False,
    )
    batch = next(iter(loader))

    assert loader.dataset is dataset
    assert batch["action_token_ids"].shape == (2, 32)
    assert batch["action_token_length"].tolist() == [32, 32]


def test_cache_rejects_corrupted_token_bytes(tmp_path: Path) -> None:
    root = _write_cache(tmp_path, native=True)
    with (root / "tokens.npy").open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(ValueError, match="tokens_sha256|token.*SHA256"):
        ActionTokenCache.open(root)


@pytest.mark.parametrize(
    "tokens_file",
    ["../tokens.npy", "/tmp/tokens.npy"],
)
def test_cache_rejects_unconfined_token_paths(
    tmp_path: Path,
    tokens_file: str,
) -> None:
    root = _write_cache(tmp_path, native=True)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["tokens_file"] = tokens_file
    (root / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="tokens_file.*relative|tokens_file.*confined"):
        ActionTokenCache.open(root)


def test_native_cache_rejects_unexpected_manifest_metadata(tmp_path: Path) -> None:
    root = _write_cache(tmp_path, native=True)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["untrusted_note"] = "accepted by loose readers"
    (root / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="manifest.*unexpected|unexpected.*manifest"):
        ActionTokenCache.open(root)


def test_native_cache_cannot_downgrade_by_deleting_kind_marker(
    tmp_path: Path,
) -> None:
    root = _write_cache(tmp_path, native=True)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest.pop("normalization_source")
    (root / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="normalization_source|manifest.*missing"):
        ActionTokenCache.open(root)


def test_native_cache_rejects_smoke_capped_rows(tmp_path: Path) -> None:
    root = _write_cache(tmp_path, native=True)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["max_valid_windows"] = 1
    (root / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="must not be capped"):
        ActionTokenCache.open(root)


def test_native_cache_recomputes_canonical_statistics_fingerprint(
    tmp_path: Path,
) -> None:
    root = _write_cache(tmp_path, native=True)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["canonical_stats_identity"]["normalization"]["predict_action"]["q001"][
        0
    ] = -1.5
    (root / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="fingerprint"):
        ActionTokenCache.open(root)


def test_native_cache_authenticates_normalization_contents(tmp_path: Path) -> None:
    root = _write_cache(tmp_path, native=True)
    (tmp_path / "normalization.json").write_text("{}\n")

    with pytest.raises(ValueError, match="normalization.*SHA256"):
        ActionTokenCache.open(root)


def test_native_cache_requires_valid_inference_artifact_digest(tmp_path: Path) -> None:
    root = _write_cache(tmp_path, native=True)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["artifact_sha256"] = "not-a-digest"
    (root / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="artifact_sha256"):
        ActionTokenCache.open(root)


@pytest.mark.parametrize(
    ("dataset_kwargs", "match"),
    [
        ({"repo_id": "other/bridge"}, "dataset.*repository"),
        (
            {"indices": [1, 0, *range(2, 33)]},
            "global.*order|global.*indices",
        ),
        ({"episode_indices": [1] * 33}, "episode.*identity"),
        ({"action_horizon": 16}, "horizon"),
    ],
)
def test_native_cache_binds_to_actual_consumer_dataset(
    tmp_path: Path,
    dataset_kwargs: dict[str, Any],
    match: str,
) -> None:
    root = _write_cache(tmp_path, native=True)
    dataset = _native_dataset(tmp_path, **dataset_kwargs)

    with pytest.raises(ValueError, match=match):
        _wrap_native(dataset, ActionTokenCache.open(root))


def test_native_cache_rejects_live_source_action_mutation(tmp_path: Path) -> None:
    root = _write_cache(tmp_path, native=True)
    actions = np.arange(33 * 7, dtype=np.float32).reshape(33, 7)
    actions[0, 0] += 1.0
    cache = ActionTokenCache.open(root)

    with pytest.raises(ValueError, match="source action"):
        _wrap_native(
            _native_dataset(tmp_path, actions=actions),
            cache,
        )
    with pytest.raises(RuntimeError, match="native.*binding|binding.*native"):
        cache.token_ids(0)


def test_native_cache_rejects_live_source_observation_mutation(
    tmp_path: Path,
) -> None:
    root = _write_cache(tmp_path, native=True)
    states = np.arange(33 * 7, dtype=np.float32).reshape(33, 7) / 10.0
    states[0, 0] += 1.0

    with pytest.raises(ValueError, match="source observation"):
        _wrap_native(
            _native_dataset(tmp_path, states=states),
            ActionTokenCache.open(root),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_action_sha256", None),
        ("source_observation_state_sha256", "not-a-digest"),
    ],
)
def test_native_cache_requires_source_value_digests(
    tmp_path: Path,
    field: str,
    value: str | None,
) -> None:
    root = _write_cache(tmp_path, native=True)
    manifest = json.loads((root / "manifest.json").read_text())
    if value is None:
        manifest.pop(field)
    else:
        manifest[field] = value
    (root / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=field):
        ActionTokenCache.open(root)


def test_native_cache_rejects_split_name_tampering(tmp_path: Path) -> None:
    root = _write_cache(tmp_path, native=True)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["split"] = "val"
    (root / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="split"):
        ActionTokenCache.open(root)


def test_native_cache_exposes_rows_after_complete_identity_binding(
    tmp_path: Path,
) -> None:
    root = _write_cache(tmp_path, native=True)

    dataset = _wrap_native(
        _native_dataset(tmp_path),
        ActionTokenCache.open(root),
    )

    assert dataset[1]["action_token_ids"].tolist() == list(range(32, 64))


def test_native_cache_never_exposes_zero_filled_episode_tail_rows(
    tmp_path: Path,
) -> None:
    root = _write_cache(tmp_path, native=True)
    dataset = _wrap_native(
        _native_dataset(tmp_path),
        ActionTokenCache.open(root),
    )

    with pytest.raises(IndexError, match="complete action window"):
        dataset[2]


def test_native_cache_always_uses_immutable_dataset_provenance(tmp_path: Path) -> None:
    root = _write_cache(tmp_path, native=True)
    dataset_root = tmp_path / "dataset"
    write_lerobot_cache_provenance(
        root=dataset_root,
        repo_id="org/bridge",
        revision="b" * 40,
    )
    dataset = _NativeDataset(root=dataset_root, revision="a" * 40)

    with pytest.raises(ValueError, match="dataset.*revision"):
        _wrap_native(dataset, ActionTokenCache.open(root))


@pytest.mark.parametrize(
    ("binding", "match"),
    [
        ({"expected_split": "val"}, "split"),
        ({"expected_split_seed": 7}, "split_seed"),
        ({"expected_val_ratio": 0.25}, "val_ratio"),
    ],
)
def test_native_cache_binds_to_callers_logical_split(
    tmp_path: Path,
    binding: dict[str, Any],
    match: str,
) -> None:
    root = _write_cache(tmp_path, native=True)

    with pytest.raises(ValueError, match=match):
        _wrap_native(
            _native_dataset(tmp_path),
            ActionTokenCache.open(root),
            **binding,
        )


def test_native_cache_validates_vl_sidecar_identity(tmp_path: Path) -> None:
    root = _write_cache(tmp_path, native=True)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["vl_sidecar_identity"]["feature_dim"] = 8
    (root / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="VL sidecar.*fingerprint"):
        ActionTokenCache.open(root)


def test_native_cache_requires_writer_token_dtype(tmp_path: Path) -> None:
    root = _write_cache(tmp_path, native=True)
    tokens_path = root / "tokens.npy"
    values = np.load(tokens_path).astype(np.int64)
    np.save(tokens_path, values)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["tokens_sha256"] = _sha256_file(tokens_path)
    (root / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="dtype.*uint16"):
        ActionTokenCache.open(root)


def test_native_cache_seals_public_tokens_before_consumer_binding(
    tmp_path: Path,
) -> None:
    cache = ActionTokenCache.open(_write_cache(tmp_path, native=True))

    with pytest.raises(RuntimeError, match="native.*binding|binding.*native"):
        _ = cache.tokens
    with pytest.raises(RuntimeError, match="native.*binding|binding.*native"):
        cache.token_ids(0)


def test_failed_native_binding_never_exposes_token_rows(tmp_path: Path) -> None:
    cache = ActionTokenCache.open(_write_cache(tmp_path, native=True))

    with pytest.raises(ValueError, match="dataset.*repository"):
        _wrap_native(
            _native_dataset(tmp_path, repo_id="other/bridge"),
            cache,
        )
    with pytest.raises(RuntimeError, match="native.*binding|binding.*native"):
        cache.token_ids(0)


def test_successful_native_binding_reads_only_complete_rows(tmp_path: Path) -> None:
    cache = ActionTokenCache.open(_write_cache(tmp_path, native=True))
    dataset = CachedActionTokenDataset(
        _native_dataset(tmp_path),
        cache,
        expected_split="train",
        expected_split_seed=42,
        expected_val_ratio=0.05,
        expected_artifact_sha256="b" * 64,
    )

    assert dataset[1]["action_token_ids"].tolist() == list(range(32, 64))
    with pytest.raises(IndexError, match="complete action window"):
        dataset[2]
    with pytest.raises(RuntimeError, match="native.*binding|binding.*native"):
        cache.token_ids(1)


def test_native_binding_requires_expected_artifact_digest(tmp_path: Path) -> None:
    cache = ActionTokenCache.open(_write_cache(tmp_path, native=True))

    with pytest.raises(ValueError, match="expected_artifact_sha256"):
        CachedActionTokenDataset(
            _native_dataset(tmp_path),
            cache,
            expected_split="train",
            expected_split_seed=42,
            expected_val_ratio=0.05,
        )


def test_native_binding_rejects_wrong_expected_artifact_digest(
    tmp_path: Path,
) -> None:
    cache = ActionTokenCache.open(_write_cache(tmp_path, native=True))

    with pytest.raises(ValueError, match="artifact_sha256.*does not match"):
        CachedActionTokenDataset(
            _native_dataset(tmp_path),
            cache,
            expected_split="train",
            expected_split_seed=42,
            expected_val_ratio=0.05,
            expected_artifact_sha256="a" * 64,
        )
    with pytest.raises(RuntimeError, match="native.*binding|binding.*native"):
        cache.token_ids(0)


def test_native_binding_accepts_trusted_artifact_digest(tmp_path: Path) -> None:
    cache = ActionTokenCache.open(_write_cache(tmp_path, native=True))

    dataset = CachedActionTokenDataset(
        _native_dataset(tmp_path),
        cache,
        expected_split="train",
        expected_split_seed=42,
        expected_val_ratio=0.05,
        expected_artifact_sha256="b" * 64,
    )

    assert dataset[0]["action_token_ids"].tolist() == list(range(32))
