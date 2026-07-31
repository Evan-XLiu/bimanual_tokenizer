#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Compute Bridge WidowX statistics and offline X-Tokenizer caches."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pyarrow.dataset as pads
import torch

from praxis.codecs.latents.xtokenizer.inference import (
    XTokenizerCacheAdapter,
    load_xtokenizer_cache_adapter,
)
from praxis.data.action_token_cache import ActionTokenCache, sha256_float32_matrix
from praxis.data.bridge_xtokenizer import (
    BridgeXTokenizerEpisode,
    build_bridge_xtokenizer_episode,
    resolve_bridge_xtokenizer_layout,
)
from praxis.data.lerobot_adapter import (
    _missing_lerobot_metadata_paths,
    read_lerobot_cache_provenance,
)
from praxis.training.core.checkpoint.staging import remove_path
from praxis.training.core.episode_splits import split_train_val_episodes

SOURCE_REVISION = "9733990b16a07d51fb8b5d305ad8c16abb57aef2"
CHECKPOINT_SHA256 = "aa14c832e68df679ba60665f0eb82fd4b4efac48b0c2b6b065d21fa663beec89"
HF_REVISION = "bb5a47a36079d085d55c303fdf73223bd3dedd1d"
ROBOT_TYPE = "WidowX"
HORIZON = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--stats-batch-size", type=int, default=32768)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--max-valid-windows",
        type=int,
        default=None,
        help="Smoke-only cap per split. Manifests record the cap.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.stats_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    output_root = args.output_root
    if output_root.exists():
        raise FileExistsError(f"cache output already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)

    token = uuid4().hex
    staging_dir = output_root.parent / f".{output_root.name}.tmp-{token}"
    remove_path(staging_dir)
    staging_dir.mkdir()

    try:
        summary = _generate_cache_bundle(
            args,
            bundle_root=staging_dir,
            requested_output_root=output_root,
        )
        for split_name in ("train", "val"):
            ActionTokenCache.open(staging_dir / split_name)
        _publish_new_cache_directory(staging_dir, output_root)
    except Exception:
        remove_path(staging_dir)
        raise
    print(json.dumps(summary))


def _publish_new_cache_directory(staging_dir: Path, output_root: Path) -> None:
    """Publish a complete cache without replacing another publisher's output."""
    try:
        # A valid cache contains non-empty train/val directories. POSIX rename is
        # atomic and refuses to replace a non-empty directory, so concurrent
        # publishers have one winner without a check-then-replace window.
        staging_dir.rename(output_root)
    except OSError as exc:
        if output_root.exists():
            raise FileExistsError(
                f"cache output already exists: {output_root}"
            ) from exc
        raise


def _generate_cache_bundle(
    args: argparse.Namespace,
    *,
    bundle_root: Path,
    requested_output_root: Path,
) -> dict[str, Any]:
    started = time.monotonic()

    native_tokenizer: XTokenizerCacheAdapter | None = None
    training_data_identity: dict[str, Any] | None = None
    bridge_trajectory: BridgeXTokenizerEpisode | None = None
    if args.checkpoint.is_dir():
        native_tokenizer = load_xtokenizer_cache_adapter(
            args.checkpoint,
            device=args.device,
            action_horizon=HORIZON,
            robot_type=ROBOT_TYPE,
        )
        training_data_identity = _require_native_training_data_identity(
            native_tokenizer.manifest
        )
        _validate_native_dataset_before_data_access(
            args.dataset_root,
            training_data_identity,
        )

    states, actions, episode_ids, global_indices = _load_bridge_columns(
        args.dataset_root
    )
    if not np.array_equal(global_indices, np.arange(len(global_indices))):
        raise ValueError("Bridge global index must be contiguous and start at zero.")
    valid_starts = _valid_window_starts(episode_ids, horizon=HORIZON)
    episode_count = int(episode_ids.max()) + 1
    train_episodes, val_episodes = split_train_val_episodes(
        list(range(episode_count)),
        val_ratio=float(args.val_ratio),
        split_seed=int(args.split_seed),
    )
    split_membership = {
        "train": _episode_membership(episode_count, train_episodes),
        "val": _episode_membership(episode_count, val_episodes),
    }
    split_rows = {
        name: np.flatnonzero(membership[episode_ids])
        for name, membership in split_membership.items()
    }
    split_starts = {
        name: valid_starts[membership[episode_ids[valid_starts]]]
        for name, membership in split_membership.items()
    }
    if args.max_valid_windows is not None:
        split_starts = {
            name: starts[: int(args.max_valid_windows)]
            for name, starts in split_starts.items()
        }

    if args.checkpoint.is_dir():
        if native_tokenizer is None or training_data_identity is None:
            raise AssertionError("Native artifact validation must precede data access.")
        tokenizer = native_tokenizer
        dof_dims = tokenizer.dof_dims
        dof_order = list(tokenizer.dof_order)
        checkpoint_digest = tokenizer.source_checkpoint_sha256
        artifact_digest = tokenizer.artifact_sha256
        robot_type_id = tokenizer.embodiment_registry[ROBOT_TYPE]
        codebook_size = tokenizer.codebook_size
        normalization_path = bundle_root / "normalization.json"
        _write_json(
            normalization_path,
            _canonicalize_native_normalization(
                tokenizer.manifest["normalization"],
                training_data_identity,
            ),
        )
        normalization_sha256 = _sha256_file(normalization_path)
        statistics_sha256 = None
    else:
        artifact_digest = None
        checkpoint_digest = _sha256_file(args.checkpoint)
        if checkpoint_digest != CHECKPOINT_SHA256:
            raise ValueError(
                f"X-Tokenizer checkpoint SHA256 mismatch: {checkpoint_digest}"
            )
        metadata = torch.load(
            args.checkpoint, map_location="cpu", weights_only=False, mmap=True
        )
        dof_dims = dict(metadata["data_spec"]["dof_dims"])
        dof_order = list(dof_dims)
        del metadata
        robot_type_id = 7
        codebook_size = 2048
        bridge_trajectory = _build_bridge_xtokenizer_trajectory(
            states,
            actions,
            episode_ids,
            dof_dims,
            dof_order=dof_order,
        )
        statistics = _compute_statistics(
            bridge_trajectory,
            split_starts["train"],
            dof_dims,
            dof_order=dof_order,
            batch_size=int(args.stats_batch_size),
        )
        statistics_path = bundle_root / "statistics.json"
        from xtokenizer.data import save_statistics_json, validate_statistics

        save_statistics_json({ROBOT_TYPE: statistics}, statistics_path)
        warnings = validate_statistics(
            {ROBOT_TYPE: statistics},
            dof_dims,
            list(dict.fromkeys(dof_dims)),
            list(dict.fromkeys(key.replace("_relative", "") for key in dof_dims)),
        )
        if warnings:
            raise ValueError("Invalid generated statistics: " + "; ".join(warnings))
        statistics_sha256 = _sha256_file(statistics_path)
        normalization_path = None
        normalization_sha256 = None
        tokenizer = _load_legacy_cache_tokenizer(
            args.checkpoint,
            device=args.device,
            statistics_path=statistics_path,
        )
    if bridge_trajectory is None:
        bridge_trajectory = _build_bridge_xtokenizer_trajectory(
            states,
            actions,
            episode_ids,
            dof_dims,
            dof_order=dof_order,
        )
    for split_name in ("train", "val"):
        _generate_split_cache(
            tokenizer=tokenizer,
            states=states,
            actions=actions,
            bridge_trajectory=bridge_trajectory,
            rows=split_rows[split_name],
            valid_starts=split_starts[split_name],
            episode_list=train_episodes if split_name == "train" else val_episodes,
            split_name=split_name,
            output_root=bundle_root / split_name,
            dof_dims=dof_dims,
            dof_order=dof_order,
            batch_size=int(args.batch_size),
            statistics_sha256=statistics_sha256,
            normalization_path=normalization_path,
            normalization_sha256=normalization_sha256,
            checkpoint_sha256=checkpoint_digest,
            artifact_sha256=artifact_digest,
            training_data_identity=training_data_identity,
            robot_type_id=robot_type_id,
            codebook_size=codebook_size,
            max_valid_windows=args.max_valid_windows,
            split_seed=int(args.split_seed),
            val_ratio=float(args.val_ratio),
        )

    summary = {
        "elapsed_seconds": time.monotonic() - started,
        "frames": len(states),
        "valid_windows": {key: len(value) for key, value in split_starts.items()},
        "rows": {key: len(value) for key, value in split_rows.items()},
        "output_root": str(requested_output_root),
    }
    (bundle_root / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _require_native_training_data_identity(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    identity = manifest.get("training_data_identity")
    if not isinstance(identity, dict):
        raise ValueError(
            "Native cache generation requires an exported training_data_identity."
        )
    normalization = manifest.get("normalization")
    modes = normalization.get("modes") if isinstance(normalization, dict) else None
    if (
        not isinstance(normalization, dict)
        or normalization.get("kind") != "xtokenizer_dual_role_v1"
        or not isinstance(modes, dict)
        or set(modes) != {"action", "observation.state"}
        or any(mode == "none" for mode in modes.values())
    ):
        raise ValueError(
            "Native cache generation rejects NONE normalization roles and requires "
            "xtokenizer_dual_role_v1."
        )
    return identity


def _canonicalize_native_normalization(
    exported: dict[str, Any],
    training_data_identity: dict[str, Any],
) -> dict[str, Any]:
    """Use canonical quantiles in JSON while preserving runtime float32 values."""
    try:
        canonical = training_data_identity["canonical_stats_identity"]["normalization"]
        canonical_stats = {
            "action": {
                "min": canonical["predict_action"]["q001"],
                "max": canonical["predict_action"]["q999"],
            },
            "observation.state": {
                "min": canonical["observation_state"]["q001"],
                "max": canonical["observation_state"]["q999"],
            },
        }
    except (KeyError, TypeError):
        raise ValueError(
            "Native cache canonical normalization identity is malformed."
        ) from None

    corrected = copy.deepcopy(exported)
    corrected["stats"] = canonical_stats
    try:
        for feature in ("action", "observation.state"):
            for bound in ("min", "max"):
                runtime = np.asarray(
                    exported["stats"][feature][bound],
                    dtype=np.float32,
                )
                source = np.asarray(
                    canonical_stats[feature][bound],
                    dtype=np.float32,
                )
                if runtime.shape != source.shape or not np.array_equal(
                    runtime,
                    source,
                ):
                    raise ValueError(
                        "Canonicalization would change runtime values for "
                        f"{feature}.{bound}."
                    )
    except (KeyError, TypeError):
        raise ValueError("Native cache exported normalization is malformed.") from None
    return corrected


def _validate_native_dataset_before_data_access(
    dataset_root: Path,
    identity: dict[str, Any],
) -> None:
    repo_id = str(identity["dataset_repo_id"])
    expected_revision = str(identity["dataset_revision"])
    actual_revision = read_lerobot_cache_provenance(
        root=dataset_root,
        repo_id=repo_id,
    )
    if actual_revision != expected_revision:
        raise ValueError(
            "LeRobot cache provenance revision does not match the exported "
            f"training dataset revision: {actual_revision!r} != "
            f"{expected_revision!r}."
        )

    missing = _missing_lerobot_metadata_paths(dataset_root)
    data_dir = dataset_root / "data"
    if not data_dir.is_dir() or not any(data_dir.rglob("*.parquet")):
        missing.append(data_dir / "*.parquet")
    if missing:
        raise FileNotFoundError(
            "Native cache dataset metadata is incomplete before data access: "
            + ", ".join(str(path) for path in missing)
        )
    for filename in ("info.json", "stats.json"):
        path = dataset_root / "meta" / filename
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Native cache metadata is malformed: {path}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Native cache metadata must be an object: {path}")

    tasks = pads.dataset(
        dataset_root / "meta" / "tasks.parquet",
        format="parquet",
    ).to_table()
    if len(tasks) == 0:
        raise ValueError("Native cache task metadata must not be empty.")
    episodes = pads.dataset(
        dataset_root / "meta" / "episodes",
        format="parquet",
    ).to_table()
    columns = set(episodes.column_names)
    if "episode_index" not in columns:
        raise ValueError("Native cache episode metadata is missing episode_index.")
    episode_indices = np.asarray(
        episodes["episode_index"].combine_chunks().to_numpy(),
        dtype=np.int64,
    )
    if {"dataset_from_index", "dataset_to_index"} <= columns:
        starts = np.asarray(
            episodes["dataset_from_index"].combine_chunks().to_numpy(),
            dtype=np.int64,
        )
        ends = np.asarray(
            episodes["dataset_to_index"].combine_chunks().to_numpy(),
            dtype=np.int64,
        )
        episode_lengths = ends - starts
    elif "length" in columns:
        episode_lengths = np.asarray(
            episodes["length"].combine_chunks().to_numpy(),
            dtype=np.int64,
        )
    else:
        raise ValueError(
            "Native cache episode metadata must contain length or dataset bounds."
        )
    if len(set(episode_indices.tolist())) != len(episode_indices) or bool(
        (episode_lengths <= 0).any()
    ):
        raise ValueError("Native cache episode metadata bounds are malformed.")
    sidecar_identity = identity["vl_sidecar_identity"]
    expected_episodes = {
        int(index): int(length)
        for index, length in zip(
            sidecar_identity["episode_indices"],
            sidecar_identity["episode_lengths"],
            strict=True,
        )
    }
    actual_episodes = {
        int(index): int(length)
        for index, length in zip(
            episode_indices,
            episode_lengths,
            strict=True,
        )
    }
    if actual_episodes != expected_episodes:
        raise ValueError(
            "Native cache episode metadata does not match the exported "
            "training sidecar identity."
        )


def _load_bridge_columns(
    dataset_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    table = pads.dataset(dataset_root / "data", format="parquet").to_table(
        columns=["observation.state", "action", "episode_index", "index"]
    )
    states = _fixed_list_column(table["observation.state"], width=7)
    actions = _fixed_list_column(table["action"], width=7)
    episode_ids = np.asarray(
        table["episode_index"].combine_chunks().to_numpy(), dtype=np.int64
    )
    global_indices = np.asarray(
        table["index"].combine_chunks().to_numpy(), dtype=np.int64
    )
    order = np.argsort(global_indices, kind="stable")
    return states[order], actions[order], episode_ids[order], global_indices[order]


def _fixed_list_column(column: Any, *, width: int) -> np.ndarray:
    array = column.combine_chunks()
    values = np.asarray(array.values.to_numpy(), dtype=np.float32)
    return values.reshape(len(array), width)


def _valid_window_starts(episode_ids: np.ndarray, *, horizon: int) -> np.ndarray:
    boundaries = np.flatnonzero(episode_ids[1:] != episode_ids[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(episode_ids)]))
    chunks = [
        np.arange(start, end - horizon + 1, dtype=np.int64)
        for start, end in zip(starts, ends, strict=False)
        if end - start >= horizon
    ]
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)


def _episode_membership(episode_count: int, selected: list[int]) -> np.ndarray:
    membership = np.zeros(episode_count, dtype=np.bool_)
    membership[np.asarray(selected, dtype=np.int64)] = True
    return membership


def _build_bridge_xtokenizer_trajectory(
    states: np.ndarray,
    actions: np.ndarray,
    episode_ids: np.ndarray,
    dof_dims: dict[str, int],
    *,
    dof_order: list[str],
) -> BridgeXTokenizerEpisode:
    boundaries = np.flatnonzero(episode_ids[1:] != episode_ids[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(episode_ids)]))
    absolute_parts: list[np.ndarray] = []
    observation_parts: list[np.ndarray] = []
    mask_parts: list[np.ndarray] = []
    seen_episodes: set[int] = set()
    for start, end in zip(starts, ends, strict=True):
        episode_id = int(episode_ids[start])
        if episode_id in seen_episodes:
            raise ValueError(f"Bridge episode {episode_id} rows must be contiguous.")
        seen_episodes.add(episode_id)
        episode = build_bridge_xtokenizer_episode(
            states[start:end],
            actions[start:end],
            dof_dims,
            dof_order=dof_order,
        )
        absolute_parts.append(episode.absolute_actions)
        observation_parts.append(episode.observation_states_absolute)
        mask_parts.append(episode.dof_mask)
    return BridgeXTokenizerEpisode(
        absolute_actions=np.concatenate(absolute_parts, axis=0),
        observation_states_absolute=np.concatenate(observation_parts, axis=0),
        dof_mask=np.concatenate(mask_parts, axis=0),
    )


def _compute_statistics(
    bridge_trajectory: BridgeXTokenizerEpisode,
    starts: np.ndarray,
    dof_dims: dict[str, int],
    *,
    dof_order: list[str],
    batch_size: int,
) -> dict[str, dict[str, list[float]]]:
    from xtokenizer.data.rotation import compute_delta_6d_batch
    from xtokenizer.tools.statistics_builder import _PerDimDigest

    active_dims = {
        "follow_left_ee_cartesian_pos": 3,
        "follow_left_ee_cartesian_pos_relative": 3,
        "follow_left_ee_rotation_6D": 6,
        "follow_left_ee_rotation_6D_relative": 6,
        "follow_left_gripper": 1,
    }
    digests = {
        key: [_PerDimDigest() for _ in range(dim)] for key, dim in active_dims.items()
    }
    layout = resolve_bridge_xtokenizer_layout(
        dof_dims,
        dof_order=dof_order,
    )
    for offset in range(0, len(starts), batch_size):
        batch_starts = starts[offset : offset + batch_size]
        anchors = bridge_trajectory.observation_states_absolute[batch_starts]
        endpoints = bridge_trajectory.absolute_actions[batch_starts + HORIZON - 1]
        relative_position = endpoints[:, layout.position] - anchors[:, layout.position]
        relative_rotation = compute_delta_6d_batch(
            endpoints[:, None, layout.rotation],
            anchors[:, layout.rotation],
        )[:, 0]
        values = {
            "follow_left_ee_cartesian_pos": anchors[:, layout.position],
            "follow_left_ee_cartesian_pos_relative": relative_position,
            "follow_left_ee_rotation_6D": anchors[:, layout.rotation],
            "follow_left_ee_rotation_6D_relative": relative_rotation,
            "follow_left_gripper": np.concatenate(
                (
                    anchors[:, layout.gripper],
                    endpoints[:, layout.gripper],
                )
            ),
        }
        for key, matrix in values.items():
            for dim, digest in enumerate(digests[key]):
                digest.update(matrix[:, dim])

    required_keys = set(dof_dims)
    required_keys.update(key.replace("_relative", "") for key in dof_dims)
    result = {
        key: _neutral_statistics(int(_dim_for_key(key, dof_dims)))
        for key in required_keys
    }
    for key, key_digests in digests.items():
        result[key] = _export_digests(key_digests)
    return result


def _dim_for_key(key: str, dof_dims: dict[str, int]) -> int:
    canonical = key.replace("_relative", "")
    for candidate, dim in dof_dims.items():
        if candidate.replace("_relative", "") == canonical:
            return int(dim)
    raise KeyError(key)


def _export_digests(digests: list[Any]) -> dict[str, list[float]]:
    from xtokenizer.tools.statistics_builder import (
        QUANTILE_LEVELS,
        QUANTILE_NAMES,
    )

    block: dict[str, list[float]] = {}
    for level, name in zip(QUANTILE_LEVELS, QUANTILE_NAMES, strict=False):
        if name == "min":
            block[name] = [float(digest.vmin) for digest in digests]
        elif name == "max":
            block[name] = [float(digest.vmax) for digest in digests]
        else:
            block[name] = [
                min(
                    max(digest.percentile(level), float(digest.vmin)),
                    float(digest.vmax),
                )
                for digest in digests
            ]
    block["delta"] = _positive_delta(block["max"], block["min"])
    block["delta_001"] = _positive_delta(block["q999"], block["q001"])
    block["delta_005"] = _positive_delta(block["q995"], block["q005"])
    block["delta_01"] = _positive_delta(block["q99"], block["q01"])
    return block


def _positive_delta(high: list[float], low: list[float]) -> list[float]:
    return [
        max(float(upper) - float(lower), 1e-8)
        for upper, lower in zip(high, low, strict=False)
    ]


def _neutral_statistics(dim: int) -> dict[str, list[float]]:
    from xtokenizer.tools.statistics_builder import QUANTILE_NAMES

    block = {name: [0.0] * dim for name in QUANTILE_NAMES}
    for key in ("delta", "delta_001", "delta_005", "delta_01"):
        block[key] = [1.0] * dim
    return block


def _generate_split_cache(
    *,
    tokenizer: Any,
    states: np.ndarray,
    actions: np.ndarray,
    bridge_trajectory: BridgeXTokenizerEpisode,
    rows: np.ndarray,
    valid_starts: np.ndarray,
    episode_list: list[int],
    split_name: str,
    output_root: Path,
    dof_dims: dict[str, int],
    dof_order: list[str],
    batch_size: int,
    statistics_sha256: str | None,
    normalization_path: Path | None,
    normalization_sha256: str | None,
    checkpoint_sha256: str,
    artifact_sha256: str | None,
    training_data_identity: dict[str, Any] | None,
    robot_type_id: int,
    codebook_size: int,
    max_valid_windows: int | None,
    split_seed: int,
    val_ratio: float,
) -> None:
    native = isinstance(tokenizer, XTokenizerCacheAdapter)
    layout = resolve_bridge_xtokenizer_layout(
        dof_dims,
        dof_order=dof_order,
    )
    if native:
        if normalization_path is None or normalization_sha256 is None:
            raise ValueError(
                "Native cache generation requires normalization provenance."
            )
        _verify_sha256(
            normalization_path,
            normalization_sha256,
            label="normalization",
        )
    else:
        if statistics_sha256 is None:
            raise ValueError("Legacy cache generation requires statistics provenance.")
        from xtokenizer.data import normalize_action as legacy_normalize_action
        from xtokenizer.data.rotation import (
            compute_delta_6d_batch as legacy_compute_delta_6d_batch,
        )

    output_root.mkdir(parents=True, exist_ok=True)
    tokens_path = output_root / "tokens.npy"
    tokens = np.lib.format.open_memmap(
        tokens_path, mode="w+", dtype=np.uint16, shape=(len(rows), 8, 4)
    )
    tokens[:] = 0
    global_to_local = np.full(len(states), -1, dtype=np.int64)
    global_to_local[rows] = np.arange(len(rows), dtype=np.int64)
    if not native:
        predict_min, predict_delta, obs_min, obs_delta = tokenizer.get_normalizer(
            ROBOT_TYPE
        )
    device = tokenizer.device

    for offset in range(0, len(valid_starts), batch_size):
        starts = valid_starts[offset : offset + batch_size]
        action_rows = starts[:, None] + np.arange(HORIZON)[None, :]
        absolute_actions = bridge_trajectory.absolute_actions[action_rows]
        absolute_anchors = bridge_trajectory.observation_states_absolute[starts]
        dof_mask = bridge_trajectory.dof_mask[action_rows]
        if native:
            physical_actions = absolute_actions
        else:
            physical_actions = np.zeros_like(absolute_actions)
            physical_actions[:, :, layout.position] = (
                absolute_actions[:, :, layout.position]
                - absolute_anchors[:, None, layout.position]
            )
            physical_actions[:, :, layout.rotation] = legacy_compute_delta_6d_batch(
                absolute_actions[:, :, layout.rotation],
                absolute_anchors[:, layout.rotation],
            )
            physical_actions[:, :, layout.gripper] = absolute_actions[
                :, :, layout.gripper
            ]

        physical_actions_tensor = torch.from_numpy(physical_actions).to(device)
        physical_obs_tensor = torch.from_numpy(absolute_anchors[:, None]).to(device)
        dof_mask_tensor = torch.from_numpy(dof_mask).to(device)
        if native:
            encoder_actions = physical_actions_tensor
            encoder_obs = physical_obs_tensor
        else:
            encoder_actions = torch.zeros(
                physical_actions.shape, dtype=torch.float32, device=device
            )
            encoder_obs = torch.zeros(
                (len(starts), 1, physical_actions.shape[-1]),
                dtype=torch.float32,
                device=device,
            )
            for segment in (layout.position, layout.rotation, layout.gripper):
                encoder_actions[:, :, segment] = legacy_normalize_action(
                    physical_actions_tensor[:, :, segment],
                    predict_min[segment].to(device),
                    predict_delta[segment].to(device),
                )
                encoder_obs[:, :, segment] = legacy_normalize_action(
                    physical_obs_tensor[:, :, segment],
                    obs_min[segment].to(device),
                    obs_delta[segment].to(device),
                )
        if native:
            indices = tokenizer.encode(
                encoder_actions,
                absolute_anchor=encoder_obs[:, 0],
                dof_mask=dof_mask_tensor,
                robot_type=ROBOT_TYPE,
                token_order="time_major",
            )["indices"]
        else:
            indices = tokenizer.encode(
                encoder_actions,
                obs_state=encoder_obs,
                dof_mask=dof_mask_tensor,
                robot_type=ROBOT_TYPE,
                token_order="time_major",
            )["indices"]
        local_rows = global_to_local[starts]
        if np.any(local_rows < 0):
            raise AssertionError(f"{split_name} contains a start outside its row set")
        tokens[local_rows] = indices.detach().cpu().numpy().astype(np.uint16)
        if offset == 0 or (offset // batch_size) % 20 == 0:
            print(
                f"{split_name}: encoded {min(offset + len(starts), len(valid_starts))}/"
                f"{len(valid_starts)}",
                flush=True,
            )
    tokens.flush()

    if native:
        if normalization_path is None or normalization_sha256 is None:
            raise ValueError(
                "Native cache generation requires normalization provenance."
            )
        _verify_sha256(
            normalization_path,
            normalization_sha256,
            label="normalization",
        )
    manifest = {
        "format_version": 2,
        "tokenizer": "x-tokenizer",
        "action_semantics": {
            "public_action": "absolute",
            "tokenized_action": "observation_relative",
            "observation_anchor": "absolute",
            "relative_reference": "single_pre_chunk_observation",
        },
        "robot_type_id": robot_type_id,
        "codebook_size": codebook_size,
        "num_quantizers": 4,
        "compression_ratio": 4,
        "action_horizon": HORIZON,
        "token_order": "time_major",
        "num_samples": len(rows),
        "num_valid_samples": len(valid_starts),
        "tokens_file": "tokens.npy",
        "tokens_sha256": _sha256_file(tokens_path),
        "split": split_name,
        "split_seed": split_seed,
        "val_ratio": val_ratio,
        "episode_ids_sha256": _sha256_ints(np.asarray(episode_list, dtype=np.int64)),
        "global_indices_sha256": _sha256_ints(rows),
        "first_global_index": int(rows[0]) if len(rows) else None,
        "last_global_index": int(rows[-1]) if len(rows) else None,
        "active_dofs": list(layout.active_dofs),
        "inactive_normalized_fill": 0.0,
        "tail_token_fill": 0,
        "gripper_clip": [0.0, 1.0],
        "max_valid_windows": max_valid_windows,
    }
    if native:
        if artifact_sha256 is None or training_data_identity is None:
            raise ValueError(
                "Native cache generation requires authenticated artifact provenance."
            )
        canonical_stats_identity = training_data_identity["canonical_stats_identity"]
        stats_horizon = canonical_stats_identity["normalization"]["stats_horizon"]
        if stats_horizon != HORIZON:
            raise ValueError(
                "Native cache statistics horizon must match action_horizon."
            )
        manifest.update(
            {
                "dataset_repo_id": training_data_identity["dataset_repo_id"],
                "dataset_revision": training_data_identity["dataset_revision"],
                "canonical_stats_identity": canonical_stats_identity,
                "content_fingerprint": canonical_stats_identity["content_fingerprint"],
                "stats_horizon": stats_horizon,
                "vl_sidecar_identity": training_data_identity["vl_sidecar_identity"],
                "artifact_sha256": artifact_sha256,
                "source_checkpoint_sha256": checkpoint_sha256,
                "normalization_source": "inference_artifact",
                "normalization_file": "../normalization.json",
                "normalization_sha256": normalization_sha256,
                "source_action_sha256": sha256_float32_matrix(actions[rows]),
                "source_observation_state_sha256": sha256_float32_matrix(states[rows]),
            }
        )
    else:
        manifest.update(
            {
                "source_revision": SOURCE_REVISION,
                "hf_revision": HF_REVISION,
                "checkpoint_sha256": checkpoint_sha256,
                "dataset_repo_id": "chaoqi-liu/bridge_single_view",
                "statistics_sha256": statistics_sha256,
                "statistics_key": ROBOT_TYPE,
            }
        )
    _write_json(output_root / "manifest.json", manifest)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_ints(values: np.ndarray) -> str:
    payload = np.ascontiguousarray(values, dtype=np.int64).tobytes()
    return hashlib.sha256(payload).hexdigest()


def _verify_sha256(path: Path, expected: str, *, label: str) -> None:
    actual = _sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: expected {expected}, got {actual}.")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _load_cache_tokenizer(
    checkpoint: Path,
    *,
    device: str,
    statistics_path: Path,
) -> Any:
    if checkpoint.is_dir():
        return load_xtokenizer_cache_adapter(
            checkpoint,
            device=device,
            action_horizon=HORIZON,
            robot_type=ROBOT_TYPE,
        )
    return _load_legacy_cache_tokenizer(
        checkpoint,
        device=device,
        statistics_path=statistics_path,
    )


def _load_legacy_cache_tokenizer(
    checkpoint: Path,
    *,
    device: str,
    statistics_path: Path,
) -> Any:
    from xtokenizer import XTokenizer

    return XTokenizer.from_pretrained(
        checkpoint,
        device=device,
        statistics_path=statistics_path,
    )


if __name__ == "__main__":
    main()
