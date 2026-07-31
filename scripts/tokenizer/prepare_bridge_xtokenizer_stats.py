#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Prepare paper-aligned Bridge statistics before X-Tokenizer training."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads
from omegaconf import OmegaConf

from praxis.core.types import NormMode
from praxis.data.bridge_xtokenizer import (
    build_bridge_xtokenizer_statistics_samples,
    resolve_bridge_xtokenizer_layout,
)
from praxis.data.lerobot_adapter import (
    _int64_column,
    _list_column_to_2d_float32,
    _missing_lerobot_metadata_paths,
    read_lerobot_cache_provenance,
)
from praxis.data.utils import resolve_action_feature_from_metadata
from praxis.data.xtokenizer_canonical_stats import (
    write_xtokenizer_quantile_statistics,
)


def prepare_bridge_xtokenizer_stats(
    dataset_config: Path,
    *,
    output: Path | None = None,
    mode: NormMode = NormMode.MIN_MAX,
    stats_horizon: int = 32,
) -> Path:
    """Write robust relative-action and absolute-anchor Bridge statistics."""
    cfg = OmegaConf.load(dataset_config)
    repo_id = _required_string(cfg, "repo_id")
    robot_type = _required_string(cfg, "robot_type_name")
    root = Path(_required_string(cfg, "root"))
    dataset_revision = _required_string(cfg, "dataset_revision")
    actual_revision = read_lerobot_cache_provenance(root=root, repo_id=repo_id)
    if actual_revision != dataset_revision.lower():
        raise ValueError(
            "LeRobot cache provenance revision does not match "
            f"dataset_revision: {actual_revision!r} != {dataset_revision!r}."
        )
    if (
        not isinstance(stats_horizon, int)
        or isinstance(stats_horizon, bool)
        or stats_horizon <= 0
    ):
        raise ValueError("stats_horizon must be a positive integer.")
    action_layout = cfg.action_layout
    if str(action_layout.convention) != "absolute":
        raise ValueError(
            "Bridge X-Tokenizer action_layout.convention must be absolute."
        )
    dof_dims = {
        str(name): int(dim)
        for name, dim in cast(
            dict[str, Any],
            OmegaConf.to_container(
                action_layout.dof_dims,
                resolve=True,
            ),
        ).items()
    }
    dof_order = tuple(str(name) for name in action_layout.dof_order)
    output_path = output
    if output_path is None:
        configured_output = getattr(cfg, "canonical_stats", None)
        if not isinstance(configured_output, str) or not configured_output:
            raise ValueError(
                "--output is required when dataset.canonical_stats is not configured."
            )
        output_path = Path(configured_output)
    if mode is not NormMode.MIN_MAX:
        raise ValueError("X-Tokenizer paper statistics require min_max mode.")

    info = _load_local_dataset_info(root)
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("LeRobot meta/info.json must define a features mapping.")
    action_feature = resolve_action_feature_from_metadata(
        cast(dict[str, Any], features),
        context="Bridge dataset metadata for canonical statistics",
    )
    configured_episodes = getattr(cfg, "episodes", None)
    episodes = (
        None
        if configured_episodes is None
        else tuple(int(index) for index in configured_episodes)
    )
    table = _load_local_bridge_table(
        root=root,
        episodes=episodes,
        info=info,
        action_key=action_feature.key,
    )
    actions = _list_column_to_2d_float32(
        table.column(action_feature.key),
        table.num_rows,
    )
    states = _list_column_to_2d_float32(
        table.column("observation.state"),
        table.num_rows,
    )
    episode_indices = _int64_column(table.column("episode_index"))
    if actions.shape[1] != 7 or states.shape[1] != 7:
        raise ValueError(
            "Bridge canonical statistics require raw 7-D action/state columns, "
            f"got action={actions.shape}, state={states.shape}."
        )

    episode_rows = [
        np.flatnonzero(episode_indices == episode_index)
        for episode_index in dict.fromkeys(int(value) for value in episode_indices)
    ]
    for episode_index, rows in zip(
        dict.fromkeys(int(value) for value in episode_indices),
        episode_rows,
        strict=True,
    ):
        if not np.array_equal(rows, np.arange(rows[0], rows[-1] + 1)):
            raise ValueError(f"Bridge episode {episode_index} rows must be contiguous.")
    sample_count = sum(max(len(rows) - stats_horizon + 1, 0) for rows in episode_rows)
    if sample_count == 0:
        raise ValueError("Bridge dataset contains no fixed-horizon endpoint samples.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    layout = resolve_bridge_xtokenizer_layout(dof_dims, dof_order=dof_order)
    with tempfile.TemporaryDirectory(
        prefix=".xtokenizer-stats-",
        dir=output_path.parent,
    ) as temporary_dir:
        relative_values = np.memmap(
            Path(temporary_dir) / "relative-actions.f32",
            mode="w+",
            dtype=np.float32,
            shape=(sample_count, layout.action_dim),
        )
        observation_values = np.memmap(
            Path(temporary_dir) / "absolute-observations.f32",
            mode="w+",
            dtype=np.float32,
            shape=(sample_count, layout.action_dim),
        )
        offset = 0
        for rows in episode_rows:
            samples = build_bridge_xtokenizer_statistics_samples(
                states[rows],
                actions[rows],
                dof_dims,
                stats_horizon=stats_horizon,
                dof_order=dof_order,
            )
            next_offset = offset + len(samples.predict_action)
            relative_values[offset:next_offset] = samples.predict_action
            observation_values[offset:next_offset] = samples.observation_state
            offset = next_offset
        if offset != sample_count:
            raise AssertionError("X-Tokenizer statistics sample accounting mismatch.")
        relative_values.flush()
        observation_values.flush()
        action_statistics = _quantile_statistics(
            relative_values,
            role="predict_action",
        )
        observation_statistics = _quantile_statistics(
            observation_values,
            role="observation_state",
        )
    return write_xtokenizer_quantile_statistics(
        output_path,
        dataset_repo_id=repo_id,
        dataset_revision=dataset_revision,
        robot_type=robot_type,
        dof_dims=dof_dims,
        dof_order=dof_order,
        stats_horizon=stats_horizon,
        predict_action_statistics=action_statistics,
        observation_state_statistics=observation_statistics,
    )


def _quantile_statistics(
    values: np.ndarray,
    *,
    role: str,
) -> dict[str, list[float]]:
    if values.ndim != 2 or len(values) == 0:
        raise ValueError(f"Bridge {role} statistics population must not be empty.")
    quantiles = np.asarray(
        np.quantile(values, [0.001, 0.999], axis=0),
        dtype=np.float64,
    )
    degenerate = np.flatnonzero(quantiles[0] >= quantiles[1])
    if len(degenerate):
        raise ValueError(
            f"Bridge {role} statistics have degenerate quantile channels: "
            f"{degenerate.tolist()}."
        )
    return {"q001": quantiles[0].tolist(), "q999": quantiles[1].tolist()}


def _load_local_dataset_info(root: Path) -> dict[str, Any]:
    missing = _missing_lerobot_metadata_paths(root)
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Incomplete local LeRobot cache at {root}; missing {missing_text}."
        )
    info_path = root / "meta" / "info.json"
    try:
        payload = json.loads(info_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Invalid local LeRobot metadata file {info_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Local LeRobot metadata file {info_path} must be an object.")
    return cast(dict[str, Any], payload)


def _load_local_bridge_table(
    *,
    root: Path,
    episodes: Sequence[int] | None,
    info: dict[str, Any],
    action_key: str,
) -> pa.Table:
    episode_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    episode_table = pads.dataset(
        [str(path) for path in episode_files],
        format="parquet",
    ).to_table(
        columns=[
            "episode_index",
            "length",
            "data/chunk_index",
            "data/file_index",
        ]
    )
    episode_indices = _int64_column(episode_table.column("episode_index"))
    episode_lengths = _int64_column(episode_table.column("length"))
    chunk_indices = _int64_column(episode_table.column("data/chunk_index"))
    file_indices = _int64_column(episode_table.column("data/file_index"))
    if len({int(index) for index in episode_indices}) != len(episode_indices):
        raise ValueError("Local LeRobot episode metadata contains duplicate episodes.")

    total_episodes = info.get("total_episodes")
    if (
        not isinstance(total_episodes, int)
        or isinstance(total_episodes, bool)
        or total_episodes < 0
    ):
        raise ValueError("LeRobot meta/info.json total_episodes must be non-negative.")
    available_episodes = {int(index) for index in episode_indices}
    expected_episodes = set(range(total_episodes))
    if available_episodes != expected_episodes:
        missing_episodes = sorted(expected_episodes - available_episodes)
        extra_episodes = sorted(available_episodes - expected_episodes)
        raise FileNotFoundError(
            "Incomplete local LeRobot cache episode metadata: "
            f"missing={missing_episodes}, extra={extra_episodes}."
        )

    if episodes is None:
        selected_episodes = expected_episodes
    else:
        selected_list = [int(index) for index in episodes]
        if not selected_list:
            raise ValueError("dataset.episodes must select at least one episode.")
        if len(set(selected_list)) != len(selected_list):
            raise ValueError("dataset.episodes must not contain duplicates.")
        selected_episodes = set(selected_list)
        missing_requested = sorted(selected_episodes - available_episodes)
        if missing_requested:
            raise ValueError(
                "Requested Bridge episodes are absent from the pinned local cache: "
                f"{missing_requested}."
            )
    if not selected_episodes:
        raise ValueError("Bridge dataset contains no episodes.")

    data_path_template = info.get("data_path")
    if not isinstance(data_path_template, str) or not data_path_template:
        raise ValueError("LeRobot meta/info.json must define data_path.")
    selected_rows = np.asarray(
        [int(index) in selected_episodes for index in episode_indices],
        dtype=np.bool_,
    )
    data_files: set[Path] = set()
    for chunk_index, file_index in zip(
        chunk_indices[selected_rows],
        file_indices[selected_rows],
        strict=True,
    ):
        relative_path = Path(
            data_path_template.format(
                chunk_index=int(chunk_index),
                file_index=int(file_index),
            )
        )
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                "LeRobot meta/info.json data_path must stay within the dataset root."
            )
        data_files.add(root / relative_path)
    missing_data = sorted(path for path in data_files if not path.is_file())
    if missing_data:
        missing_text = ", ".join(str(path) for path in missing_data)
        raise FileNotFoundError(
            f"Incomplete local LeRobot cache at {root}; missing {missing_text}."
        )

    selected_list = sorted(selected_episodes)
    table = pads.dataset(
        [str(path) for path in sorted(data_files)],
        format="parquet",
    ).to_table(
        columns=[
            action_key,
            "observation.state",
            "episode_index",
            "frame_index",
        ],
        filter=pads.field("episode_index").isin(selected_list),
    )
    loaded_episode_indices = _int64_column(table.column("episode_index"))
    loaded_frame_indices = _int64_column(table.column("frame_index"))
    loaded_episodes = {int(index) for index in loaded_episode_indices}
    if loaded_episodes != selected_episodes:
        missing_loaded = sorted(selected_episodes - loaded_episodes)
        raise FileNotFoundError(
            "Incomplete local LeRobot cache data for requested episodes: "
            f"{missing_loaded}."
        )

    validation_errors: list[str] = []
    for metadata_row in np.flatnonzero(selected_rows):
        episode_index = int(episode_indices[metadata_row])
        expected_length = int(episode_lengths[metadata_row])
        frame_indices = loaded_frame_indices[loaded_episode_indices == episode_index]
        if len(frame_indices) != expected_length:
            validation_errors.append(
                f"episode {episode_index} declares {expected_length} frames "
                f"but contains {len(frame_indices)}"
            )
        expected_frame_indices = np.arange(expected_length, dtype=np.int64)
        if not np.array_equal(np.sort(frame_indices), expected_frame_indices):
            validation_errors.append(
                f"episode {episode_index} frame_index must be unique and exactly "
                f"0..{expected_length - 1}"
            )
    if validation_errors:
        raise ValueError(
            "Incomplete local LeRobot cache data: " + "; ".join(validation_errors)
        )

    order = np.lexsort((loaded_frame_indices, loaded_episode_indices))
    return table.take(pa.array(order, type=pa.int64()))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--mode",
        choices=(NormMode.MIN_MAX.value,),
        default=NormMode.MIN_MAX.value,
    )
    parser.add_argument("--stats-horizon", type=int, default=32)
    args = parser.parse_args(argv)
    destination = prepare_bridge_xtokenizer_stats(
        args.dataset_config,
        output=args.output,
        mode=NormMode(args.mode),
        stats_horizon=args.stats_horizon,
    )
    print(destination)
    return 0


def _required_string(cfg: Any, key: str) -> str:
    value = getattr(cfg, key, None)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"dataset.{key} must be a non-empty string.")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
