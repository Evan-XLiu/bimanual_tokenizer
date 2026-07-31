# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from praxis.data.bridge_xtokenizer import build_bridge_xtokenizer_statistics_samples
from praxis.data.lerobot_adapter import write_lerobot_cache_provenance
from praxis.data.xtokenizer_canonical_stats import XTokenizerQuantileStatistics
from scripts.tokenizer import prepare_bridge_xtokenizer_stats
from scripts.tokenizer.prepare_bridge_xtokenizer_stats import main

DATASET_REVISION = "a" * 40
DOF_DIMS = {
    "follow_left_gripper": 1,
    "follow_left_ee_cartesian_pos": 3,
    "follow_left_ee_rotation_6D": 6,
}
DOF_ORDER = tuple(DOF_DIMS)
STATES = np.array(
    [
        [0.0, 0.0, 0.0, 0.2, -0.3, 0.4, 0.1],
        [1.0, 2.0, 3.0, -0.4, 0.25, -0.1, 0.2],
        [4.0, -1.0, 2.0, 0.35, 0.15, -0.45, 0.3],
        [9.0, 8.0, -2.0, -0.2, -0.5, 0.3, 0.4],
        [2.0, 7.0, 6.0, 0.45, -0.25, 0.15, 0.5],
    ],
    dtype=np.float32,
)
ACTIONS = np.array(
    [
        [10.0, -7.0, 4.0, 0.4, -0.2, 0.3, 0.11],
        [-9.0, 8.0, -3.0, -0.1, 0.35, -0.2, 0.22],
        [2.0, -5.0, 7.0, 0.25, -0.15, 0.4, 0.33],
        [-6.0, 1.0, -2.0, -0.3, 0.2, -0.1, 0.44],
        [1.0, 2.0, 3.0, 0.1, 0.1, 0.1, 0.55],
    ],
    dtype=np.float32,
)


def _write_bridge_fixture(
    root: Path,
    *,
    episode_count: int = 1,
    states: np.ndarray = STATES,
    actions: np.ndarray = ACTIONS,
) -> None:
    dataset = LeRobotDataset.create(
        repo_id="org/bridge",
        fps=10,
        root=root,
        robot_type="widowx",
        use_videos=False,
        features={
            "observation.state": {
                "dtype": "float32",
                "shape": (7,),
                "names": None,
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": None,
            },
        },
    )
    for _episode_index in range(episode_count):
        for state, action in zip(states, actions, strict=True):
            dataset.add_frame(
                {
                    "observation.state": state,
                    "action": action,
                    "task": "move",
                }
            )
        dataset.save_episode(parallel_encoding=False)
    dataset.finalize()
    write_lerobot_cache_provenance(
        root=root,
        repo_id="org/bridge",
        revision=DATASET_REVISION,
    )


def _write_dataset_config(path: Path, *, dataset_root: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "repo_id: org/bridge",
                f"root: {dataset_root}",
                f'dataset_revision: "{DATASET_REVISION}"',
                "robot_type_name: WidowX",
                "episodes: null",
                "action_horizon: 2",
                "action_layout:",
                "  convention: absolute",
                "  dof_dims:",
                "    follow_left_gripper: 1",
                "    follow_left_ee_cartesian_pos: 3",
                "    follow_left_ee_rotation_6D: 6",
                "  dof_order:",
                "    - follow_left_gripper",
                "    - follow_left_ee_cartesian_pos",
                "    - follow_left_ee_rotation_6D",
            ]
        )
        + "\n"
    )


def test_prepare_bridge_stats_cli_uses_relative_actions_and_absolute_anchors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_root = tmp_path / "dataset"
    output = tmp_path / "canonical_stats.json"
    config = tmp_path / "dataset.yaml"
    _write_bridge_fixture(dataset_root)
    _write_dataset_config(config, dataset_root=dataset_root)

    assert (
        main(
            [
                "--dataset-config",
                str(config),
                "--output",
                str(output),
                "--stats-horizon",
                "2",
            ]
        )
        == 0
    )

    artifact = XTokenizerQuantileStatistics.from_path(
        output,
        expected_dataset_repo_id="org/bridge",
        expected_dataset_revision=DATASET_REVISION,
        expected_robot_type="WidowX",
        expected_dof_dims=DOF_DIMS,
        expected_dof_order=DOF_ORDER,
        expected_action_dim=10,
        expected_stats_horizon=2,
    )
    assert artifact.robot_type == "WidowX"
    assert artifact.stats_horizon == 2
    assert artifact.clamp is True
    assert artifact.quantile_levels == (0.001, 0.999)
    assert artifact.predict_space == "observation_relative"
    assert artifact.observation_space == "absolute"
    samples = build_bridge_xtokenizer_statistics_samples(
        STATES,
        ACTIONS,
        DOF_DIMS,
        stats_horizon=2,
        dof_order=DOF_ORDER,
    )
    expected_predict = np.quantile(samples.predict_action, [0.001, 0.999], axis=0)
    expected_observation = np.quantile(
        samples.observation_state,
        [0.001, 0.999],
        axis=0,
    )
    np.testing.assert_allclose(
        artifact.predict_action_statistics["q001"],
        expected_predict[0],
    )
    np.testing.assert_allclose(
        artifact.predict_action_statistics["q999"],
        expected_predict[1],
    )
    np.testing.assert_allclose(
        artifact.observation_state_statistics["q001"],
        expected_observation[0],
    )
    np.testing.assert_allclose(
        artifact.observation_state_statistics["q999"],
        expected_observation[1],
    )
    assert output.as_posix() in capsys.readouterr().out


def test_prepare_bridge_stats_counts_the_final_complete_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "dataset"
    output = tmp_path / "canonical_stats.json"
    config = tmp_path / "dataset.yaml"
    _write_bridge_fixture(
        dataset_root,
        states=STATES[:3],
        actions=ACTIONS[:3],
    )
    _write_dataset_config(config, dataset_root=dataset_root)
    observed_counts: dict[str, int] = {}
    original = prepare_bridge_xtokenizer_stats._quantile_statistics

    def _capture(values: np.ndarray, *, role: str):
        observed_counts[role] = len(values)
        return original(values, role=role)

    monkeypatch.setattr(
        prepare_bridge_xtokenizer_stats,
        "_quantile_statistics",
        _capture,
    )

    prepare_bridge_xtokenizer_stats.prepare_bridge_xtokenizer_stats(
        config,
        output=output,
        stats_horizon=2,
    )

    assert observed_counts == {"predict_action": 2, "observation_state": 2}


def test_prepare_bridge_stats_defaults_to_horizon_32(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _capture(*args: object, **kwargs: object) -> Path:
        captured["args"] = args
        captured.update(kwargs)
        return tmp_path / "stats.json"

    monkeypatch.setattr(
        prepare_bridge_xtokenizer_stats,
        "prepare_bridge_xtokenizer_stats",
        _capture,
    )

    assert main(["--dataset-config", str(tmp_path / "dataset.yaml")]) == 0
    assert captured["stats_horizon"] == 32


def test_prepare_bridge_stats_missing_local_data_never_downloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "dataset"
    config = tmp_path / "dataset.yaml"
    _write_bridge_fixture(dataset_root)
    _write_dataset_config(config, dataset_root=dataset_root)
    next((dataset_root / "data").rglob("*.parquet")).unlink()
    download_calls: list[tuple[object, ...]] = []

    def _reject_download(*args: object, **kwargs: object) -> None:
        del kwargs
        download_calls.append(args)
        raise AssertionError("snapshot_download must not be called")

    monkeypatch.setattr(
        "lerobot.datasets.lerobot_dataset.snapshot_download",
        _reject_download,
    )
    monkeypatch.setattr(
        "lerobot.datasets.lerobot_dataset.get_safe_version",
        lambda repo_id, revision: revision,
    )

    with pytest.raises(FileNotFoundError, match="Incomplete local LeRobot cache"):
        main(
            [
                "--dataset-config",
                str(config),
                "--output",
                str(tmp_path / "canonical_stats.json"),
                "--stats-horizon",
                "2",
            ]
        )
    assert download_calls == []


def test_prepare_bridge_stats_rejects_balanced_per_episode_corruption(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    output = tmp_path / "canonical_stats.json"
    config = tmp_path / "dataset.yaml"
    _write_bridge_fixture(dataset_root, episode_count=2)
    _write_dataset_config(config, dataset_root=dataset_root)
    data_path = next((dataset_root / "data").rglob("*.parquet"))
    table = pq.read_table(data_path)
    corrupted = table.take(pa.array([0, 1, 2, 3, 5, 6, 7, 8, 9, 9], type=pa.int64()))
    pq.write_table(corrupted, data_path)

    with pytest.raises(ValueError) as exc_info:
        main(
            [
                "--dataset-config",
                str(config),
                "--output",
                str(output),
                "--stats-horizon",
                "2",
            ]
        )
    message = str(exc_info.value)
    assert "episode 0 declares 5 frames but contains 4" in message
    assert "episode 1 declares 5 frames but contains 6" in message
    assert "episode 1 frame_index must be unique and exactly 0..4" in message
    assert not output.exists()


def test_prepare_bridge_stats_orders_physical_rows_by_frame_index(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    ordered_output = tmp_path / "ordered_stats.json"
    shuffled_output = tmp_path / "shuffled_stats.json"
    config = tmp_path / "dataset.yaml"
    _write_bridge_fixture(dataset_root)
    _write_dataset_config(config, dataset_root=dataset_root)

    assert (
        main(
            [
                "--dataset-config",
                str(config),
                "--output",
                str(ordered_output),
                "--stats-horizon",
                "2",
            ]
        )
        == 0
    )
    data_path = next((dataset_root / "data").rglob("*.parquet"))
    table = pq.read_table(data_path)
    pq.write_table(
        table.take(pa.array([2, 0, 4, 1, 3], type=pa.int64())),
        data_path,
    )

    assert (
        main(
            [
                "--dataset-config",
                str(config),
                "--output",
                str(shuffled_output),
                "--stats-horizon",
                "2",
            ]
        )
        == 0
    )
    assert shuffled_output.read_bytes() == ordered_output.read_bytes()


def test_prepare_bridge_stats_rejects_empty_fixed_horizon_population(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    output = tmp_path / "stats.json"
    config = tmp_path / "dataset.yaml"
    _write_bridge_fixture(dataset_root)
    _write_dataset_config(config, dataset_root=dataset_root)

    with pytest.raises(ValueError, match="no fixed-horizon endpoint samples"):
        main(
            [
                "--dataset-config",
                str(config),
                "--output",
                str(output),
                "--stats-horizon",
                "6",
            ]
        )
    assert not output.exists()


def test_prepare_bridge_stats_rejects_degenerate_quantile_channels(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    output = tmp_path / "stats.json"
    config = tmp_path / "dataset.yaml"
    _write_bridge_fixture(
        dataset_root,
        states=np.zeros((5, 7), dtype=np.float32),
        actions=np.zeros((5, 7), dtype=np.float32),
    )
    _write_dataset_config(config, dataset_root=dataset_root)

    with pytest.raises(ValueError, match="degenerate quantile channels"):
        main(
            [
                "--dataset-config",
                str(config),
                "--output",
                str(output),
                "--stats-horizon",
                "2",
            ]
        )
    assert not output.exists()
