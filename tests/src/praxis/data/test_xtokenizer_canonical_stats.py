# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import pytest
import torch

from praxis.core.types import NormMode
from praxis.data.xtokenizer_canonical_stats import (
    CanonicalActionStatistics,
    XTokenizerQuantileStatistics,
    write_canonical_action_statistics,
    write_xtokenizer_quantile_statistics,
)

DATASET_REVISION = "a" * 40
DOF_DIMS = {
    "follow_left_gripper": 1,
    "follow_left_ee_cartesian_pos": 3,
    "follow_left_ee_rotation_6D": 6,
}
DOF_ORDER = tuple(DOF_DIMS)
ACTION_STATISTICS = {
    "min": [-1.0] * 10,
    "max": [1.0] * 10,
}
OBSERVATION_STATISTICS = {
    "min": [-2.0] * 10,
    "max": [2.0] * 10,
}
PREDICT_QUANTILES = {
    "q001": [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5],
    "q999": [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5],
}
OBSERVATION_QUANTILES = {
    "q001": [-8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0],
    "q999": [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
}


def _write_valid(path: Path) -> None:
    write_canonical_action_statistics(
        path,
        dataset_repo_id="org/bridge",
        dataset_revision=DATASET_REVISION,
        dof_dims=DOF_DIMS,
        dof_order=DOF_ORDER,
        mode=NormMode.MIN_MAX,
        action_statistics=ACTION_STATISTICS,
        observation_statistics=OBSERVATION_STATISTICS,
    )


def _load(path: Path) -> CanonicalActionStatistics:
    return CanonicalActionStatistics.from_path(
        path,
        expected_dataset_repo_id="org/bridge",
        expected_dataset_revision=DATASET_REVISION,
        expected_dof_dims=DOF_DIMS,
        expected_dof_order=DOF_ORDER,
        expected_action_dim=10,
    )


def _write_quantile_valid(path: Path) -> None:
    write_xtokenizer_quantile_statistics(
        path,
        dataset_repo_id="org/bridge",
        dataset_revision=DATASET_REVISION,
        robot_type="WidowX",
        dof_dims=DOF_DIMS,
        dof_order=DOF_ORDER,
        stats_horizon=32,
        predict_action_statistics=PREDICT_QUANTILES,
        observation_state_statistics=OBSERVATION_QUANTILES,
    )


def _load_quantile(
    path: Path,
    *,
    expected_stats_horizon: int = 32,
) -> XTokenizerQuantileStatistics:
    return XTokenizerQuantileStatistics.from_path(
        path,
        expected_dataset_repo_id="org/bridge",
        expected_dataset_revision=DATASET_REVISION,
        expected_robot_type="WidowX",
        expected_dof_dims=DOF_DIMS,
        expected_dof_order=DOF_ORDER,
        expected_action_dim=10,
        expected_stats_horizon=expected_stats_horizon,
    )


def _rewrite_quantile_artifact(
    path: Path,
    mutate,
    *,
    refresh_fingerprint: bool = True,
) -> None:
    raw = json.loads(path.read_text())
    mutate(raw)
    if refresh_fingerprint:
        payload = {
            key: value for key, value in raw.items() if key != "content_fingerprint"
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        raw["content_fingerprint"] = hashlib.sha256(encoded).hexdigest()
    path.write_text(json.dumps(raw))


def test_xtokenizer_stats_source_has_no_bridge_module_import() -> None:
    source_path = Path("src/praxis/data/xtokenizer_canonical_stats.py")
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    imported_modules = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }

    assert "praxis.data.bridge_xtokenizer" not in imported_modules


def test_xtokenizer_quantile_stats_round_trip_preserves_distinct_roles(
    tmp_path: Path,
) -> None:
    path = tmp_path / "quantile_stats.json"
    _write_quantile_valid(path)

    artifact = _load_quantile(path)

    assert artifact.dataset_repo_id == "org/bridge"
    assert artifact.dataset_revision == DATASET_REVISION
    assert artifact.robot_type == "WidowX"
    assert artifact.action_dim == 10
    assert artifact.dof_order == DOF_ORDER
    assert artifact.stats_horizon == 32
    assert artifact.quantile_levels == (0.001, 0.999)
    assert artifact.clamp is True
    assert artifact.predict_space == "observation_relative"
    assert artifact.observation_space == "absolute"
    assert artifact.identity == {
        "format_version": 3,
        "dataset_repo_id": "org/bridge",
        "dataset_revision": DATASET_REVISION,
        "robot_type": "WidowX",
        "action_layout": {
            "convention": "absolute",
            "dof_dims": DOF_DIMS,
            "dof_order": list(DOF_ORDER),
        },
        "normalization": {
            "mode": "quantile_min_max",
            "quantile_low": 0.001,
            "quantile_high": 0.999,
            "clamp": True,
            "stats_horizon": 32,
            "predict_action": {
                "space": "observation_relative",
                "q001": PREDICT_QUANTILES["q001"],
                "q999": PREDICT_QUANTILES["q999"],
            },
            "observation_state": {
                "space": "absolute",
                "q001": OBSERVATION_QUANTILES["q001"],
                "q999": OBSERVATION_QUANTILES["q999"],
            },
        },
        "content_fingerprint": artifact.content_fingerprint,
    }
    predict = artifact.predict_normalization_stats("action")
    observation = artifact.observation_normalization_stats("observation.state")
    torch.testing.assert_close(
        predict["action"]["min"],
        torch.tensor(PREDICT_QUANTILES["q001"], dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        predict["action"]["max"],
        torch.tensor(PREDICT_QUANTILES["q999"], dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        observation["observation.state"]["min"],
        torch.tensor(OBSERVATION_QUANTILES["q001"], dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        observation["observation.state"]["max"],
        torch.tensor(OBSERVATION_QUANTILES["q999"], dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    with pytest.raises(FrozenInstanceError):
        artifact.stats_horizon = 16  # type: ignore[misc]


def test_xtokenizer_quantile_stats_rejects_pre_episode_target_v2(
    tmp_path: Path,
) -> None:
    path = tmp_path / "quantile_stats.json"
    _write_quantile_valid(path)
    _rewrite_quantile_artifact(
        path,
        lambda raw: raw.update(format_version=2),
    )

    with pytest.raises(ValueError, match="format_version mismatch"):
        _load_quantile(path)


def test_xtokenizer_quantile_stats_are_deeply_immutable(tmp_path: Path) -> None:
    path = tmp_path / "quantile_stats.json"
    _write_quantile_valid(path)
    artifact = _load_quantile(path)
    identity = artifact.identity
    predict = artifact.predict_normalization_stats("action")["action"]["min"].clone()
    observation = artifact.observation_normalization_stats("observation.state")[
        "observation.state"
    ]["max"].clone()

    assert isinstance(artifact.dof_dims, MappingProxyType)
    assert isinstance(artifact.predict_action_statistics, MappingProxyType)
    assert isinstance(artifact.observation_state_statistics, MappingProxyType)
    with pytest.raises(TypeError):
        cast(Any, artifact.dof_dims)["follow_left_gripper"] = 2
    with pytest.raises(TypeError):
        cast(Any, artifact.predict_action_statistics)["q001"] = (0.0,) * 10
    with pytest.raises(TypeError):
        cast(Any, artifact.observation_state_statistics)["q999"] = (1.0,) * 10
    with pytest.raises(TypeError):
        cast(Any, artifact.predict_action_statistics["q001"])[0] = -99.0

    assert artifact.identity == identity
    assert artifact.content_fingerprint == identity["content_fingerprint"]
    torch.testing.assert_close(
        artifact.predict_normalization_stats("action")["action"]["min"],
        predict,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        artifact.observation_normalization_stats("observation.state")[
            "observation.state"
        ]["max"],
        observation,
        rtol=0,
        atol=0,
    )


def test_xtokenizer_quantile_stats_defensively_copy_constructor_mappings() -> None:
    source_dims = dict(DOF_DIMS)
    source_predict = {name: tuple(values) for name, values in PREDICT_QUANTILES.items()}
    source_observation = {
        name: tuple(values) for name, values in OBSERVATION_QUANTILES.items()
    }
    artifact = XTokenizerQuantileStatistics(
        dataset_repo_id="org/bridge",
        dataset_revision=DATASET_REVISION,
        robot_type="WidowX",
        dof_dims=source_dims,
        dof_order=DOF_ORDER,
        stats_horizon=32,
        predict_action_statistics=source_predict,
        observation_state_statistics=source_observation,
        content_fingerprint="f" * 64,
    )

    source_dims["follow_left_gripper"] = 99
    source_predict["q001"] = (-99.0,) * 10
    source_observation["q999"] = (99.0,) * 10

    assert artifact.action_dim == 10
    assert artifact.predict_action_statistics["q001"] == tuple(
        PREDICT_QUANTILES["q001"]
    )
    assert artifact.observation_state_statistics["q999"] == tuple(
        OBSERVATION_QUANTILES["q999"]
    )


def test_xtokenizer_quantile_stats_publication_is_idempotent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "quantile_stats.json"
    _write_quantile_valid(path)
    original_bytes = path.read_bytes()
    original_inode = path.stat().st_ino

    _write_quantile_valid(path)

    assert path.read_bytes() == original_bytes
    assert path.stat().st_ino == original_inode


def test_xtokenizer_quantile_stats_refuses_different_existing_content(
    tmp_path: Path,
) -> None:
    path = tmp_path / "quantile_stats.json"
    _write_quantile_valid(path)
    original_bytes = path.read_bytes()
    different_predict = {
        name: list(values) for name, values in PREDICT_QUANTILES.items()
    }
    different_predict["q001"][0] = -2.0

    with pytest.raises(ValueError, match="already exists.*different"):
        write_xtokenizer_quantile_statistics(
            path,
            dataset_repo_id="org/bridge",
            dataset_revision=DATASET_REVISION,
            robot_type="WidowX",
            dof_dims=DOF_DIMS,
            dof_order=DOF_ORDER,
            stats_horizon=32,
            predict_action_statistics=different_predict,
            observation_state_statistics=OBSERVATION_QUANTILES,
        )

    assert path.read_bytes() == original_bytes


def test_xtokenizer_quantile_stats_never_overwrites_or_reinterprets_legacy(
    tmp_path: Path,
) -> None:
    legacy_path = tmp_path / "canonical_stats.json"
    quantile_path = tmp_path / "quantile_stats.json"
    _write_valid(legacy_path)
    _write_quantile_valid(quantile_path)
    legacy_bytes = legacy_path.read_bytes()

    with pytest.raises(ValueError, match="already exists.*different"):
        write_xtokenizer_quantile_statistics(
            legacy_path,
            dataset_repo_id="org/bridge",
            dataset_revision=DATASET_REVISION,
            robot_type="WidowX",
            dof_dims=DOF_DIMS,
            dof_order=DOF_ORDER,
            stats_horizon=32,
            predict_action_statistics=PREDICT_QUANTILES,
            observation_state_statistics=OBSERVATION_QUANTILES,
        )

    assert legacy_path.read_bytes() == legacy_bytes
    assert (
        _load(legacy_path).normalization_stats("action")["action"]["min"].tolist()
        == ACTION_STATISTICS["min"]
    )
    with pytest.raises(ValueError, match="fields"):
        _load(quantile_path)
    with pytest.raises(ValueError, match="fields"):
        _load_quantile(legacy_path)


@pytest.mark.parametrize("role", ["predict_action", "observation_state"])
def test_xtokenizer_quantile_stats_rejects_missing_role(
    tmp_path: Path,
    role: str,
) -> None:
    path = tmp_path / "quantile_stats.json"
    _write_quantile_valid(path)
    _rewrite_quantile_artifact(
        path,
        lambda raw: raw["normalization"].pop(role),
    )

    with pytest.raises(ValueError, match="normalization.*malformed"):
        _load_quantile(path)


@pytest.mark.parametrize("role", ["predict_action", "observation_state"])
def test_xtokenizer_quantile_stats_rejects_nonpositive_quantile_range(
    tmp_path: Path,
    role: str,
) -> None:
    path = tmp_path / "quantile_stats.json"
    _write_quantile_valid(path)

    def _make_degenerate(raw: dict[str, Any]) -> None:
        raw["normalization"][role]["q001"][0] = raw["normalization"][role]["q999"][0]

    _rewrite_quantile_artifact(path, _make_degenerate)

    with pytest.raises(ValueError, match=r"q001 < q999.*every.*channel"):
        _load_quantile(path)


def test_xtokenizer_quantile_stats_rejects_changed_horizon(tmp_path: Path) -> None:
    path = tmp_path / "quantile_stats.json"
    _write_quantile_valid(path)

    with pytest.raises(ValueError, match="stats_horizon mismatch"):
        _load_quantile(path, expected_stats_horizon=16)


@pytest.mark.parametrize("invalid_horizon", [True, 32.0, "32", 0, -1])
def test_xtokenizer_quantile_stats_require_exact_positive_integer_horizons(
    tmp_path: Path,
    invalid_horizon: object,
) -> None:
    path = tmp_path / "quantile_stats.json"
    _write_quantile_valid(path)

    with pytest.raises(ValueError, match="positive integer"):
        XTokenizerQuantileStatistics.from_path(
            path,
            expected_dataset_repo_id="org/bridge",
            expected_dataset_revision=DATASET_REVISION,
            expected_robot_type="WidowX",
            expected_dof_dims=DOF_DIMS,
            expected_dof_order=DOF_ORDER,
            expected_action_dim=10,
            expected_stats_horizon=cast(Any, invalid_horizon),
        )


def test_xtokenizer_quantile_stats_rejects_swapped_spaces(tmp_path: Path) -> None:
    path = tmp_path / "quantile_stats.json"
    _write_quantile_valid(path)

    def _swap_spaces(raw: dict[str, Any]) -> None:
        predict = raw["normalization"]["predict_action"]
        observation = raw["normalization"]["observation_state"]
        predict["space"], observation["space"] = (
            observation["space"],
            predict["space"],
        )

    _rewrite_quantile_artifact(path, _swap_spaces)

    with pytest.raises(ValueError, match="space"):
        _load_quantile(path)


def test_xtokenizer_quantile_stats_rejects_changed_fingerprint(
    tmp_path: Path,
) -> None:
    path = tmp_path / "quantile_stats.json"
    _write_quantile_valid(path)
    _rewrite_quantile_artifact(
        path,
        lambda raw: raw.update(content_fingerprint="0" * 64),
        refresh_fingerprint=False,
    )

    with pytest.raises(ValueError, match="fingerprint"):
        _load_quantile(path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda raw: raw["normalization"]["predict_action"].update(q001=[0.0] * 9),
            "dimension",
        ),
        (
            lambda raw: raw["normalization"]["observation_state"].update(
                q999=[4.0] * 9 + [float("inf")]
            ),
            "finite",
        ),
    ],
)
def test_xtokenizer_quantile_stats_rejects_malformed_vectors(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    path = tmp_path / "quantile_stats.json"
    _write_quantile_valid(path)
    _rewrite_quantile_artifact(
        path,
        mutate,
        refresh_fingerprint=False,
    )

    with pytest.raises(ValueError, match=message):
        _load_quantile(path)


@pytest.mark.parametrize("invalid_dimension", ["1", True, 1.5])
def test_xtokenizer_quantile_stats_reject_loaded_noninteger_dimensions(
    tmp_path: Path,
    invalid_dimension: object,
) -> None:
    path = tmp_path / "quantile_stats.json"
    _write_quantile_valid(path)
    _rewrite_quantile_artifact(
        path,
        lambda raw: raw["action_layout"]["dof_dims"].update(
            follow_left_gripper=invalid_dimension
        ),
    )

    with pytest.raises(ValueError, match="positive integer"):
        _load_quantile(path)


@pytest.mark.parametrize("invalid_dimension", ["1", True, 1.5])
def test_xtokenizer_quantile_stats_reject_writer_noninteger_dimensions(
    tmp_path: Path,
    invalid_dimension: object,
) -> None:
    dims = dict(DOF_DIMS)
    dims["follow_left_gripper"] = cast(Any, invalid_dimension)
    path = tmp_path / "quantile_stats.json"

    with pytest.raises(ValueError, match="positive integer"):
        write_xtokenizer_quantile_statistics(
            path,
            dataset_repo_id="org/bridge",
            dataset_revision=DATASET_REVISION,
            robot_type="WidowX",
            dof_dims=dims,
            dof_order=DOF_ORDER,
            stats_horizon=32,
            predict_action_statistics=PREDICT_QUANTILES,
            observation_state_statistics=OBSERVATION_QUANTILES,
        )

    assert not path.exists()


@pytest.mark.parametrize("invalid_dimension", ["1", True, 1.5])
def test_xtokenizer_quantile_stats_reject_expected_noninteger_dimensions(
    tmp_path: Path,
    invalid_dimension: object,
) -> None:
    path = tmp_path / "quantile_stats.json"
    _write_quantile_valid(path)
    expected_dims = dict(DOF_DIMS)
    expected_dims["follow_left_gripper"] = cast(Any, invalid_dimension)

    with pytest.raises(ValueError, match="positive integer"):
        XTokenizerQuantileStatistics.from_path(
            path,
            expected_dataset_repo_id="org/bridge",
            expected_dataset_revision=DATASET_REVISION,
            expected_robot_type="WidowX",
            expected_dof_dims=expected_dims,
            expected_dof_order=DOF_ORDER,
            expected_action_dim=10,
            expected_stats_horizon=32,
        )


@pytest.mark.parametrize("invalid_action_dim", ["10", True, 10.5])
def test_xtokenizer_quantile_stats_reject_invalid_expected_action_dim(
    tmp_path: Path,
    invalid_action_dim: object,
) -> None:
    path = tmp_path / "quantile_stats.json"
    _write_quantile_valid(path)

    with pytest.raises(ValueError, match="expected_action_dim.*positive integer"):
        XTokenizerQuantileStatistics.from_path(
            path,
            expected_dataset_repo_id="org/bridge",
            expected_dataset_revision=DATASET_REVISION,
            expected_robot_type="WidowX",
            expected_dof_dims=DOF_DIMS,
            expected_dof_order=DOF_ORDER,
            expected_action_dim=cast(Any, invalid_action_dim),
            expected_stats_horizon=32,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda raw: raw["normalization"]["predict_action"]["q001"].__setitem__(
                0, "-1.0"
            ),
            "real numbers",
        ),
        (
            lambda raw: raw["normalization"]["observation_state"]["q999"].__setitem__(
                0, True
            ),
            "real numbers",
        ),
        (
            lambda raw: raw["normalization"]["predict_action"].update(
                q001={str(value): 0 for value in PREDICT_QUANTILES["q001"]}
            ),
            "JSON array",
        ),
    ],
)
def test_xtokenizer_quantile_stats_require_json_numeric_arrays(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    path = tmp_path / "quantile_stats.json"
    _write_quantile_valid(path)
    _rewrite_quantile_artifact(path, mutate)

    with pytest.raises(ValueError, match=message):
        _load_quantile(path)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [("q001", "-1.0"), ("q999", True)],
)
def test_xtokenizer_quantile_stats_writer_rejects_nonreal_values(
    tmp_path: Path,
    field: str,
    invalid_value: object,
) -> None:
    predict = {name: list(values) for name, values in PREDICT_QUANTILES.items()}
    predict[field][0] = cast(Any, invalid_value)

    with pytest.raises(ValueError, match="real numbers"):
        write_xtokenizer_quantile_statistics(
            tmp_path / "quantile_stats.json",
            dataset_repo_id="org/bridge",
            dataset_revision=DATASET_REVISION,
            robot_type="WidowX",
            dof_dims=DOF_DIMS,
            dof_order=DOF_ORDER,
            stats_horizon=32,
            predict_action_statistics=predict,
            observation_state_statistics=OBSERVATION_QUANTILES,
        )


def test_canonical_stats_round_trip_separates_relative_actions_from_absolute_anchor(
    tmp_path: Path,
) -> None:
    path = tmp_path / "canonical_stats.json"
    _write_valid(path)

    artifact = _load(path)

    assert artifact.mode is NormMode.MIN_MAX
    assert artifact.action_dim == 10
    assert artifact.dof_order == DOF_ORDER
    assert artifact.action_representation == "observation_relative"
    assert artifact.observation_representation == "absolute"
    assert artifact.public_action_representation == "absolute"
    assert artifact.relative_reference == "single_pre_chunk_observation"
    assert artifact.quantile_range == pytest.approx((0.001, 0.999))
    assert artifact.normalization_modes("action") == {
        "action": NormMode.MIN_MAX,
        "observation.state": NormMode.MIN_MAX,
    }
    stats = artifact.normalization_stats("action")
    assert set(stats) == {"action", "observation.state"}
    for name, values in ACTION_STATISTICS.items():
        torch.testing.assert_close(
            stats["action"][name],
            torch.tensor(values, dtype=torch.float32),
        )
    for name, values in OBSERVATION_STATISTICS.items():
        torch.testing.assert_close(
            stats["observation.state"][name],
            torch.tensor(values, dtype=torch.float32),
        )


def test_canonical_stats_publication_is_idempotent_without_replacing_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "canonical_stats.json"
    _write_valid(path)
    original_bytes = path.read_bytes()
    original_inode = path.stat().st_ino

    _write_valid(path)

    assert path.read_bytes() == original_bytes
    assert path.stat().st_ino == original_inode


def test_canonical_stats_publication_refuses_different_existing_content(
    tmp_path: Path,
) -> None:
    path = tmp_path / "canonical_stats.json"
    _write_valid(path)
    original_bytes = path.read_bytes()
    different_statistics = {
        name: list(values) for name, values in ACTION_STATISTICS.items()
    }
    different_statistics["min"][0] = -0.25

    with pytest.raises(ValueError, match="already exists.*different"):
        write_canonical_action_statistics(
            path,
            dataset_repo_id="org/bridge",
            dataset_revision=DATASET_REVISION,
            dof_dims=DOF_DIMS,
            dof_order=DOF_ORDER,
            mode=NormMode.MIN_MAX,
            action_statistics=different_statistics,
            observation_statistics=OBSERVATION_STATISTICS,
        )

    assert path.read_bytes() == original_bytes


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda raw: raw.update(dataset_repo_id="other/bridge"),
            "dataset repository",
        ),
        (
            lambda raw: raw.update(dataset_revision="b" * 40),
            "dataset revision",
        ),
        (
            lambda raw: raw["action_layout"].update(
                dof_order=list(reversed(DOF_ORDER))
            ),
            "action layout",
        ),
        (
            lambda raw: raw["statistics"]["action"].update(min=[0.0] * 9),
            "dimension",
        ),
        (
            lambda raw: raw["statistics"]["observation.state"].update(
                max=[2.0] * 9 + [float("inf")]
            ),
            "finite",
        ),
        (
            lambda raw: raw["representations"].update(action="temporal_delta"),
            "representation",
        ),
        (
            lambda raw: raw.update(content_fingerprint="0" * 64),
            "fingerprint",
        ),
    ],
)
def test_canonical_stats_fail_closed_on_contract_mismatch(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    path = tmp_path / "canonical_stats.json"
    _write_valid(path)
    raw = json.loads(path.read_text())
    mutate(raw)
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match=message):
        _load(path)


@pytest.mark.parametrize("revision", ["main", "g" * 40, "a" * 39])
def test_canonical_stats_require_immutable_dataset_sha(
    tmp_path: Path,
    revision: str,
) -> None:
    with pytest.raises(ValueError, match="40-character hexadecimal"):
        write_canonical_action_statistics(
            tmp_path / "canonical_stats.json",
            dataset_repo_id="org/bridge",
            dataset_revision=revision,
            dof_dims=DOF_DIMS,
            dof_order=DOF_ORDER,
            mode=NormMode.MIN_MAX,
            action_statistics=ACTION_STATISTICS,
            observation_statistics=OBSERVATION_STATISTICS,
        )


def test_canonical_stats_rejects_old_absolute_action_artifact(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "dataset_repo_id": "org/bridge",
                "dataset_revision": DATASET_REVISION,
                "action_layout": {
                    "convention": "absolute",
                    "dof_dims": DOF_DIMS,
                    "dof_order": list(DOF_ORDER),
                },
                "mode": "min_max",
                "statistics": {
                    "min": [-1.0] * 10,
                    "max": [1.0] * 10,
                    "mean": [0.0] * 10,
                    "std": [1.0] * 10,
                },
                "content_fingerprint": "0" * 64,
            }
        )
    )

    with pytest.raises(ValueError, match="format_version|fields"):
        _load(path)
