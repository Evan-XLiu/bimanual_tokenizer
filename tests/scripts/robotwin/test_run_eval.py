# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.robotwin.run_eval import (
    build_eval_command,
    run_task_eval,
)


class _FakeEvaluatorProcess:
    def __init__(self, command: list[str], **_kwargs: object) -> None:
        self.stdout = iter(())
        assert "--resume" in command
        output = Path(command[command.index("--output-dir") + 1])
        task = command[command.index("--task") + 1]
        manifest_path = output / task / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["status"] = "rollout_completed"
        manifest_path.write_text(json.dumps(manifest))

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return 0

    def poll(self) -> int:
        return 0


def _write_runtime_inputs(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "RoboTwin"
    entrypoint = root / "script" / "eval_policy.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("# evaluator\n")
    plugin = tmp_path / "policy" / "praxis_ar"
    plugin.mkdir(parents=True)
    for name in ("__init__.py", "deploy_policy.py", "deploy_policy.yml"):
        (plugin / name).write_text("# plugin\n")
    return root, plugin


def test_build_eval_command_carries_auditable_episode_contract(tmp_path: Path) -> None:
    robotwin_root = tmp_path / "RoboTwin"
    command = build_eval_command(
        robotwin_root=robotwin_root,
        task_name="pick_apple",
        task_config="demo_clean",
        episodes_per_task=10,
        record_episodes_per_task=10,
        seed=42,
        policy_host="policy-server",
        policy_port=50051,
        output_dir=tmp_path / "eval",
        run_id="smoke",
        expected_checkpoint=Path("/checkpoints/step_10/pretrained_model"),
        plugin_config=Path("/repo/scripts/robotwin/policy/praxis_ar/deploy_policy.yml"),
        python_executable="python",
        evaluator_overlay=Path("/repo/scripts/robotwin/stable2_evaluator.py"),
    )

    rendered = " ".join(command)
    assert command[:3] == [
        "python",
        "-u",
        "/repo/scripts/robotwin/stable2_evaluator.py",
    ]
    assert "--episodes-per-task 10" in rendered
    assert "--record-episodes-per-task 10" in rendered
    assert "--task-config demo_clean" in rendered
    assert "--policy-host policy-server" in rendered
    assert "--run-id smoke" in rendered
    assert "--max-seed-attempts 500" in rendered
    assert (
        "--expected-checkpoint /checkpoints/step_10/pretrained_model" in rendered
    )
    assert (
        "--plugin-config /repo/scripts/robotwin/policy/praxis_ar/deploy_policy.yml"
        in rendered
    )


def test_build_eval_command_rejects_partial_recording(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="record_episodes_per_task"):
        build_eval_command(
            robotwin_root=tmp_path / "RoboTwin",
            task_name="pick_apple",
            task_config="demo_clean",
            episodes_per_task=10,
            record_episodes_per_task=2,
            seed=42,
            policy_host="localhost",
            policy_port=50051,
            output_dir=tmp_path / "eval",
            run_id="smoke",
            expected_checkpoint=Path("/checkpoints/step_10/pretrained_model"),
            plugin_config=Path(
                "/repo/scripts/robotwin/policy/praxis_ar/deploy_policy.yml"
            ),
            python_executable="python",
            evaluator_overlay=Path("/repo/scripts/robotwin/stable2_evaluator.py"),
        )


def test_run_eval_supports_direct_script_help() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/robotwin/run_eval.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--expected-checkpoint" in result.stdout


def test_parent_marks_completed_only_after_media_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, plugin = _write_runtime_inputs(tmp_path)
    monkeypatch.setattr(
        "scripts.robotwin.run_eval.subprocess.Popen", _FakeEvaluatorProcess
    )
    monkeypatch.setattr(
        "scripts.robotwin.run_eval.verify_task_eval",
        lambda **_kwargs: {"media_complete": True},
    )

    result = run_task_eval(
        robotwin_root=root,
        task_name="pick_apple",
        task_config="demo_clean",
        episodes_per_task=1,
        record_episodes_per_task=1,
        seed=0,
        policy_host="localhost",
        policy_port=50051,
        output_dir=tmp_path / "eval",
        run_id="smoke",
        expected_checkpoint=tmp_path / "checkpoint",
        plugin_source=plugin,
    )

    manifest = json.loads(
        (tmp_path / "eval" / "pick_apple" / "run_manifest.json").read_text()
    )
    assert result["media_complete"] is True
    assert manifest["status"] == "completed"
    assert manifest["verification"] == {"media_complete": True}


def test_parent_marks_failed_when_media_verification_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, plugin = _write_runtime_inputs(tmp_path)
    monkeypatch.setattr(
        "scripts.robotwin.run_eval.subprocess.Popen", _FakeEvaluatorProcess
    )
    monkeypatch.setattr(
        "scripts.robotwin.run_eval.verify_task_eval",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("video is truncated")),
    )

    with pytest.raises(ValueError, match="video is truncated"):
        run_task_eval(
            robotwin_root=root,
            task_name="pick_apple",
            task_config="demo_clean",
            episodes_per_task=1,
            record_episodes_per_task=1,
            seed=0,
            policy_host="localhost",
            policy_port=50051,
            output_dir=tmp_path / "eval",
            run_id="smoke",
            expected_checkpoint=tmp_path / "checkpoint",
            plugin_source=plugin,
        )

    manifest = json.loads(
        (tmp_path / "eval" / "pick_apple" / "run_manifest.json").read_text()
    )
    assert manifest["status"] == "failed"
    assert manifest["failure"]["stage"] == "verification"


def test_parent_preserves_manifest_for_early_preflight_failure(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Official RoboTwin evaluator"):
        run_task_eval(
            robotwin_root=tmp_path / "missing-robotwin",
            task_name="pick_apple",
            task_config="demo_clean",
            episodes_per_task=1,
            record_episodes_per_task=1,
            seed=0,
            policy_host="localhost",
            policy_port=50051,
            output_dir=tmp_path / "eval",
            run_id="smoke",
            expected_checkpoint=tmp_path / "checkpoint",
        )

    manifest = json.loads(
        (tmp_path / "eval" / "pick_apple" / "run_manifest.json").read_text()
    )
    assert manifest["status"] == "failed"
    assert manifest["failure"]["stage"] == "preflight"
